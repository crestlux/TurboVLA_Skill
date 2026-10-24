"""Reuse recorded preprocessing and normalization without fitting on inference inputs."""
import json
from pathlib import Path

import h5py
import numpy as np

from .convert import StateDecoder, transform_observation, latest
from .normalization import transform


class InferencePreprocessor:
    def __init__(self, bundle):
        path = Path(bundle)
        if path.is_file() and path.suffix in ('.h5', '.hdf5'):
            with h5py.File(path) as f:
                self.config = json.loads(f['metadata/inference_config'][()])
                self.stats = json.loads(f['metadata/normalization'][()])
        else:
            self.config = json.loads((path/'inference_config.json').read_text())
            self.stats = json.loads((path/self.config['normalization_file']).read_text())
        self.decoder = StateDecoder(self.config['collection_config'])
        expected_columns = {e['target']: self.decoder.columns(e['keys'])
                            for e in self.config['observations'] if e['type'] == 'state'}
        action = dict(self.config['actions'])
        action.pop('declared_formats', None)
        signature = self.stats.get('schema', {})
        schema_action = dict(signature.get('action', {}))
        schema_action.pop('declared_formats', None)
        if (self.config.get('version') != 1 or self.stats.get('version') != 1 or
                signature.get('observations') != self.config['observations'] or
                signature.get('state_columns') != expected_columns or schema_action != action):
            raise ValueError('Inference config and normalization statistics do not match')

    def observation(self, observation, collection_config=None):
        decoder = StateDecoder(collection_config) if collection_config else self.decoder
        if self.config.get('jack_source_image_resize'):
            import cv2
            height, width = self.config['jack_source_image_resize']
            observation = dict(observation)
            for entry in self.config['observations']:
                if entry['type'] == 'image':
                    frame = latest(observation[entry['source']], 3)
                    observation[entry['source']] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        result = transform_observation(observation, self.config['observations'], decoder)
        for key, coefficients in self.stats['observations'].items():
            result[key] = transform(result[key], coefficients).astype(np.float32)
        return result

    def normalize_actions(self, values):
        return transform(values, self.stats['actions'])

    @property
    def uses_opening_gripper(self):
        gripper = self.config['actions'].get('gripper')
        return isinstance(gripper, dict) and gripper.get('label_version') == 2

    def execution_config(self, config):
        """Keep the packaged legacy gripper commands in their trained units."""
        if self.uses_opening_gripper:
            raise ValueError('This Jack runtime supports legacy gripper labels only; '
                             'use the current project preprocessor for label_version=2 bundles')
        return config

    def denormalize_actions(self, values):
        """Restore trained labels; relative/binary grippers still need decode_gripper_actions."""
        return transform(values, self.stats['actions'], inverse=True)

    def decode_gripper_actions(self, values, previous_commands=None, *, observed_gripper=None):
        """Use raw measured openings for state-relative, last targets for binary/legacy."""
        gripper = self.config['actions'].get('gripper')
        if not isinstance(gripper, dict):
            return np.asarray(values, dtype=float).copy()
        from .gripper import decode
        return decode(values, gripper, previous_commands, observed_gripper=observed_gripper)

    def interpreter_options(self, *, decoded_gripper=False):
        action = self.config['actions']
        if action['mode'] == 'raw':
            raise ValueError('This legacy bundle has no policy chunk convention')
        gripper = action.get('gripper')
        if isinstance(gripper, dict) and gripper['mode'] != 'absolute' and not decoded_gripper:
            raise ValueError('Use decode_gripper_actions with observed_gripper for next_state relative labels '
                             'or previous_commands for binary/legacy labels, then interpreter_options(decoded_gripper=True)')
        return dict(mode=action['interpreter_mode'], policy_sides=tuple(action['sides']),
                    relative_gripper_action_mode=action['relative_gripper_action_mode'],
                    relative_action_frame_yaw_deg=action['relative_action_frame_yaw_deg'],
                    action_period_s=self.config['action_period_s'])
