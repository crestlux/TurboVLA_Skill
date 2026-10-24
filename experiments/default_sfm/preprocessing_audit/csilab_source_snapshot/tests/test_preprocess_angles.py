"""Euler observation wrapping must match offline/online inputs without changing actions."""
import contextlib
import copy
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import h5py
import numpy as np

from preprocess.convert import PreprocessError, StateDecoder, run, transform_observation, validate_config
from preprocess.inference import InferencePreprocessor
from preprocess.normalization import transform
from test_preprocess import COLLECTION, observation, settings, transitions


class AngleRangeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def rows(self):
        rows = transitions([.2, .21, .22, .23])
        angles = [(np.pi - .01, -.2, -1.), (-np.pi + .01, -.1, -.5),
                  (-np.pi + .02, .1, 0.), (-np.pi + .03, .2, .5)]
        for row, angles_at_t in zip(rows, angles):
            row['observations']['state'][0, 7:10] = angles_at_t
            row['next_observations']['state'][0, 7:10] = np.asarray(angles_at_t) + .005
        return rows

    def configuration(self, wrap=True, normalization='none'):
        cfg = settings()
        cfg['crop_initial_stationary']['enabled'] = False
        cfg['actions'] = dict(mode='delta', length=2, source_format='absolute')
        cfg['normalization'] = dict(state={'method': normalization}, action={'method': 'none'})
        pose = dict(key='right/tcp_pose', components=['yaw', 'x', 'roll', 'pitch'])
        if wrap:
            pose['wrap_angles'] = ['roll', 'yaw']
        cfg['observations'][0]['keys'] = ['right/gripper_pose', pose]
        return cfg

    def convert(self, cfg, rows, name):
        source = self.root / 'input.pkl'
        source.write_bytes(pickle.dumps(dict(metadata={'config': COLLECTION}, transitions=rows)))
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = run(cfg, self.root / 'config.yaml', input_path=source, output_dir=output)
        return output, Path(manifest[0]['output'])

    def test_named_axes_reordered_components_and_input_not_mutated(self):
        decoder = StateDecoder(COLLECTION)
        row = self.rows()[1]['observations']
        original = copy.deepcopy(row['state'])
        entry = self.configuration()['observations'][0]
        result = transform_observation(row, [entry], decoder)['state']
        # [gripper, yaw, x, roll, pitch], only yaw/roll are wrapped.
        np.testing.assert_allclose(result, [1., 2*np.pi - .5, .21, np.pi + .01, -.1], atol=5e-7)
        np.testing.assert_array_equal(row['state'], original)
        self.assertAlmostEqual(decoder.channel(row, 'right/tcp_pose')[3], -np.pi + .01, places=6)

    def test_left_arm_dict_state_boolean_and_integer_component_selection(self):
        collection = {'train': {'arm_type': 'left', 'proprio_keys': ['left/tcp_pose']}}
        decoder = StateDecoder(collection)
        source = {'state': {'left/tcp_pose': np.array([[-.3, .2, .1, -3., -.2, -1.]])}}
        for components in ({'roll': True, 'x': True}, [3, 0]):
            item = dict(key='left/tcp_pose', components=components, wrap_angles=['roll'])
            actual = decoder.selected(source, item, 'state')
            expected = [-.3, 2*np.pi - 3.] if isinstance(components, dict) else [2*np.pi - 3., -.3]
            np.testing.assert_allclose(actual, expected)

    def test_open_upper_bound_and_new_boundary_at_zero(self):
        decoder = StateDecoder(COLLECTION)
        item = dict(key='right/tcp_pose', components=['yaw'], wrap_angles=['yaw'])
        for angle in (-2*np.pi, -np.pi, 0., np.pi, 2*np.pi, -1e-12):
            obs = observation(yaw=angle)
            entry = dict(type='state', source='state', target='state', keys=[item])
            value = transform_observation(obs, [entry], decoder)['state'][0]
            self.assertGreaterEqual(value, 0.)
            self.assertLess(float(value), 2*np.pi)
        a = decoder.selected(observation(yaw=-.01), item, 'state')[0]
        b = decoder.selected(observation(yaw=.01), item, 'state')[0]
        self.assertGreater(abs(a - b), 6.)  # Changing interval moves the discontinuity.

    def test_offline_next_observation_online_normalization_and_unchanged_actions(self):
        rows = self.rows()
        for normalization in ('none', 'standard'):
            cfg = self.configuration(normalization=normalization)
            output, path = self.convert(cfg, rows, 'wrapped-' + normalization)
            _, baseline_path = self.convert(self.configuration(wrap=False, normalization=normalization), rows,
                                            'original-' + normalization)
            preprocessor = InferencePreprocessor(output)
            with h5py.File(path) as file, h5py.File(baseline_path) as baseline:
                for t, row in enumerate(rows):
                    online = preprocessor.observation(row['observations'])
                    next_online = preprocessor.observation(row['next_observations'])
                    np.testing.assert_allclose(online['state'], file['observations/state'][t], atol=1e-6)
                    np.testing.assert_allclose(next_online['state'], file['next_observations/state'][t], atol=1e-6)
                stats = preprocessor.stats['observations']['state']
                physical = transform(file['observations/state'][:], stats, inverse=True)
                self.assertLess(abs(physical[1, 3] - physical[0, 3]), .03)  # roll crosses +/-pi smoothly.
                self.assertTrue(np.all((physical[:, 1] >= -1e-6) & (physical[:, 1] < 2*np.pi)))
                for key in ('actions', 'delta_actions', 'action_reference_poses', 'source_indices'):
                    np.testing.assert_array_equal(file[key][:], baseline[key][:])
                metadata = json.loads(file['metadata/inference_config'][()])
                self.assertEqual(metadata['observations'][0]['keys'][1]['wrap_angles'], ['roll', 'yaw'])
                self.assertEqual(json.loads(file['observations/state'].attrs['columns']),
                    ['right/gripper_pose/pos', 'right/tcp_pose/yaw', 'right/tcp_pose/x',
                     'right/tcp_pose/roll', 'right/tcp_pose/pitch'])

    def test_statistics_from_original_range_cannot_be_reused(self):
        rows = self.rows()
        output, _ = self.convert(self.configuration(wrap=False, normalization='standard'), rows, 'baseline')
        cfg = self.configuration(normalization='standard')
        cfg['normalization']['stats_file'] = str(output / 'normalization.json')
        with self.assertRaisesRegex(ValueError, 'schema does not match'):
            with contextlib.redirect_stdout(io.StringIO()):
                run(cfg, self.root / 'config.yaml', input_path=self.root / 'input.pkl',
                    output_dir=self.root / 'new')

    def test_invalid_wrap_options_and_unselected_components_are_rejected(self):
        for wrap in ('roll', ['x'], ['roll', 'roll'], [True], None):
            cfg = self.configuration()
            cfg['observations'][0]['keys'][1]['wrap_angles'] = wrap
            with self.subTest(wrap=wrap), self.assertRaisesRegex(PreprocessError, 'wrap_angles'):
                validate_config(cfg)
        cfg = self.configuration()
        cfg['observations'][0]['keys'][1] = dict(key='right/tcp_vel', components=['wx'], wrap_angles=['roll'])
        with self.assertRaisesRegex(PreprocessError, 'only for tcp_pose'):
            validate_config(cfg)
        decoder = StateDecoder(COLLECTION)
        with self.assertRaisesRegex(PreprocessError, 'retained rotation'):
            decoder.columns([dict(key='right/tcp_pose', components=['x'], wrap_angles=['yaw'])])
        # An absent/empty option keeps legacy data exactly as before.
        raw = decoder.selected(observation(yaw=-1.), 'right/tcp_pose', 'state')
        unchanged = decoder.selected(observation(yaw=-1.),
            dict(key='right/tcp_pose', components=[0, 1, 2, 3, 4, 5], wrap_angles=[]), 'state')
        np.testing.assert_array_equal(raw, unchanged)


if __name__ == '__main__':
    unittest.main()
