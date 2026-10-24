"""Semantic regression tests for SFM preprocessing; no optimizer or training."""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from turbovla.data.sfm_preprocessing import (
    Moments, chunk_actions, delta_actions, held_targets, initial_start,
    normalize, normalized_actions, preprocess_image, state_physical, variant_spec,
)
from preprocess import references


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = yaml.safe_load((ROOT / "experiments/default_sfm/experiment_plan.yaml").read_text())
        cls.reference = references()
        source = ROOT / "experiments/default_sfm/preprocessing_audit/csilab_source_snapshot/preprocess/normalization.py"
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {"Moments", "transform"}]
        cls.norm_reference = {"np": np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), cls.norm_reference)

    def test_held_preserves_command_while_observed_gripper_moves(self):
        state = np.zeros((5, 19), np.float32)
        state[:, 0] = [1., .8, .5, .3, .7]
        rows = []
        for i, percent in enumerate([100., 0., None, None, 100.]):
            receipt = {} if percent is None else {"gripper_target_percent": percent}
            rows.append({"observations": {"state": state[i][None]}, "actions": np.zeros(7),
                         "infos": {"command_receipts": {"right": receipt}}})
        labels, _, present, fallback = held_targets(rows, state, {"mode": "binary"})
        np.testing.assert_array_equal(labels, [1, -1, -1, -1, 1])
        self.assertEqual(fallback, 2)
        self.assertEqual(present.sum(), 3)
        rows[2]["infos"]["command_receipts"]["right"] = {"gripper_target_percent": 25}
        with self.assertRaisesRegex(ValueError, "Nonbinary held"):
            held_targets(rows, state, {"mode": "binary"})

    def test_world_delta_uses_each_action_timestep(self):
        state = np.zeros((4, 19), np.float32)
        state[:, 4:7] = [[1, 2, 3], [2, 4, 6], [4, 8, 12], [8, 16, 24]]
        state[:, 7:10] = [[.4, -.6, 1.2], [-.3, .5, -.8], [2., .1, 1.], [-2., .4, -.9]]
        base = Rotation.from_euler("xyz", state[:, 7:10])
        world_delta = np.array([[.1, -.2, .3], [-.2, .1, -.1], [.2, .4, -.3], [.3, -.1, .2]])
        target = np.zeros((4, 7))
        target[:, :3] = state[:, 4:7] + np.array([.01, -.02, .03])
        target[:, 3:6] = (Rotation.from_rotvec(world_delta) * base).as_euler("xyz")
        delta = delta_actions(state, target, np.array([-1., 1., 1., -1.]))
        np.testing.assert_allclose(delta[:, :3], np.tile([.01, -.02, .03], (4, 1)), atol=1e-12)
        np.testing.assert_allclose(delta[:, 3:6], world_delta, atol=1e-12)
        chunks, mask = chunk_actions(delta, 5)
        np.testing.assert_allclose(chunks[0, :4], delta)
        np.testing.assert_array_equal(mask.sum(axis=1), [4, 3, 2, 1])
        self.assertTrue(np.all(chunks[~mask] == 0))

    def test_selective_angle_wrap_keeps_pitch_and_open_upper_bound(self):
        state = np.zeros((4, 19), np.float32)
        state[:, 7] = [-np.pi, np.pi, -1e-8, 0]
        state[:, 8] = [-.2, .2, -.1, .1]
        state[:, 9] = [-.2, .2, -np.pi, np.pi]
        wrapped = state_physical(state)
        self.assertEqual(wrapped.shape, (4, 13))
        np.testing.assert_array_equal(wrapped[:, 5], state[:, 8])
        self.assertTrue(np.all(wrapped[:, [4, 6]] >= 0))
        self.assertTrue(np.all(wrapped[:, [4, 6]] < np.float32(2 * np.pi)))
        self.assertEqual(wrapped[2, 4], np.nextafter(np.float32(2 * np.pi), np.float32(0)))

    def test_initial_trim_rejects_one_frame_spike_and_keeps_preroll(self):
        state = np.zeros((12, 19), np.float32)
        state[2, 4] = .02
        state[7:, 4] = .02
        start, motion = initial_start(state, self.plan["dataset"]["frame_selection"]["detection_config"])
        self.assertEqual((start, motion), (5, 7))

    def test_normal_zero_ft_and_std_floor_match_csilab(self):
        values = np.zeros((20, 50, 6), np.float32)
        values[18:] = np.array([.001, -.002, .003, .0001, .0002, -.0003])
        ours, ref = Moments(), self.norm_reference["Moments"]()
        for block in np.array_split(values, 3):
            ours.add(block); ref.add(block)
        stats = ours.result("standard", .01)
        ref_stats = ref.result({"method": "standard", "epsilon": .01})
        self.assertEqual(stats, ref_stats)
        self.assertEqual(stats["count"], 1000)
        self.assertEqual(stats["scale"], [.01] * 6)
        actual = normalize(values, stats)
        np.testing.assert_array_equal(actual, self.norm_reference["transform"](values, ref_stats))
        np.testing.assert_allclose(normalize(actual, stats, inverse=True), values, atol=1e-12)
        self.assertFalse(np.all(actual[0] == 0))

    def test_action_padding_is_zero_after_normalization(self):
        values = np.arange(7 * 7, dtype=float).reshape(7, 7) / 100
        values[:, 6] = [1, 1, -1, -1, -1, 1, 1]
        chunks, mask = chunk_actions(values)
        moments = Moments(); moments.add(chunks[mask][:, :6])
        stats = moments.result("symlog_minmax", 1e-6)
        actual = normalized_actions(chunks, mask, stats)
        expected = self.norm_reference["transform"](chunks[..., :6], stats)
        np.testing.assert_allclose(actual[..., :6][mask], expected[mask], atol=1e-7)
        self.assertTrue(np.all(actual[~mask] == 0))
        np.testing.assert_array_equal(actual[..., 6][mask], chunks[..., 6][mask])

    def test_native_camera_pixels_match_csilab(self):
        image = np.random.default_rng(42).integers(0, 256, (1, 480, 640, 3), dtype=np.uint8)
        for camera, crop in self.plan["vision"]["crop_tlhw"].items():
            actual = preprocess_image(image, camera, self.plan["vision"])
            expected = self.reference["image_sample"](image, {"crop": crop, "resize": [224, 224], "resize_mode": "letterbox", "interpolation": "area"})
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(np.all(actual[:28] == 0) and np.all(actual[-28:] == 0))
            self.assertGreater(actual[28:196].std(), 0)

    def test_variants_remove_only_requested_state_axes(self):
        dimensions = []
        for v in self.plan["variants"]:
            spec = variant_spec(self.plan, v["id"])
            dimensions.append(spec["model_state_dim"])
            if spec["state_condition"] == "xy_vxy_remove":
                self.assertEqual(spec["state_columns"], [0, 3, 4, 5, 6, 9, 10, 11, 12])
        self.assertEqual(dimensions, [313, 311, 309, 13, 11, 9, 13, 11, 9])


if __name__ == "__main__":
    unittest.main()
