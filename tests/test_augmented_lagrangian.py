import copy
import tempfile
import unittest
from pathlib import Path

import torch

from molsteer.molexecutor.augmented_lagrangian_reward import (
    AugmentedLagrangianController,
    AugmentedLagrangianReward,
)


def constraints():
    return [
        {
            "id": "geometry",
            "kind": "local_geometry_advantage",
            "penalty": 2.0,
            "lambda_initial": 0.5,
            "lambda_max": 10.0,
            "satisfaction_tolerance": 0.01,
            "satisfaction_frames": 2,
            "satisfied_decay": 0.5,
        },
        {
            "id": "strain",
            "kind": "local_strain_advantage",
            "penalty": 1.0,
            "lambda_initial": 0.0,
            "lambda_max": 10.0,
            "satisfaction_tolerance": 0.0,
            "satisfaction_frames": 3,
            "satisfied_decay": 1.0,
        },
    ]


class AugmentedLagrangianControllerTests(unittest.TestCase):
    def test_violation_raises_multiplier_and_persistent_repair_decays_it(self):
        controller = AugmentedLagrangianController("rp_test", constraints())
        first = controller.update({"geometry": .25, "strain": .4}, 1)
        self.assertTrue(first["updated"])
        self.assertAlmostEqual(controller.duals["geometry"], 1.0)
        self.assertAlmostEqual(controller.duals["strain"], .4)
        controller.update({"geometry": 0.0, "strain": .1}, 2)
        self.assertAlmostEqual(controller.duals["geometry"], 1.0)
        controller.update({"geometry": 0.0, "strain": .1}, 3)
        self.assertAlmostEqual(controller.duals["geometry"], .5)

    def test_same_step_is_idempotent(self):
        controller = AugmentedLagrangianController("rp_test", constraints())
        controller.update({"geometry": .25, "strain": .4}, 4)
        before = copy.deepcopy(controller.state_dict())
        event = controller.update({"geometry": .8, "strain": .9}, 4)
        self.assertFalse(event["updated"])
        self.assertEqual(controller.state_dict(), before)

    def test_dual_state_checkpoint_roundtrip_continues_exactly(self):
        first = AugmentedLagrangianController("rp_test", constraints())
        first.update({"geometry": .25, "strain": .4}, 1)
        payload = {
            "arm": "creativity",
            "program_id": "rp_test",
            "path_used": torch.tensor([.1, .2]),
            "augmented_lagrangian": first.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dual_runtime.pt"
            torch.save(payload, path)
            restored_payload = torch.load(path, weights_only=True)
        second = AugmentedLagrangianController("rp_test", constraints())
        second.load_state_dict(restored_payload["augmented_lagrangian"])
        event_a = first.update({"geometry": .3, "strain": .2}, 2)
        event_b = second.update({"geometry": .3, "strain": .2}, 2)
        self.assertEqual(event_a, event_b)
        self.assertEqual(first.state_dict(), second.state_dict())
        torch.testing.assert_close(restored_payload["path_used"], payload["path_used"], atol=0, rtol=0)

    def test_changed_definition_or_program_is_rejected(self):
        controller = AugmentedLagrangianController("rp_test", constraints())
        state = controller.state_dict()
        with self.assertRaisesRegex(ValueError, "program lineage"):
            AugmentedLagrangianController("rp_other", constraints()).load_state_dict(state)
        changed = constraints()
        changed[0]["penalty"] = 3.0
        with self.assertRaisesRegex(ValueError, "definition mismatch"):
            AugmentedLagrangianController("rp_test", changed).load_state_dict(state)

    def test_missing_native_scalar_frame_is_explicitly_interpolated(self):
        reward = AugmentedLagrangianReward.__new__(AugmentedLagrangianReward)
        reward.local_frames = [
            {"time": .50, "status": "ok", "local_geometry_loss": .2,
             "local_geometry_max_residual": .3, "local_strain_kcal_mol": 10.},
            {"time": .51, "status": "unavailable", "reason": "invalid graph"},
            {"time": .52, "status": "ok", "local_geometry_loss": .1,
             "local_geometry_max_residual": .2, "local_strain_kcal_mol": 6.},
        ]
        reward.local_times = [row["time"] for row in reward.local_frames]
        reward.time = .51
        reward.al_cfg = {"native_time_tolerance": .002, "native_interpolation_max_gap": .03}
        row = reward.native_local()
        self.assertEqual(row["reference_mode"], "linear_scalar_interpolation")
        self.assertEqual(row["reference_bracket"], [.5, .52])
        self.assertAlmostEqual(row["local_strain_kcal_mol"], 8.)


if __name__ == "__main__":
    unittest.main()
