from __future__ import annotations

from collections import Counter
import unittest

from neural_instance_culling.model.v5.schedule import (
    balanced_step_schedule,
    parameter_scan_matrix,
)


class V5ScheduleTest(unittest.TestCase):
    def test_formal_schedule_balances_real_scenes_and_half_synthetic_updates(self) -> None:
        rows = list(
            balanced_step_schedule(
                ["a", "b", "c", "d", "e"],
                [f"s{i}" for i in range(96)],
                real_updates_per_scene=8,
                seed=17,
            )
        )
        real = Counter(row.scene_id for row in rows if row.source_kind == "real")
        self.assertEqual(real, Counter({key: 8 for key in "abcde"}))
        self.assertEqual(sum(row.source_kind == "synthetic" for row in rows), 20)
        self.assertTrue(all(0 <= row.yaw_quarter_turns < 4 for row in rows))

    def test_scan_matrix_contains_only_two_registered_axes(self) -> None:
        matrix = parameter_scan_matrix()
        self.assertEqual(len(matrix), 6)
        self.assertEqual({row.model_learning_rate for row in matrix}, {1e-4, 2e-4, 4e-4})
        self.assertEqual({row.dual_learning_rate for row in matrix}, {1e-3, 3e-3})
        self.assertTrue(all(row.updates == 12_000 for row in matrix))


if __name__ == "__main__":
    unittest.main()
