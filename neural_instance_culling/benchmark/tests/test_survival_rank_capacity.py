from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neural_instance_culling.benchmark.run_survival_rank_capacity import (
    CAPACITY_RANKS,
    EXPECTED_RUNTIME_DIMS,
    EXPERIMENT,
    SURVIVAL_PARAMETER_DIM,
    _argument,
    _member_name,
    build_capacity_train_command,
    preflight,
)
from neural_instance_culling.benchmark.run_pvs import (
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    FORMAL_STEPS_PER_EPOCH,
)


DATA_ROOT = Path("/mnt/sda/rhyang/slm")


class SurvivalRankCapacityRunnerTests(unittest.TestCase):
    def test_registered_matrix_changes_only_direction_rank(self) -> None:
        self.assertEqual(CAPACITY_RANKS, (2, 4, 8, 12))
        self.assertEqual(SURVIVAL_PARAMETER_DIM, 7)
        self.assertEqual(
            EXPECTED_RUNTIME_DIMS,
            {2: 110, 4: 124, 8: 152, 12: 180},
        )
        self.assertEqual(FORMAL_SEEDS, (20260801, 20260802, 20260803))
        self.assertEqual(FORMAL_EPOCHS, 40)
        self.assertEqual(FORMAL_STEPS_PER_EPOCH, 900)

    def test_commands_use_no_contrastive_full_from_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for rank in CAPACITY_RANKS:
                with self.subTest(rank=rank):
                    member = Path(directory) / _member_name(rank, FORMAL_SEEDS[0])
                    command = build_capacity_train_command(
                        DATA_ROOT,
                        member,
                        rank=rank,
                        seed=FORMAL_SEEDS[0],
                    )
                    self.assertNotIn("--initial-checkpoint", command)
                    self.assertEqual(_argument(command, "--survival-rank"), str(rank))
                    self.assertEqual(
                        _argument(command, "--integrated-contrastive-mix"), "0.0"
                    )
                    self.assertEqual(
                        _argument(command, "--relation-source"),
                        "bounded_hierarchical",
                    )
                    self.assertEqual(
                        _argument(command, "--spectral-mode"), "moment_envelope"
                    )
                    self.assertEqual(_argument(command, "--epochs"), "40")
                    self.assertEqual(_argument(command, "--steps-per-epoch"), "900")
                    self.assertEqual(
                        _argument(command, "--experiment-name"),
                        f"{EXPERIMENT}_{member.name}",
                    )

    def test_preflight_records_the_complete_capacity_contract(self) -> None:
        contract = preflight(DATA_ROOT)
        self.assertEqual(contract["ranks"], [2, 4, 8, 12])
        self.assertEqual(contract["survivalDims"], [14, 28, 56, 84])
        self.assertEqual(contract["runtimeFeatureDims"], [110, 124, 152, 180])
        self.assertEqual(contract["contrastiveMix"], 0.0)
        self.assertTrue(contract["fromScratch"])
        self.assertFalse(contract["testRead"])


if __name__ == "__main__":
    unittest.main()
