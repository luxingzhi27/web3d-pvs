from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from neural_instance_culling.benchmark.v5.contracts import (
    ALL_VARIANTS,
    LOSO_VARIANTS,
    V5Run,
    validate_matrix,
)
from neural_instance_culling.benchmark.v5.evaluation import (
    evaluate_loso_matrix,
    evaluate_shared_matrix,
)
from neural_instance_culling.benchmark.v5.score_bundle import PoseScores
from neural_instance_culling.benchmark.v5.summary import summarize_loso, summarize_shared


SCENES = ("hkust", "ifcbench", "sponza", "viking", "big_city")


@dataclass
class _FixtureSceneScores:
    scene: str

    @property
    def sidecars(self) -> dict[str, bool]:
        return {"calibration": True, "validation": True}

    @property
    def splits(self) -> tuple[str, str]:
        return ("calibration", "validation")

    def records(self, split: str, *, allow_test: bool = False):
        del allow_test
        return [PoseScores(
            scene=self.scene,
            pose_index=0,
            split=split,
            candidate_ids=np.asarray([0, 1], dtype=np.uint32),
            scores=np.asarray([0.9, 0.1], dtype=np.float32),
            targets=np.asarray([1, 0], dtype=np.uint8),
            visible_weights=np.asarray([1.0, 0.0], dtype=np.float32),
        )]


def _shared_runs() -> list[V5Run]:
    return [
        V5Run(
            protocol="shared",
            variant=variant,
            seed=seed,
            scenes={scene: _FixtureSceneScores(scene) for scene in SCENES},
        )
        for variant in ALL_VARIANTS
        for seed in (1, 2, 3)
    ]


def _loso_runs() -> list[V5Run]:
    return [
        V5Run(
            protocol="loso",
            variant=variant,
            seed=seed,
            scenes={scene: _FixtureSceneScores(scene) for scene in SCENES},
            held_out_scene=held_out,
            source_scenes=tuple(scene for scene in SCENES if scene != held_out),
        )
        for variant in LOSO_VARIANTS
        for seed in (1, 2, 3)
        for held_out in SCENES
    ]


class V5MatrixTests(unittest.TestCase):
    def test_shared_matrix_requires_all_four_variants_and_three_seeds(self) -> None:
        runs = _shared_runs()
        self.assertEqual(len(validate_matrix(runs, protocol="shared", expected_scenes=SCENES)), 12)
        with self.assertRaisesRegex(ValueError, "variant"):
            validate_matrix(runs[:-3], protocol="shared", expected_scenes=SCENES)

    def test_loso_matrix_requires_three_variants_and_all_five_folds(self) -> None:
        runs = _loso_runs()
        self.assertEqual(len(validate_matrix(runs, protocol="loso", expected_scenes=SCENES)), 45)
        with self.assertRaisesRegex(ValueError, "fold coverage"):
            validate_matrix(runs[:-1], protocol="loso", expected_scenes=SCENES)

    def test_shared_evaluation_and_summary_keep_scene_equal_weight(self) -> None:
        rows = evaluate_shared_matrix(
            _shared_runs(),
            calibration_bootstrap_replicates=0,
            evaluation_bootstrap_replicates=0,
            expected_scenes=SCENES,
        )
        payload = summarize_shared(rows, expected_scenes=SCENES)
        self.assertEqual(payload["scene_count"], 5)
        self.assertEqual(payload["seed_count"], 3)
        self.assertEqual(payload["summary"]["FULL"]["scene_equal"]["weighted_recall"]["mean"], 1.0)
        self.assertFalse(payload["test_read"])

    def test_loso_outputs_both_threshold_modes_without_test_selection(self) -> None:
        rows = evaluate_loso_matrix(
            _loso_runs(),
            calibration_bootstrap_replicates=0,
            evaluation_bootstrap_replicates=0,
            expected_scenes=SCENES,
        )
        self.assertEqual(len(rows), 90)
        self.assertEqual({row["threshold_mode"] for row in rows}, {"source_global", "target_calibrated"})
        self.assertTrue(all(row["selection_split"] == "calibration" for row in rows))
        payload = summarize_loso(rows, expected_scenes=SCENES)
        self.assertEqual(payload["summary"]["FULL"]["source_global"]["scene_equal"]["weighted_recall"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
