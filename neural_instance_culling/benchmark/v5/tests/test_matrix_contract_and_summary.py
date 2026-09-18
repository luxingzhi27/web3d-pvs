from __future__ import annotations

import copy
import unittest

from neural_instance_culling.benchmark.v5.contracts import (
    ALL_VARIANTS,
    LOSO_VARIANTS,
    PoseRecord,
    SceneScores,
    V5Run,
    validate_matrix,
)
from neural_instance_culling.benchmark.v5.evaluation import (
    evaluate_loso_matrix,
    evaluate_shared_matrix,
)
from neural_instance_culling.benchmark.v5.summary import summarize_loso, summarize_shared


SCENES = ("hkust", "ifcbench", "sponza", "viking", "big_city")


def _scene(scene: str, *, split_names: tuple[str, ...] = ("calibration", "validation")) -> SceneScores:
    splits: dict[str, tuple[PoseRecord, ...]] = {}
    for split in split_names:
        splits[split] = (
            PoseRecord(
                pose_id=f"{scene}-{split}",
                split=split,
                candidate_ids=[0, 1],
                scores=[0.9, 0.1],
                targets=[1, 0],
                visible_weights=[1.0, 0.0],
                scene=scene,
            ),
        )
    return SceneScores(scene=scene, splits=splits)


def _shared_runs() -> list[V5Run]:
    return [
        V5Run(
            protocol="shared",
            variant=variant,
            seed=seed,
            scenes={scene: _scene(scene) for scene in SCENES},
        )
        for variant in ALL_VARIANTS
        for seed in (1, 2, 3)
    ]


def _loso_runs() -> list[V5Run]:
    result: list[V5Run] = []
    for variant in LOSO_VARIANTS:
        for seed in (1, 2, 3):
            for held_out in SCENES:
                result.append(
                    V5Run(
                        protocol="loso",
                        variant=variant,
                        seed=seed,
                        scenes={scene: _scene(scene) for scene in SCENES},
                        held_out_scene=held_out,
                        source_scenes=tuple(scene for scene in SCENES if scene != held_out),
                    )
                )
    return result


class V5MatrixTests(unittest.TestCase):
    def test_shared_matrix_requires_all_four_variants_and_three_seeds(self) -> None:
        runs = _shared_runs()
        self.assertEqual(len(validate_matrix(runs, protocol="shared", expected_scenes=SCENES)), 12)
        with self.assertRaisesRegex(ValueError, "variants"):
            validate_matrix(runs[:-3], protocol="shared", expected_scenes=SCENES)

    def test_loso_matrix_requires_three_variants_and_all_five_folds(self) -> None:
        runs = _loso_runs()
        self.assertEqual(len(validate_matrix(runs, protocol="loso", expected_scenes=SCENES)), 45)
        with self.assertRaisesRegex(ValueError, "fold coverage"):
            validate_matrix(runs[:-1], protocol="loso", expected_scenes=SCENES)

    def test_shared_evaluation_and_summary_keep_scene_equal_weight(self) -> None:
        rows = evaluate_shared_matrix(
            _shared_runs(),
            evaluation_split="validation",
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
            evaluation_split="validation",
            calibration_bootstrap_replicates=0,
            evaluation_bootstrap_replicates=0,
            expected_scenes=SCENES,
        )
        self.assertEqual(len(rows), 90)
        self.assertEqual({row["threshold_mode"] for row in rows}, {"source_global", "target_calibrated"})
        self.assertTrue(all(row["selection_split"] == "calibration" for row in rows))
        payload = summarize_loso(rows, expected_scenes=SCENES)
        self.assertEqual(payload["summary"]["FULL"]["source_global"]["scene_equal"]["weighted_recall"]["mean"], 1.0)

    def test_result_bundle_marked_test_read_is_rejected(self) -> None:
        run = _shared_runs()[0]
        payload = run.to_mapping()
        payload["testRead"] = True
        with self.assertRaisesRegex(ValueError, "test-tainted"):
            V5Run.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
