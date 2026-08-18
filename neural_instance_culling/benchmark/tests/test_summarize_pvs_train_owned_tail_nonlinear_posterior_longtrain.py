from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from neural_instance_culling.benchmark.summarize_pvs_train_owned_tail_nonlinear_posterior_longtrain import (
    FORMAL_SEEDS,
    summarize,
)


def _pose(pose_index: int, *, tp: int, fp: int, fn: int, tn: int) -> dict:
    candidate = tp + fp + fn + tn
    gt = tp + fn
    pred = tp + fp
    precision = tp / pred
    recall = tp / gt
    specificity = tn / (tn + fp)
    return {
        "poseIndex": pose_index,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "weightedTp": float(tp),
        "weightedGt": float(gt),
        "precision": precision,
        "recall": recall,
        "weightedRecall": recall,
        "f1": 2.0 * precision * recall / (precision + recall),
        "jaccard": tp / (tp + fp + fn),
        "accuracy": (tp + tn) / candidate,
        "balancedAccuracy": 0.5 * (recall + specificity),
        "specificity": specificity,
        "usefulCull": tn / candidate,
        "badCull": fn / candidate,
        "predCount": pred,
        "candidateCount": candidate,
        "gtCount": gt,
        "predictedGlbCount": float(pred),
        "predictedGlbBytes": float(pred * 100),
        "glbCountReduction": 1.0 - pred / candidate,
        "glbByteReduction": 1.0 - pred / candidate,
        "downloadUtilityRecall": recall,
    }


def _member(alpha: float, fp: int) -> dict:
    rows = [
        _pose(3, tp=8, fp=fp, fn=2, tn=20 - fp),
        _pose(7, tp=9, fp=fp, fn=1, tn=20 - fp),
    ]
    return {
        "mode": "dual_rescue",
        "alpha": alpha,
        "temperature": 0.05,
        "coverageWeight": 1.0,
        "calibration": {
            "status": "safe",
            "selected": {
                "threshold": 0.1,
                "weightedRecall": 0.995,
                "weightedRecallLowerConfidenceBound": 0.991,
            },
        },
        "validation": {
            "threshold": 0.1,
            "weightedRecallLowerConfidenceBound": 0.991,
            "poseMacro": {},
            "aggregate": {},
            "perPose": rows,
        },
    }


class NonlinearTailPosteriorLongtrainSummaryTest(unittest.TestCase):
    def test_three_seed_paired_summary_prefers_fewer_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in FORMAL_SEEDS:
                seed_root = root / f"seed{seed}"
                seed_root.mkdir(parents=True)
                (seed_root / "linear_dual_probe_formal.json").write_text(
                    json.dumps({"testRead": False, "members": [_member(0.5, 3)]}),
                    encoding="utf-8",
                )
                (seed_root / "nonlinear_dual_probe_formal.json").write_text(
                    json.dumps(
                        {
                            "testRead": False,
                            "members": [_member(0.0, 5), _member(0.5, 1)],
                        }
                    ),
                    encoding="utf-8",
                )
            payload = summarize(root, replicates=100, seed=9)
        self.assertFalse(payload["testRead"])
        variants = payload["variants"]
        self.assertGreater(
            variants["hinge_mlp_dual_probe"]["pooledValidation"]["aggregatePrecision"],
            variants["linear_dual_probe"]["pooledValidation"]["aggregatePrecision"],
        )
        comparison = payload["pairedBootstrap"]["comparisons"]["nonlinear_minus_linear"]
        self.assertGreater(comparison["metrics"]["aggregatePrecision"]["difference"], 0.0)
        self.assertLess(comparison["metrics"]["avgPredCount"]["difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
