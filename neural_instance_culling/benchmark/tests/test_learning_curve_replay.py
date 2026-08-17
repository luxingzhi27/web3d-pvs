from __future__ import annotations

import sys
import unittest
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from evaluate_hierarchical_relation_learning_curve import _epoch_manifest  # noqa: E402


class LearningCurveReplayTests(unittest.TestCase):
    def test_epoch_manifest_records_cli_diagnostic_permission(self) -> None:
        source_manifest = {
            "_path": str(Path(__file__).resolve()),
            "mode": "learning_curve32",
            "variants": {"R0_free_survival": {"relationSource": "free"}},
            "splitPoseCounts": {"validation": 213},
            "splitCandidateDigests": {"validation": "a" * 64},
            "stepsPerEpoch": 100,
        }
        manifest = _epoch_manifest(
            source_manifest,
            epoch=4,
            epoch_root=Path("/tmp/epoch-root"),
            variants=["R0_free_survival"],
            seeds=[20260801],
            allow_unsafe_diagnostic=True,
        )
        self.assertTrue(manifest["allowUnsafeDiagnostic"])
        self.assertFalse(manifest["testRead"])


if __name__ == "__main__":
    unittest.main()
