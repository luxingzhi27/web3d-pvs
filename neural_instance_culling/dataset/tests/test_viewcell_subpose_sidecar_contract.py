from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_viewcell_subpose_sidecar import build_sidecar  # noqa: E402


class ViewcellSubposeSidecarContractTest(unittest.TestCase):
    def _make_inputs(self, root: Path) -> tuple[Path, Path]:
        source = root / "source"
        csr = root / "csr"
        source.mkdir()
        csr.mkdir()
        source_meta = {
            "schema": "proxy-viewcell-pvs-dataset-v1",
            "viewcellCount": 2,
            "visibleCount": 3,
            "subposeCount": 3,
            "splitIds": {"train": 0, "val": 1},
        }
        csr_meta = {
            "schema": "viewcell-csr-color-id-fov66-v2",
            "poseCount": 2,
            "sourceViewcellCount": 2,
            "visibleCount": 3,
            "candidateCount": 3,
            "sourceSubposeCount": 3,
            "poseStrideBytes": 64,
            "splitIds": {"train": 0, "validation": 1},
        }
        (source / "dataset_meta.json").write_text(json.dumps(source_meta), encoding="utf-8")
        (csr / "dataset_meta.json").write_text(json.dumps(csr_meta), encoding="utf-8")
        np.arange(2, dtype="<u4").tofile(source / "viewcell_ids.bin")
        np.asarray([0, 1], dtype="u1").tofile(source / "viewcell_split_ids.bin")
        np.asarray([0, 2, 3], dtype="<u8").tofile(source / "visible_offsets.bin")
        np.asarray([1, 2, 3], dtype="<u4").tofile(source / "visible_ids.bin")
        np.asarray([1.0, 2.0, 3.0], dtype="<f4").tofile(source / "visible_weights.bin")
        np.asarray([2, 1, 0], dtype="<u2").tofile(source / "visible_hit_counts.bin")
        np.asarray([0, 2, 3], dtype="<u8").tofile(source / "subpose_offsets.bin")

        poses = np.zeros(2 * 64, dtype="u1")
        poses[44] = 0
        poses[64 + 44] = 1
        poses.tofile(csr / "poses.bin")
        np.asarray([0, 2, 3], dtype="<u8").tofile(csr / "visible_offsets.bin")
        np.asarray([1, 2, 3], dtype="<u4").tofile(csr / "visible_ids.bin")
        np.asarray([1.0, 2.0, 3.0], dtype="<f4").tofile(csr / "visible_weights.bin")
        np.asarray([0, 2, 3], dtype="<u8").tofile(csr / "candidate_offsets.bin")
        np.asarray([1, 2, 3], dtype="<u4").tofile(csr / "candidate_ids.bin")
        return source, csr

    @staticmethod
    def _tree_digest(root: Path) -> dict[str, str]:
        result = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def test_manifest_records_contract_and_repeat_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, csr = self._make_inputs(Path(temporary))
            output = Path(temporary) / "sidecar"
            first = build_sidecar(source, csr, output)
            before = self._tree_digest(output)
            second = build_sidecar(source, csr, output)
            self.assertEqual(first, second)
            self.assertEqual(before, self._tree_digest(output))
            self.assertEqual(first["version"], 2)
            self.assertEqual(first["representedSubposeCount"], 3)
            self.assertEqual(first["subposeDenominatorSemantics"].startswith("H is"), True)
            self.assertEqual(set(first["fileSha256"]), {
                "visible_hit_counts.bin",
                "subpose_offsets.bin",
                "pose_indices.bin",
                "source_viewcell_split_ids.bin",
                "csr_pose_split_ids.bin",
            })
            self.assertEqual(first["fileSha256"]["visible_hit_counts.bin"], hashlib.sha256(
                (output / "visible_hit_counts.bin").read_bytes()
            ).hexdigest())

    def test_rejects_hit_count_above_subpose_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, csr = self._make_inputs(Path(temporary))
            np.asarray([3, 1, 0], dtype="<u2").tofile(source / "visible_hit_counts.bin")
            with self.assertRaisesRegex(ValueError, "greater than"):
                build_sidecar(source, csr, Path(temporary) / "sidecar")

    def test_rejects_offset_terminal_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, csr = self._make_inputs(Path(temporary))
            np.asarray([0, 2, 4], dtype="<u8").tofile(source / "visible_offsets.bin")
            with self.assertRaisesRegex(ValueError, "terminal offset"):
                build_sidecar(source, csr, Path(temporary) / "sidecar")

    def test_rejects_nonfinite_visible_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, csr = self._make_inputs(Path(temporary))
            np.asarray([1.0, np.nan, 3.0], dtype="<f4").tofile(source / "visible_weights.bin")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                build_sidecar(source, csr, Path(temporary) / "sidecar")


if __name__ == "__main__":
    unittest.main()
