from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_viewcell_subpose_sidecar import build_sidecar  # noqa: E402


class ViewcellSubposeSidecarTest(unittest.TestCase):
    def _dataset(self, root: Path, bad_subset: bool = False) -> tuple[Path, Path]:
        source, csr = root / "source", root / "csr"
        source.mkdir()
        csr.mkdir()
        (source / "dataset_meta.json").write_text(json.dumps({"viewcellCount": 2}), encoding="utf-8")
        (csr / "dataset_meta.json").write_text(json.dumps({"poseCount": 2}), encoding="utf-8")
        np.arange(2, dtype="<u4").tofile(source / "viewcell_ids.bin")
        np.asarray([0, 1], dtype="u1").tofile(source / "viewcell_split_ids.bin")
        np.asarray([0, 2, 3], dtype="<u8").tofile(source / "visible_offsets.bin")
        np.asarray([1, 2, 3], dtype="<u4").tofile(source / "visible_ids.bin")
        np.asarray([1.0, 2.0, 3.0], dtype="<f4").tofile(source / "visible_weights.bin")
        poses = bytearray(64 * 2)
        poses[44] = 0
        poses[64 + 44] = 3
        (csr / "poses.bin").write_bytes(poses)
        np.asarray([0, 2, 3], dtype="<u8").tofile(csr / "visible_offsets.bin")
        np.asarray([1, 2, 3], dtype="<u4").tofile(csr / "visible_ids.bin")
        np.asarray([1.0, 2.0, 3.0], dtype="<f4").tofile(csr / "visible_weights.bin")
        np.asarray([0, 2, 3], dtype="<u8").tofile(csr / "candidate_offsets.bin")
        np.asarray([1, 2, 3], dtype="<u4").tofile(csr / "candidate_ids.bin")
        np.asarray([2, 1, 0], dtype="<u2").tofile(source / "visible_hit_counts.bin")
        np.asarray([0, 2, 3], dtype="<u8").tofile(source / "subpose_offsets.bin")
        if bad_subset:
            np.asarray([1, 4, 3], dtype="<u4").tofile(source / "visible_ids.bin")
        return source, csr

    def test_writes_independent_manifest_and_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, csr = self._dataset(Path(temporary))
            output = Path(temporary) / "sidecar"
            manifest = build_sidecar(source, csr, output)
            self.assertEqual(manifest["schema"], "pvs-viewcell-subpose-supervision-sidecar-v1")
            self.assertEqual(np.fromfile(output / "visible_hit_counts.bin", dtype="<u2").tolist(), [2, 1, 0])
            self.assertEqual(np.fromfile(output / "subpose_offsets.bin", dtype="<u8").tolist(), [0, 2, 3])

    def test_rejects_visible_id_outside_candidate_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, csr = self._dataset(Path(temporary), bad_subset=True)
            with self.assertRaises(ValueError):
                build_sidecar(source, csr, Path(temporary) / "sidecar")


if __name__ == "__main__":
    unittest.main()
