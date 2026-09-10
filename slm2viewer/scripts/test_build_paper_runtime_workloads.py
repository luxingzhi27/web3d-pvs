from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_paper_runtime_workloads as target


class PaperRuntimeWorkloadTest(unittest.TestCase):
    def _runtime(self, root: Path, scene: dict) -> Path:
        runtime = root / scene["source"]
        runtime.mkdir(parents=True)
        (runtime / "weights.bin").write_bytes(b"weights")
        (runtime / "model_meta.json").write_text(
            json.dumps(
                {
                    "schema": target.RUNTIME_SCHEMA,
                    "testRead": False,
                    "numInstances": scene["instances"],
                    "threshold": scene["threshold"],
                    "viewcell": {"radiusM": scene["radius"]},
                    "query": {"viewcellRadiusM": scene["radius"]},
                    "calibration": {"safe": True},
                    "files": {"weights": {"file": "weights.bin"}},
                }
            ),
            encoding="utf-8",
        )
        return runtime

    def test_build_uses_final_scene_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            viewer = root / "slm2viewer"
            (viewer / "scripts").mkdir(parents=True)
            (viewer / "scripts/build_pvs_runtime_workload.py").write_text("", encoding="utf-8")
            for scene in target.SCENES:
                (root / "neural_instance_culling/dataset/out" / scene["dataset"]).mkdir(parents=True)
                self._runtime(root, scene)
            with mock.patch.object(target.subprocess, "run") as run:
                manifest = target.build(root, viewer)
            self.assertEqual([row["id"] for row in manifest["scenes"]], ["hkust", "ifcbench"])
            self.assertEqual(run.call_count, 2)
            for scene in target.SCENES:
                copied = viewer / "public/assets/neural_instance_culling" / scene["asset"]
                target._validate_runtime(copied, scene)

    def test_rejects_wrong_final_viewcell_radius(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = copy.deepcopy(target.SCENES[1])
            runtime = self._runtime(root, scene)
            meta_path = runtime / "model_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["viewcell"]["radiusM"] = 2.0
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "view-cell radius"):
                target._validate_runtime(runtime, scene)


if __name__ == "__main__":
    unittest.main()
