from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from neural_instance_culling.benchmark.generate_asset_runtime_table import build_outputs


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _hzb_fixture(root: Path, scene: str, variant: str) -> None:
    stem = f"geometry_shell_hzb_{variant.replace('-', '_')}_{scene}"
    directory = root / stem
    directory.mkdir(parents=True)
    files = {
        "positions.meshopt.bin": b"123",
        "indices.meshopt.bin": b"1234",
        "transforms.meshopt.bin": b"12345",
        "instance_aabb_fp32.bin": b"12",
        "instance_to_glb_uint32.bin": b"123",
    }
    for name, data in files.items():
        (directory / name).write_bytes(data)
    shell = {
        "schema": "geometry-shell-hzb-v2", "variant": variant,
        "sceneName": scene, "instanceCount": 3, "globalGlbCount": 2,
        "prototypeCount": 2,
        "queryContract": {"candidateTest": "all-candidate-conservative-aabb-hzb"},
        "files": {
            "instanceAabbs": {"file": "instance_aabb_fp32.bin"},
            "instanceToGlb": {"file": "instance_to_glb_uint32.bin"},
        },
        "streams": {
            "positions": {"file": "positions.meshopt.bin", "segments": [{"byteLength": 3, "decodedByteLength": 12}]},
            "indices": {"file": "indices.meshopt.bin", "segments": [{"byteLength": 4, "decodedByteLength": 16}]},
            "transforms": {"file": "transforms.meshopt.bin", "segments": [{"byteLength": 5, "decodedByteLength": 20}]},
        },
        "stats": {
            "compressedGeometryBytes": 12, "binaryPayloadBytes": 17,
            "runtimePayloadBytes": 5, "prototypeTriangles": 4,
            "rasterizedTriangleInstanceCount": 6, "sourceGlbCount": 2,
            "opaquePrimitiveCount": 2,
        },
        "selection": None,
    }
    if variant == "equal-asset":
        shell["selection"] = {"totalAssetBytes": 0}
    offline = {
        "schema": "geometry-shell-hzb-offline-report-v1", "runtimeAsset": False,
        "sceneName": scene, "variant": variant,
        "aggregate": {"sourceGlbCount": 2, "opaquePrimitiveCount": 2},
    }
    for _ in range(5):
        _write_json(directory / "shell_meta.json", shell)
        if variant == "equal-asset":
            shell["selection"]["totalAssetBytes"] = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    _write_json(directory / "shell_meta.json", shell)
    _write_json(root / f"{stem}.offline.json", offline)


def _runtime_csv(path: Path) -> None:
    fields = [
        "status", "scene", "device", "backend", "timingSource", "formalSessions", "sampleCount",
        "candidateMean", "candidateP95", "runtimeAssetMiB", "latencyMeanMs", "latencyMeanCi95LowMs",
        "latencyMeanCi95HighMs", "latencyP50Ms", "latencyP50Ci95LowMs", "latencyP50Ci95HighMs",
        "latencyP95Ms", "latencyP95Ci95LowMs", "latencyP95Ci95HighMs", "reason",
    ]
    rows = [{field: "" for field in fields} for _ in range(3)]
    rows[0].update({
        "status": "formal", "scene": "hkust", "device": "M2", "backend": "webgpu-v4",
        "timingSource": "gpuKernelMs", "formalSessions": "5", "sampleCount": "10",
        "candidateMean": "100", "candidateP95": "200", "runtimeAssetMiB": "1",
        "latencyMeanMs": "4", "latencyMeanCi95LowMs": "3", "latencyMeanCi95HighMs": "5",
        "latencyP50Ms": "2", "latencyP50Ci95LowMs": "1", "latencyP50Ci95HighMs": "3",
        "latencyP95Ms": "8", "latencyP95Ci95LowMs": "7", "latencyP95Ci95HighMs": "9",
    })
    rows[1].update({
        "status": "unavailable", "scene": "hkust", "device": "A6000", "backend": "webgpu-v4",
        "reason": "smoke-excluded: concurrent smoke", "latencyMeanMs": "999", "latencyP50Ms": "999", "latencyP95Ms": "999",
    })
    rows[2].update({"status": "unavailable", "scene": "ifcbench", "device": "phone", "backend": "webgpu-v4", "reason": "unavailable: no formal upload"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class AssetRuntimeTableTests(unittest.TestCase):
    def test_assets_stats_pareto_and_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper = root / "paper_results"
            hzb = paper / "hzb"
            for scene in ("hkust", "ifcbench"):
                for variant in ("lossless", "equal-asset"):
                    _hzb_fixture(hzb, scene, variant)
            neural = {}
            for scene, size in (("hkust", 10), ("ifcbench", 20)):
                asset = root / f"neural-{scene}"
                asset.mkdir()
                (asset / "model_meta.json").write_text(json.dumps({"schema": "fixture", "numInstances": 3, "numGlbs": 2}), encoding="utf-8")
                (asset / "weights.bin").write_bytes(b"x" * size)
                neural[scene] = asset
            (paper / "rank_sweep").mkdir(parents=True)
            (paper / "rank_sweep/rank_capacity.csv").write_text(
                "rank,survivalDim,runtimeFeatureDim,runtimeFeatureMiB,aggregateUsefulCull,aggregateBalancedAccuracy,aggregateWeightedRecall,aggregatePrecision,aggregateBadCull,avgPredCount,seedCount,safeSeedCount\n"
                "2,14,110,1,0.80,0.90,0.998,0.20,0.001,10,3,3\n"
                "4,28,124,2,0.85,0.91,0.997,0.21,0.001,9,3,3\n"
                "8,56,152,3,0.82,0.92,0.996,0.19,0.001,11,3,3\n", encoding="utf-8")
            _runtime_csv(paper / "mobile_runtime/runtime_paper_summary.csv")
            output = root / "figures"
            summary = build_outputs(paper, output, neural, hzb)
            self.assertEqual(summary["formal_latency_rows"], 1)
            self.assertEqual(summary["smoke_excluded_rows"], 1)
            self.assertEqual(summary["unavailable_rows"], 1)
            self.assertEqual(summary["pareto_ranks"], [2, 4])
            with (output / "table4_asset_runtime.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            hzb_row = next(row for row in rows if row["scene"] == "hkust" and row["variant"] == "lossless")
            shell_bytes = (hzb / "geometry_shell_hzb_lossless_hkust/shell_meta.json").stat().st_size
            self.assertEqual(hzb_row["transfer_bytes"], str(17 + shell_bytes))
            self.assertEqual(hzb_row["expanded_geometry_bytes"], "48")
            self.assertEqual(hzb_row["expanded_memory_bytes"], "53")
            self.assertEqual(hzb_row["expanded_triangles"], "6")
            neural_row = next(row for row in rows if row["scene"] == "hkust" and row["method"] == "neural")
            self.assertEqual(neural_row["transfer_bytes"], str(sum(p.stat().st_size for p in neural["hkust"].iterdir())))
            self.assertAlmostEqual(float(neural_row["neural_to_lossless_ratio"]), float(neural_row["transfer_to_lossless_ratio"]))
            smoke = next(row for row in rows if row["status"] == "smoke-excluded")
            self.assertEqual(smoke["latency_mean_ms"], "")
            self.assertEqual(smoke["latency_p95_ms"], "")
            unavailable = next(row for row in rows if row["status"] == "unavailable" and row["row_type"] == "runtime")
            self.assertEqual(unavailable["latency_p50_ms"], "")
            self.assertIn("equal-asset", (output / "table4_asset_runtime.md").read_text(encoding="utf-8"))
            for suffix in ("csv", "pdf", "svg", "png"):
                self.assertTrue((output / f"rank_capacity_pareto.{suffix}").is_file())


if __name__ == "__main__":
    unittest.main()
