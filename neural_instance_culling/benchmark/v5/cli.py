"""V5 columnar score export and streaming evaluation command line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import V5Run
from .evaluation import evaluate_loso_matrix, evaluate_shared_matrix, evaluate_shared_run
from .inference import export_score_bundle
from .scan_selection import read_json, select_scan_candidates, summarize_single_shared_run
from .summary import summarize_loso, summarize_shared, write_summary
from .streaming_export import export_streaming_score_sidecar


def _add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="calibrate and evaluate columnar score manifests")
    parser.add_argument("--protocol", choices=("shared", "loso"), required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True, help="V5 manifest.json; repeat for matrix members")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--final-test", action="store_true", help="permit a final frozen test replay")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--evaluation-bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)


def _add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export", help="compile checkpoint geometry and export columnar scores")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--protocol", choices=("shared", "loso"))
    parser.add_argument("--held-out-scene")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--geometry-chunk-size", type=int, default=512)
    parser.add_argument("--pose-chunk-size", type=int, default=4096)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def _add_scan_parsers(subparsers: argparse._SubParsersAction) -> None:
    evaluate = subparsers.add_parser("evaluate-run", help="evaluate one five-scene scan member")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--calibration-bootstrap-replicates", type=int, default=10_000)
    evaluate.add_argument("--evaluation-bootstrap-replicates", type=int, default=10_000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=0)
    select = subparsers.add_parser("select-scan", help="rank a complete pilot or confirmation scan")
    select.add_argument("--scan-matrix", type=Path, required=True)
    select.add_argument("--result", action="append", required=True, help="CONFIG=single_run_validation.json")
    select.add_argument("--top-k", type=int, default=2)
    select.add_argument("--output", type=Path, required=True)


def _add_streaming_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "export-streaming", help="convert one V5 scene to the cold-cache streaming score contract"
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_evaluate_parser(subparsers)
    _add_export_parser(subparsers)
    _add_scan_parsers(subparsers)
    _add_streaming_parser(subparsers)
    return parser.parse_args(argv)


def _evaluate(args: argparse.Namespace) -> None:
    if args.split == "test" and not args.final_test:
        raise PermissionError("--split test requires --final-test")
    runs = [V5Run.from_manifest(path, allow_test_read=bool(args.final_test)) for path in args.bundle]
    common = {
        "evaluation_split": args.split,
        "calibration_bootstrap_replicates": args.calibration_bootstrap_replicates,
        "evaluation_bootstrap_replicates": args.evaluation_bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "final_test": bool(args.final_test),
    }
    if args.protocol == "shared":
        rows = evaluate_shared_matrix(runs, **common)
        payload = summarize_shared(rows)
    else:
        rows = evaluate_loso_matrix(runs, **common)
        payload = summarize_loso(rows)
    write_summary(args.output, payload)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "protocol": args.protocol,
        "rows": len(rows),
        "split": args.split,
        "selectionSplit": "calibration",
        "testRead": bool(args.final_test),
    }, ensure_ascii=False))


def _export(args: argparse.Namespace) -> None:
    from neural_instance_culling.config.pvs_v5_scene_registry import DEFAULT_REGISTRY

    manifest = export_score_bundle(
        args.checkpoint,
        args.output_dir,
        registry_path=args.registry or DEFAULT_REGISTRY,
        scene_ids=args.scene,
        protocol=args.protocol,
        held_out_scene=args.held_out_scene,
        device=args.device,
        geometry_chunk_size=args.geometry_chunk_size,
        pose_chunk_size=args.pose_chunk_size,
        final_test=bool(args.final_test),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({"manifest": str(manifest), "testRead": bool(args.final_test)}, ensure_ascii=False))


def _evaluate_run(args: argparse.Namespace) -> None:
    run = V5Run.from_manifest(args.bundle)
    rows = evaluate_shared_run(
        run,
        calibration_bootstrap_replicates=args.calibration_bootstrap_replicates,
        evaluation_bootstrap_replicates=args.evaluation_bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload = summarize_single_shared_run(rows)
    write_summary(args.output, payload)
    print(json.dumps({"output": str(Path(args.output).resolve()), "rows": len(rows), "testRead": False}))


def _select_scan(args: argparse.Namespace) -> None:
    summaries: dict[str, dict] = {}
    for value in args.result:
        if "=" not in value:
            raise ValueError("--result must use CONFIG=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in summaries:
            raise ValueError("scan result names must be non-empty and unique")
        summaries[name] = dict(read_json(raw_path))
    payload = select_scan_candidates(
        read_json(args.scan_matrix),
        summaries,
        top_k=args.top_k,
    )
    write_summary(args.output, payload)
    print(json.dumps({"output": str(Path(args.output).resolve()), "selected": payload["selectedConfigurations"], "testRead": False}))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "evaluate":
        _evaluate(args)
    elif args.command == "export":
        _export(args)
    elif args.command == "evaluate-run":
        _evaluate_run(args)
    elif args.command == "export-streaming":
        manifest = export_streaming_score_sidecar(
            bundle_manifest=args.bundle,
            evaluation_summary=args.evaluation_summary,
            scene=args.scene,
            split=args.split,
            output_dir=args.output_dir,
            final_test=bool(args.final_test),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps({
            "manifest": str(manifest),
            "split": args.split,
            "testRead": args.split == "test",
        }, ensure_ascii=False))
    else:
        _select_scan(args)


if __name__ == "__main__":
    main()
