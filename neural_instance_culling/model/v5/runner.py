"""Formal, resumable GCOF-PVS V5 training runner.

The runner is deliberately separate from :mod:`train`: ``train_step`` owns
one-scene differentiable computation, while this module owns experiment
permissions, the fixed scene schedule, optimizer state, and recovery.  It
never opens calibration, validation, or test pose rows during training.

Examples (Linux/conda):

    python -m neural_instance_culling.model.v5.runner scan \
        --phase pilot --protocol shared --output-dir <scan-dir>

    python -m neural_instance_culling.model.v5.runner train \
        --protocol shared --variant FULL --seed 0 \
        --real-updates-per-scene 36000 --output-dir <run-dir> --device cuda
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from neural_instance_culling.benchmark.v5.contracts import (
    ALL_VARIANTS as BENCHMARK_ALL_VARIANTS,
    LOSO_VARIANTS as BENCHMARK_LOSO_VARIANTS,
)
from neural_instance_culling.config.pvs_v5_scene_registry import (
    DEFAULT_REGISTRY,
    REPO_ROOT,
    load_registry,
)
from neural_instance_culling.dataset.v5.permissions import (
    FoldAccessPolicy,
    make_loso_training_policy,
    make_universal_training_policy,
)
from neural_instance_culling.dataset.v5.synthetic_scene_manifest import (
    generate_synthetic_scene_manifest,
)
from neural_instance_culling.model.v5.core import GCOFPVSV5
from neural_instance_culling.model.v5.losses import SceneDualState
from neural_instance_culling.model.v5.schedule import (
    balanced_step_schedule,
    parameter_scan_matrix,
)
from neural_instance_culling.model.v5.train import (
    CHECKPOINT_SCHEMA,
    checkpoint_payload,
    train_step,
)
from neural_instance_culling.model.v5.training_data import V5SceneTrainingData


RUN_SCHEMA = "gcof-pvs-v5-formal-training-run-v1"
SCAN_SCHEMA = "gcof-pvs-v5-parameter-scan-matrix-v1"
RNG_SCHEMA = "gcof-pvs-v5-rng-state-v1"
DEFAULT_SYNTHETIC_BASE_SEED = 20260918
DEFAULT_SYNTHETIC_ROOT = (
    REPO_ROOT / "neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/synthetic"
)
FORMAL_REAL_UPDATES_PER_SCENE = 36_000
PILOT_UPDATES = 12_000
CONFIRMATION_UPDATES = 36_000
ALL_RUN_VARIANTS = tuple(BENCHMARK_ALL_VARIANTS)
LOSO_RUN_VARIANTS = tuple(BENCHMARK_LOSO_VARIANTS)


@dataclass(frozen=True)
class TrainingSceneSpec:
    """The geometry and source-train paths needed by one training scene."""

    scene_id: str
    source_kind: str
    pose_dataset: Path
    runtime_meta: Path
    compiled_dir: Path
    viewcell_shape: str
    viewcell_half_extent_m: tuple[float, float, float]
    camera_clip_m: tuple[float, float]
    region_manifest: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "sourceKind": self.source_kind,
            "poseDataset": str(self.pose_dataset),
            "runtimeMeta": str(self.runtime_meta),
            "compiledDir": str(self.compiled_dir),
            "viewcellShape": self.viewcell_shape,
            "viewcellHalfExtentM": list(self.viewcell_half_extent_m),
            "cameraClipM": list(self.camera_clip_m),
            "regionManifest": None if self.region_manifest is None else str(self.region_manifest),
        }


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _real_scene_specs(
    registry: Mapping[str, Any],
    *,
    source_scene_ids: Sequence[str],
    require_files: bool,
) -> list[TrainingSceneSpec]:
    entries = {str(entry["id"]): entry for entry in registry["scenes"]}
    compiled_root = REPO_ROOT / str(registry["compiledOutputRoot"])
    result: list[TrainingSceneSpec] = []
    for scene_id in source_scene_ids:
        if scene_id not in entries:
            raise ValueError(f"unknown registered V5 scene: {scene_id}")
        entry = entries[scene_id]
        spec = TrainingSceneSpec(
            scene_id=scene_id,
            source_kind="real",
            pose_dataset=REPO_ROOT / str(entry["poseDataset"]),
            runtime_meta=REPO_ROOT / str(entry["runtimeMeta"]),
            compiled_dir=compiled_root / scene_id,
            viewcell_shape=str(entry["viewCell"]["shape"]),
            viewcell_half_extent_m=tuple(float(x) for x in entry["viewCell"]["halfExtentM"]),
            camera_clip_m=tuple(float(x) for x in entry["cameraClipM"]),
        )
        if require_files:
            _require_scene_files(spec, require_probe=True)
        result.append(spec)
    return result


def _synthetic_scene_specs(
    root: Path,
    *,
    base_seed: int,
    require_files: bool,
) -> list[TrainingSceneSpec]:
    """Build the 96 seed-level synthetic source scenes without reading labels."""

    catalog = generate_synthetic_scene_manifest(int(base_seed))
    result: list[TrainingSceneSpec] = []
    for entry in sorted(catalog["scenes"], key=lambda item: str(item["sceneId"])):
        if str(entry["split"]) != "train":
            continue
        scene_id = str(entry["sceneId"])
        scene_root = root / scene_id
        spec = TrainingSceneSpec(
            scene_id=scene_id,
            source_kind="synthetic",
            pose_dataset=scene_root / "pose_csr",
            runtime_meta=scene_root / "runtimeVisibilityMeta.json",
            compiled_dir=scene_root / "compiled",
            # The synthetic manifest carries per-pose disk/box support points.
            # These values are only the fixed constructor fallback.
            viewcell_shape="horizontal_disk",
            viewcell_half_extent_m=(0.75, 0.75, 0.0),
            camera_clip_m=(0.05, 1_000_000.0),
            region_manifest=scene_root / "scene_manifest.json",
        )
        if require_files:
            _require_scene_files(spec, require_probe=True)
        result.append(spec)
    if len(result) != 96:
        raise ValueError(f"synthetic source catalog must contain 96 train scenes, found {len(result)}")
    return result


def _require_scene_files(spec: TrainingSceneSpec, *, require_probe: bool) -> None:
    required = (
        spec.pose_dataset / "dataset_meta.json",
        spec.runtime_meta,
        spec.compiled_dir / "surface/surface_manifest.json",
        spec.compiled_dir / "relation/relation_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if spec.region_manifest is not None and not spec.region_manifest.is_file():
        missing.append(str(spec.region_manifest))
    if require_probe and not (spec.compiled_dir / "probes/external_hit_probe_manifest.json").is_file():
        missing.append(str(spec.compiled_dir / "probes/external_hit_probe_manifest.json"))
    if missing:
        raise FileNotFoundError(f"V5 source scene {spec.scene_id} is incomplete: {', '.join(missing)}")


def build_training_context(
    *,
    protocol: str,
    held_out_scene: str | None,
    registry_path: Path = DEFAULT_REGISTRY,
    synthetic_root: Path = DEFAULT_SYNTHETIC_ROOT,
    synthetic_base_seed: int = DEFAULT_SYNTHETIC_BASE_SEED,
    require_files: bool,
) -> tuple[tuple[TrainingSceneSpec, ...], FoldAccessPolicy]:
    """Resolve only source scenes for a shared or LOSO training run."""

    protocol = str(protocol).lower()
    if protocol not in {"shared", "loso"}:
        raise ValueError("protocol must be shared or loso")
    registry = load_registry(Path(registry_path), validate_files=require_files)
    real_ids = [str(scene["id"]) for scene in registry["scenes"]]
    if protocol == "loso":
        if not held_out_scene or held_out_scene not in real_ids:
            raise ValueError("LOSO requires one registered --held-out-scene")
        source_real_ids = [scene_id for scene_id in real_ids if scene_id != held_out_scene]
    else:
        if held_out_scene is not None:
            raise ValueError("shared training cannot declare a held-out scene")
        source_real_ids = real_ids
    real_specs = _real_scene_specs(
        registry,
        source_scene_ids=source_real_ids,
        require_files=require_files,
    )
    synthetic_specs = _synthetic_scene_specs(
        Path(synthetic_root),
        base_seed=int(synthetic_base_seed),
        require_files=require_files,
    )
    source_ids = [spec.scene_id for spec in (*real_specs, *synthetic_specs)]
    if protocol == "loso":
        policy = make_loso_training_policy(source_ids, str(held_out_scene))
    else:
        policy = make_universal_training_policy(source_ids)
    return tuple((*real_specs, *synthetic_specs)), policy


def expected_total_updates(
    real_scene_count: int,
    *,
    real_updates_per_scene: int | None = None,
    total_updates: int | None = None,
) -> int:
    if real_scene_count <= 0:
        raise ValueError("real_scene_count must be positive")
    if (real_updates_per_scene is None) == (total_updates is None):
        raise ValueError("choose exactly one update budget")
    if real_updates_per_scene is not None:
        if real_updates_per_scene <= 0:
            raise ValueError("real_updates_per_scene must be positive")
        real_steps = int(real_scene_count) * int(real_updates_per_scene)
        return real_steps + real_steps // 2
    if total_updates is None or total_updates <= 0:
        raise ValueError("total_updates must be positive")
    return int(total_updates)


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_states(data_rng: np.random.Generator) -> dict[str, Any]:
    return {
        "schema": RNG_SCHEMA,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "data": data_rng.bit_generator.state,
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: Mapping[str, Any], data_rng: np.random.Generator) -> None:
    if states.get("schema") != RNG_SCHEMA:
        raise ValueError("unsupported V5 RNG checkpoint schema")
    random.setstate(tuple(states["python"]))
    numpy_state = states["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), numpy_state[2], numpy_state[3], numpy_state[4]))
    data_rng.bit_generator.state = states["data"]
    torch.random.set_rng_state(torch.as_tensor(states["torch"], dtype=torch.uint8, device="cpu"))
    cuda_states = states.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(value, dtype=torch.uint8, device="cpu") for value in cuda_states])


class _SceneDataStore:
    def __init__(
        self,
        specs: Sequence[TrainingSceneSpec],
        policy: FoldAccessPolicy,
        *,
        require_probe: bool,
    ) -> None:
        self.specs = {spec.scene_id: spec for spec in specs}
        self.policy = policy
        self.require_probe = bool(require_probe)
        self.loaded: dict[str, V5SceneTrainingData] = {}

    def get(self, scene_id: str) -> V5SceneTrainingData:
        if scene_id not in self.specs:
            raise KeyError(f"unknown source training scene: {scene_id}")
        if scene_id not in self.loaded:
            spec = self.specs[scene_id]
            data = V5SceneTrainingData(
                scene_id=spec.scene_id,
                pose_dataset=spec.pose_dataset,
                runtime_meta=spec.runtime_meta,
                compiled_dir=spec.compiled_dir,
                viewcell_shape=spec.viewcell_shape,
                viewcell_half_extent_m=spec.viewcell_half_extent_m,
                camera_clip_m=spec.camera_clip_m,
                probe_policy=self.policy if self.require_probe else None,
                region_manifest=spec.region_manifest,
            )
            if self.require_probe and data.probe_table is None:
                raise FileNotFoundError(f"source scene {scene_id} has no authorized train probe table")
            self.loaded[scene_id] = data
        return self.loaded[scene_id]


def _human_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"gcof_pvs_v5.{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _prepare_metrics_log(path: Path, *, resume_step: int | None) -> None:
    """Create a fresh metric stream or rewind it to the resumed checkpoint."""

    if resume_step is None:
        if path.exists():
            raise FileExistsError(
                f"refusing to append a new V5 run to an existing metric log: {path}"
            )
        return
    if not path.exists():
        if resume_step != 0:
            raise FileNotFoundError(
                f"resume checkpoint step {resume_step} has no metric log: {path}"
            )
        return
    retained: list[str] = []
    previous = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.get("globalStep", -1))
        if step <= previous:
            raise ValueError(
                f"V5 metric log steps are not strictly increasing at line {line_number}"
            )
        previous = step
        if step <= resume_step:
            retained.append(json.dumps(row, allow_nan=False))
    if resume_step > 0 and not retained:
        raise ValueError(
            f"metric log does not contain resumed checkpoint step {resume_step}"
        )
    if retained and int(json.loads(retained[-1])["globalStep"]) != resume_step:
        raise ValueError(
            f"metric log does not contain resumed checkpoint step {resume_step}"
        )
    path.write_text(("\n".join(retained) + "\n") if retained else "", encoding="utf-8")


def _resume_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol", "variant", "objective", "seed", "heldOutScene",
        "modelLearningRate", "dualLearningRate", "weightDecay",
        "totalUpdates", "realUpdatesPerScene", "scheduleSeed",
        "poseCount", "probeCount", "geometryChunkSize", "sourceSceneIds",
    )
    return {key: config.get(key) for key in keys}


def _validate_checkpoint(payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    required = {
        "schema", "modelConfig", "model", "optimizer", "scheduler", "dualState",
        "globalStep", "sceneUpdates", "rngStates", "config", "testRead",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"V5 checkpoint is missing fields: {', '.join(missing)}")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a current V5 training checkpoint")
    if payload.get("testRead") is not False:
        raise ValueError("V5 training checkpoints must declare testRead=false")
    stored_config = payload["config"]
    if not isinstance(stored_config, Mapping):
        raise ValueError("V5 checkpoint config is invalid")
    if stored_config.get("schema") != RUN_SCHEMA:
        raise ValueError("checkpoint config is not a current V5 run contract")
    if _resume_contract(stored_config) != _resume_contract(config):
        raise ValueError("checkpoint training contract disagrees with the requested run")
    if int(payload["globalStep"]) < 0 or int(payload["globalStep"]) > int(config["totalUpdates"]):
        raise ValueError("checkpoint global step is outside the requested schedule")
    if not isinstance(payload["sceneUpdates"], Mapping):
        raise ValueError("checkpoint scene update counts are invalid")
    if payload["rngStates"].get("schema") != RNG_SCHEMA:
        raise ValueError("checkpoint RNG state is invalid")


def _model_variant_and_objective(variant: str) -> tuple[str, str]:
    normalized = str(variant).upper()
    if normalized not in ALL_RUN_VARIANTS:
        raise ValueError(f"variant must be one of {ALL_RUN_VARIANTS}")
    if normalized == "PBCE_OBJECTIVE":
        return "FULL", "PBCE_OBJECTIVE"
    return normalized, "FULL"


def train_run(
    *,
    protocol: str,
    variant: str,
    seed: int,
    output_dir: Path,
    registry_path: Path = DEFAULT_REGISTRY,
    synthetic_root: Path = DEFAULT_SYNTHETIC_ROOT,
    synthetic_base_seed: int = DEFAULT_SYNTHETIC_BASE_SEED,
    held_out_scene: str | None = None,
    model_learning_rate: float = 2e-4,
    dual_learning_rate: float = 3e-3,
    weight_decay: float = 1e-5,
    total_updates: int | None = None,
    real_updates_per_scene: int | None = FORMAL_REAL_UPDATES_PER_SCENE,
    device: str = "cuda",
    resume: Path | None = None,
    checkpoint_every: int = 500,
    log_every: int = 100,
    pose_count: int = 4,
    probe_count: int = 8192,
    geometry_chunk_size: int = 512,
) -> dict[str, Any]:
    """Train one registered V5 member and return its non-test summary."""

    model_variant, objective = _model_variant_and_objective(variant)
    if protocol == "loso" and variant == "PBCE_OBJECTIVE":
        raise ValueError("PBCE_OBJECTIVE is a shared ablation only")
    if checkpoint_every <= 0 or log_every <= 0 or pose_count <= 0 or probe_count <= 0:
        raise ValueError("checkpoint/log/pose/probe counts must be positive")
    if model_learning_rate <= 0.0 or dual_learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("optimizer parameters are invalid")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    specs, policy = build_training_context(
        protocol=protocol,
        held_out_scene=held_out_scene,
        registry_path=Path(registry_path),
        synthetic_root=Path(synthetic_root),
        synthetic_base_seed=synthetic_base_seed,
        require_files=True,
    )
    real_ids = tuple(spec.scene_id for spec in specs if spec.source_kind == "real")
    synthetic_ids = tuple(spec.scene_id for spec in specs if spec.source_kind == "synthetic")
    total = expected_total_updates(
        len(real_ids),
        real_updates_per_scene=real_updates_per_scene,
        total_updates=total_updates,
    )
    if real_updates_per_scene is not None and total_updates is not None:
        raise ValueError("train_run received both total_updates and real_updates_per_scene")
    schedule_seed = int(seed) + 0x5EED
    scene_ids = tuple(spec.scene_id for spec in specs)
    config: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "protocol": str(protocol),
        "variant": str(variant).upper(),
        "objective": objective,
        "modelVariant": model_variant,
        "seed": int(seed),
        "heldOutScene": held_out_scene,
        "modelLearningRate": float(model_learning_rate),
        "dualLearningRate": float(dual_learning_rate),
        "weightDecay": float(weight_decay),
        "totalUpdates": int(total),
        "realUpdatesPerScene": None if real_updates_per_scene is None else int(real_updates_per_scene),
        "scheduleSeed": schedule_seed,
        "poseCount": int(pose_count),
        "probeCount": int(probe_count),
        "geometryChunkSize": int(geometry_chunk_size),
        "sourceSceneIds": list(scene_ids),
        "realSceneIds": list(real_ids),
        "syntheticSceneIds": list(synthetic_ids),
        "labelSplitsRead": ["train"],
        "selectionSplitsRead": [],
        "testRead": False,
        "trainingPermissions": policy.to_manifest(),
    }
    config["resumeContract"] = _resume_contract(config)
    run_config_path = output_dir / "run_config.json"
    metrics_path = output_dir / "train_metrics.jsonl"
    if resume is None and any(
        path.exists()
        for path in (
            run_config_path,
            metrics_path,
            output_dir / "checkpoint_last.pt",
            output_dir / "training_summary.json",
        )
    ):
        raise FileExistsError(
            f"refusing to overwrite an existing V5 run without --resume: {output_dir}"
        )
    resume_payload: Mapping[str, Any] | None = None
    if resume is not None:
        resume_payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
        _validate_checkpoint(resume_payload, config)
    _json_dump(run_config_path, config)
    logger = _human_logger(output_dir)
    logger.info(
        "start V5 run protocol=%s variant=%s seed=%s total_updates=%s device=%s testRead=false",
        protocol, variant, seed, total, device_obj,
    )
    logger.info("source scenes real=%d synthetic=%d; calibration/validation/test are not read by trainer", len(real_ids), len(synthetic_ids))

    seed_everything(int(seed))
    data_rng = np.random.default_rng(int(seed) + 0xDADA)
    model = GCOFPVSV5(model_variant).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_learning_rate), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(total), eta_min=0.0
    )
    dual_state = SceneDualState(scene_ids, dual_learning_rate, device="cpu")
    scene_updates = {scene_id: 0 for scene_id in scene_ids}
    global_step = 0
    if resume_payload is not None:
        payload = resume_payload
        if payload["modelConfig"] != model.config:
            raise ValueError("checkpoint model configuration disagrees with the requested variant")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        dual_state.load_state_dict(payload["dualState"], strict=True)
        global_step = int(payload["globalStep"])
        saved_updates = {str(key): int(value) for key, value in payload["sceneUpdates"].items()}
        if set(saved_updates) != set(scene_updates) or any(
            value < 0 or value > total for value in saved_updates.values()
        ):
            raise ValueError("checkpoint scene IDs disagree with current source schedule")
        scene_updates = saved_updates
        restore_rng_states(payload["rngStates"], data_rng)
        logger.info("resumed checkpoint=%s at global_step=%d", Path(resume), global_step)

    require_probe = variant != "GENERIC_RELATION_28"
    store = _SceneDataStore(specs, policy, require_probe=require_probe)
    assignments = balanced_step_schedule(
        real_ids,
        synthetic_ids,
        **(
            {"real_updates_per_scene": int(real_updates_per_scene)}
            if real_updates_per_scene is not None
            else {"total_updates": int(total_updates)}
        ),
        seed=schedule_seed,
    )
    _prepare_metrics_log(
        metrics_path,
        resume_step=(global_step if resume is not None else None),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with metrics_path.open("a", encoding="utf-8") as metrics_stream:
        for assignment in itertools.islice(assignments, global_step, None):
            current_count = int(scene_updates[assignment.scene_id])
            if current_count != int(assignment.scene_update):
                raise ValueError(
                    f"schedule/checkpoint mismatch for {assignment.scene_id}: "
                    f"expected {assignment.scene_update}, restored {current_count}"
                )
            scene = store.get(assignment.scene_id)
            pose_batch = scene.sample_pose_batch(data_rng, pose_count=pose_count)
            probe_batch = (
                scene.sample_probe_batch(data_rng, observation_count=probe_count)
                if require_probe
                else None
            )
            result = train_step(
                model=model,
                scene=scene,
                pose_batch=pose_batch,
                probe_batch=probe_batch,
                optimizer=optimizer,
                dual_state=dual_state,
                device=device_obj,
                quarter_turns=assignment.yaw_quarter_turns,
                geometry_chunk_size=geometry_chunk_size,
                objective=objective,
                scheduler=scheduler,
            )
            scene_updates[assignment.scene_id] += 1
            global_step += 1
            row = {
                "globalStep": global_step,
                "sceneId": result.scene_id,
                "sourceKind": assignment.source_kind,
                "sceneUpdate": scene_updates[assignment.scene_id],
                "yawQuarterTurns": assignment.yaw_quarter_turns,
                "loss": result.loss,
                "fieldLoss": result.loss_field,
                "riskExtra": result.risk_extra,
                "riskCount": result.risk_count,
                "riskVisual": result.risk_visual,
                "lambdaCount": result.lambda_count,
                "lambdaVisual": result.lambda_visual,
                "candidateCount": result.candidate_count,
                "targetUnitCount": result.target_unit_count,
                "geometryUnitCount": result.geometry_unit_count,
                "learningRate": float(optimizer.param_groups[0]["lr"]),
            }
            metrics_stream.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_stream.flush()
            if global_step % log_every == 0 or global_step == 1:
                elapsed = max(time.monotonic() - started, 1e-6)
                logger.info(
                    "step=%d/%d scene=%s loss=%.6g field=%.6g risk_count=%.6g risk_visual=%.6g lr=%.4g %.2f step/s",
                    global_step, total, result.scene_id, result.loss, result.loss_field,
                    result.risk_count, result.risk_visual, optimizer.param_groups[0]["lr"],
                    global_step / elapsed,
                )
            if global_step % checkpoint_every == 0:
                payload = checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    dual_state=dual_state,
                    global_step=global_step,
                    scene_updates=scene_updates,
                    rng_states=capture_rng_states(data_rng),
                    config=config,
                )
                _atomic_torch_save(payload, output_dir / "checkpoint_last.pt")
                _atomic_torch_save(payload, output_dir / f"checkpoint_step_{global_step:07d}.pt")

    if global_step != total:
        raise RuntimeError(f"V5 schedule ended at {global_step} updates, expected {total}")
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dual_state=dual_state,
        global_step=global_step,
        scene_updates=scene_updates,
        rng_states=capture_rng_states(data_rng),
        config=config,
    )
    _atomic_torch_save(payload, output_dir / "checkpoint_last.pt")
    _atomic_torch_save(payload, output_dir / f"checkpoint_step_{global_step:07d}.pt")
    summary = {
        "schema": RUN_SCHEMA,
        "run": config,
        "completed": True,
        "globalStep": global_step,
        "sceneUpdates": scene_updates,
        "testRead": False,
        "labelSplitsRead": ["train"],
        "checkpoint": str((output_dir / "checkpoint_last.pt").resolve()),
    }
    _json_dump(output_dir / "training_summary.json", summary)
    logger.info("completed V5 run at %d updates", global_step)
    return summary


def _scan_run_name(phase: str, config_name: str, protocol: str, seed: int, held_out: str | None) -> str:
    suffix = f"{protocol}_seed{int(seed)}"
    if held_out:
        suffix += f"_holdout_{held_out}"
    return f"pvs_v5_scan_{phase}_{config_name}_{suffix}"


def build_scan_matrix(
    *,
    phase: str,
    protocol: str,
    seed: int,
    held_out_scene: str | None,
    source_scene_ids: Sequence[str],
    output_dir: Path,
    selected_configurations: Sequence[str] = (),
) -> dict[str, Any]:
    phase = str(phase).lower()
    if phase not in {"pilot", "confirmation"}:
        raise ValueError("scan phase must be pilot or confirmation")
    configurations = {config.name: config for config in parameter_scan_matrix()}
    if phase == "pilot":
        selected = list(configurations.values())
        updates = PILOT_UPDATES
    else:
        names = list(selected_configurations)
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError("confirmation requires exactly two distinct pilot configuration names")
        unknown = sorted(set(names) - set(configurations))
        if unknown:
            raise ValueError(f"unknown confirmation configuration(s): {', '.join(unknown)}")
        selected = [configurations[name] for name in names]
        updates = CONFIRMATION_UPDATES
    rows = []
    for config in selected:
        row = {
            "name": config.name,
            "runName": _scan_run_name(phase, config.name, protocol, seed, held_out_scene),
            "modelLearningRate": config.model_learning_rate,
            "dualLearningRate": config.dual_learning_rate,
            "weightDecay": 1e-5,
            "fieldCoefficient": 0.25,
            "updates": updates,
            "selectionSplit": "validation",
            "thresholdSplit": "calibration",
            "testRead": False,
            "status": "planned",
        }
        rows.append(row)
    payload = {
        "schema": SCAN_SCHEMA,
        "protocol": str(protocol),
        "phase": phase,
        "seed": int(seed),
        "heldOutScene": held_out_scene,
        "sourceSceneIds": list(source_scene_ids),
        "pilotUpdates": PILOT_UPDATES,
        "confirmationUpdates": CONFIRMATION_UPDATES,
        "selectionRule": [
            "strict_lcb_scene_count",
            "mean_target_scene_count",
            "scene_equal_weighted_recall_lcb",
            "scene_equal_cnor_useful_cull",
            "lower_predicted_over_gt",
        ],
        "selectionSplit": "validation",
        "thresholdSplit": "calibration",
        "testRead": False,
        "runs": rows,
    }
    _json_dump(output_dir / "scan_matrix.json", payload)
    return payload


def _execute_scan(
    matrix: Mapping[str, Any],
    *,
    output_dir: Path,
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in matrix["runs"]:
        run_dir = output_dir / str(row["runName"])
        summary = train_run(
            protocol=str(matrix["protocol"]),
            variant="FULL",
            seed=int(matrix["seed"]),
            output_dir=run_dir,
            held_out_scene=matrix.get("heldOutScene"),
            model_learning_rate=float(row["modelLearningRate"]),
            dual_learning_rate=float(row["dualLearningRate"]),
            weight_decay=float(row["weightDecay"]),
            total_updates=int(row["updates"]),
            real_updates_per_scene=None,
            **dict(common),
        )
        results.append({"name": row["name"], "status": "completed", "summary": summary})
    return results


def _parse_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--synthetic-base-seed", type=int, default=DEFAULT_SYNTHETIC_BASE_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--pose-count", type=int, default=4)
    parser.add_argument("--probe-count", type=int, default=8192)
    parser.add_argument("--geometry-chunk-size", type=int, default=512)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train one shared or LOSO member")
    train_parser.add_argument("--protocol", choices=("shared", "loso"), required=True)
    train_parser.add_argument("--variant", choices=ALL_RUN_VARIANTS, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--held-out-scene")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--model-lr", type=float, default=2e-4)
    train_parser.add_argument("--dual-lr", type=float, default=3e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    budget = train_parser.add_mutually_exclusive_group()
    budget.add_argument("--real-updates-per-scene", type=int)
    budget.add_argument("--total-updates", type=int)
    train_parser.add_argument("--resume", type=Path)
    _parse_common(train_parser)

    scan_parser = subparsers.add_parser("scan", help="write or execute the fixed pilot/confirmation matrix")
    scan_parser.add_argument("--phase", choices=("pilot", "confirmation"), required=True)
    scan_parser.add_argument("--protocol", choices=("shared", "loso"), required=True)
    scan_parser.add_argument("--seed", type=int, default=0)
    scan_parser.add_argument("--held-out-scene")
    scan_parser.add_argument("--output-dir", type=Path, required=True)
    scan_parser.add_argument("--selected-config", action="append", default=[])
    scan_parser.add_argument("--execute", action="store_true", help="actually train planned members")
    _parse_common(scan_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "train":
        train_run(
            protocol=args.protocol,
            variant=args.variant,
            seed=args.seed,
            output_dir=args.output_dir,
            registry_path=args.registry,
            synthetic_root=args.synthetic_root,
            synthetic_base_seed=args.synthetic_base_seed,
            held_out_scene=args.held_out_scene,
            model_learning_rate=args.model_lr,
            dual_learning_rate=args.dual_lr,
            weight_decay=args.weight_decay,
            total_updates=args.total_updates,
            real_updates_per_scene=(
                FORMAL_REAL_UPDATES_PER_SCENE
                if args.total_updates is None and args.real_updates_per_scene is None
                else args.real_updates_per_scene
            ),
            device=args.device,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            pose_count=args.pose_count,
            probe_count=args.probe_count,
            geometry_chunk_size=args.geometry_chunk_size,
        )
        return

    specs, _policy = build_training_context(
        protocol=args.protocol,
        held_out_scene=args.held_out_scene,
        registry_path=args.registry,
        synthetic_root=args.synthetic_root,
        synthetic_base_seed=args.synthetic_base_seed,
        require_files=False,
    )
    matrix = build_scan_matrix(
        phase=args.phase,
        protocol=args.protocol,
        seed=args.seed,
        held_out_scene=args.held_out_scene,
        source_scene_ids=[spec.scene_id for spec in specs],
        output_dir=args.output_dir,
        selected_configurations=args.selected_config,
    )
    if args.execute:
        results = _execute_scan(
            matrix,
            output_dir=args.output_dir,
            common={
                "registry_path": args.registry,
                "synthetic_root": args.synthetic_root,
                "synthetic_base_seed": args.synthetic_base_seed,
                "device": args.device,
                "checkpoint_every": args.checkpoint_every,
                "log_every": args.log_every,
                "pose_count": args.pose_count,
                "probe_count": args.probe_count,
                "geometry_chunk_size": args.geometry_chunk_size,
            },
        )
        _json_dump(args.output_dir / "scan_execution.json", {"schema": SCAN_SCHEMA, "testRead": False, "results": results})
    print(json.dumps({"output": str((args.output_dir / "scan_matrix.json").resolve()), "runs": len(matrix["runs"]), "testRead": False}))


if __name__ == "__main__":
    main()


__all__ = [
    "ALL_RUN_VARIANTS",
    "CONFIRMATION_UPDATES",
    "FORMAL_REAL_UPDATES_PER_SCENE",
    "PILOT_UPDATES",
    "SCAN_SCHEMA",
    "TrainingSceneSpec",
    "build_scan_matrix",
    "build_training_context",
    "capture_rng_states",
    "expected_total_updates",
    "restore_rng_states",
    "train_run",
]
