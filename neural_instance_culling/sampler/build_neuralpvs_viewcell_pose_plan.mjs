#!/usr/bin/env node
/**
 * Build NeuralPVS-style viewcell subpose plans from an existing pose plan.
 *
 * Input rows are representative model-query cameras.  For each representative
 * camera this script emits K subposes with the same view direction and FOV, but
 * with camera positions sampled uniformly inside that view cell.  The sampler
 * later renders every subpose; dataset construction unions visible ids per
 * viewcell, matching the from-region PVS supervision used by NeuralPVS.
 */
import fs from 'node:fs';
import path from 'node:path';
import {
  backOffsetForRadius,
  horizontalFovFromVertical,
  NEURALPVS_FOV_PROTOCOL,
  MODEL_FOV_Y_DEG,
  RENDER_FOV_Y_DEG,
} from './neuralpvs_fov_protocol.mjs';

const CATEGORY_IDS = {
  street_gap: 0,
  near_building: 1,
  plaza: 2,
  perimeter: 3,
  sky: 4,
  far: 5,
  unknown: 255,
};

function parseArgs(argv) {
  const args = {
    input: '',
    output: '',
    summary: '',
    scene: '',
    subposesPerViewcell: 16,
    radius: null,
    halfRight: null,
    halfForward: null,
    halfUp: null,
    seed: 20260623,
    renderFovY: MODEL_FOV_Y_DEG,
    modelFovY: MODEL_FOV_Y_DEG,
    baseFovY: RENDER_FOV_Y_DEG,
    keepAspect: true,
    width: 512,
    height: 288,
    maxViewcells: 0,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    const value = argv[i + 1];
    i += 1;
    const name = key.slice(2);
    if (name === 'input') args.input = path.resolve(value);
    else if (name === 'output') args.output = path.resolve(value);
    else if (name === 'summary') args.summary = path.resolve(value);
    else if (name === 'scene') args.scene = String(value);
    else if (name === 'subposes-per-viewcell') args.subposesPerViewcell = Number(value);
    else if (name === 'radius') args.radius = Number(value);
    else if (name === 'viewcell-half-right') args.halfRight = Number(value);
    else if (name === 'viewcell-half-forward') args.halfForward = Number(value);
    else if (name === 'viewcell-half-up') args.halfUp = Number(value);
    else if (name === 'seed') args.seed = Number(value);
    else if (name === 'fov-y') {
      args.renderFovY = Number(value);
      args.modelFovY = Number(value);
    }
    else if (name === 'render-fov-y' || name === 'sampling-fov-y') args.renderFovY = Number(value);
    else if (name === 'model-fov-y') args.modelFovY = Number(value);
    else if (name === 'base-fov-y') args.baseFovY = Number(value);
    else if (name === 'width') args.width = Number(value);
    else if (name === 'height') args.height = Number(value);
    else if (name === 'max-viewcells') args.maxViewcells = Number(value);
    else if (name === 'keep-aspect') args.keepAspect = value !== 'false' && value !== '0';
  }
  if (!args.input || !args.output) {
    throw new Error('Usage: build_neuralpvs_viewcell_pose_plan.mjs --input pose_plan.jsonl --output viewcell_pose_plan.jsonl');
  }
  if (!Number.isFinite(args.subposesPerViewcell) || args.subposesPerViewcell <= 0) {
    throw new Error('--subposes-per-viewcell must be positive');
  }
  if (!Number.isFinite(args.renderFovY) || args.renderFovY <= 0 || args.renderFovY >= 180) {
    throw new Error('--render-fov-y must be between 0 and 180 degrees');
  }
  if (!Number.isFinite(args.baseFovY) || args.baseFovY <= 0 || args.baseFovY >= 180) {
    throw new Error('--base-fov-y must be between 0 and 180 degrees');
  }
  if (!Number.isFinite(args.modelFovY) || args.modelFovY <= 0 || args.modelFovY >= 180) {
    throw new Error('--model-fov-y must be between 0 and 180 degrees');
  }
  if (Math.abs(args.renderFovY - MODEL_FOV_Y_DEG) > 1e-6 || Math.abs(args.modelFovY - MODEL_FOV_Y_DEG) > 1e-6) {
    throw new Error(`The current view-cell sampler requires a 66 degree sampling/model FOV.`);
  }
  if (Math.abs(args.baseFovY - RENDER_FOV_Y_DEG) > 1e-6) {
    throw new Error(`The current view-cell sampler requires a 60 degree frontend render FOV.`);
  }
  applySceneDefaults(args);
  if (!Number.isFinite(args.halfRight) || args.halfRight < 0) throw new Error('--viewcell-half-right must be non-negative');
  if (!Number.isFinite(args.halfForward) || args.halfForward < 0) throw new Error('--viewcell-half-forward must be non-negative');
  if (!Number.isFinite(args.halfUp) || args.halfUp < 0) throw new Error('--viewcell-half-up must be non-negative');
  return args;
}

function applySceneDefaults(args) {
  if (Number.isFinite(args.radius)) {
    args.halfRight ??= args.radius;
    args.halfForward ??= args.radius;
    args.halfUp ??= args.radius * 0.35;
    return;
  }
  const key = String(args.scene || '').toLowerCase().replace(/\\/g, '/');
  if (key.includes('hkust')) {
    args.halfRight ??= 4.0;
    args.halfForward ??= 4.0;
    args.halfUp ??= 1.5;
  } else if (key.includes('metropolis') || key.includes('ifcbench')) {
    args.halfRight ??= 2.5;
    args.halfForward ??= 2.5;
    args.halfUp ??= 1.0;
  } else {
    args.halfRight ??= 1.0;
    args.halfForward ??= 1.0;
    args.halfUp ??= 0.35;
  }
}

function readJsonl(file) {
  return fs.readFileSync(file, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`Failed to parse ${file}:${index + 1}: ${error.message}`);
      }
    });
}

function hash32(value) {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

function random01(seed, salt) {
  return hash32((seed ^ Math.imul(salt >>> 0, 0x9e3779b1)) >>> 0) / 0x100000000;
}

function normalize(v, fallback = [0, 0, -1]) {
  const x = Number(v?.[0] ?? fallback[0]);
  const y = Number(v?.[1] ?? fallback[1]);
  const z = Number(v?.[2] ?? fallback[2]);
  const len = Math.hypot(x, y, z);
  if (!(len > 1e-8)) return fallback.slice();
  return [x / len, y / len, z / len];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function cameraBasis(forward) {
  const f = normalize(forward);
  const upSeed = Math.abs(f[1]) > 0.98 ? [0, 0, 1] : [0, 1, 0];
  const right = normalize(cross(f, upSeed), [1, 0, 0]);
  const up = normalize(cross(right, f), [0, 1, 0]);
  return { forward: f, right, up };
}

function yawPitchFromForward(forward) {
  const f = normalize(forward);
  return {
    yawDeg: Math.atan2(-f[0], -f[2]) * 180 / Math.PI,
    pitchDeg: Math.asin(Math.max(-1, Math.min(1, f[1]))) * 180 / Math.PI,
  };
}

function stableSplit(viewcellId, seed) {
  const r = random01(seed, viewcellId + 101);
  if (r < 0.8) return 'train';
  if (r < 0.9) return 'val';
  return 'test';
}

function sampleViewcellOffset(viewcellId, subposeId, halfRight, halfForward, halfUp, basis, seed) {
  if (subposeId === 0) return [0, 0, 0];
  const localSeed = (seed + Math.imul(viewcellId + 1, 73856093) + Math.imul(subposeId + 1, 19349663)) >>> 0;
  const rightScale = (random01(localSeed, 1) * 2.0 - 1.0) * halfRight;
  const forwardScale = (random01(localSeed, 2) * 2.0 - 1.0) * halfForward;
  const vertical = (random01(localSeed, 3) * 2.0 - 1.0) * halfUp;
  return [
    basis.right[0] * rightScale + basis.forward[0] * forwardScale + basis.up[0] * vertical,
    basis.right[1] * rightScale + basis.forward[1] * forwardScale + basis.up[1] * vertical,
    basis.right[2] * rightScale + basis.forward[2] * forwardScale + basis.up[2] * vertical,
  ];
}

function main() {
  const args = parseArgs(process.argv);
  const reps = readJsonl(args.input);
  const selected = args.maxViewcells > 0 ? reps.slice(0, args.maxViewcells) : reps;
  fs.mkdirSync(path.dirname(args.output), { recursive: true });
  const out = fs.createWriteStream(args.output, { encoding: 'utf8' });
  let poseIndex = 0;
  const categoryCounts = {};
  for (let viewcellId = 0; viewcellId < selected.length; viewcellId += 1) {
    const rep = selected[viewcellId];
    const center = (rep.camera_pos || [0, 0, 0]).map(Number);
    const forward = normalize(rep.camera_forward || [0, 0, -1]);
    const basis = cameraBasis(forward);
    const { yawDeg, pitchDeg } = yawPitchFromForward(forward);
    const aspect = args.keepAspect && Number.isFinite(Number(rep.aspect))
      ? Number(rep.aspect)
      : args.width / Math.max(1, args.height);
    const height = Number.isFinite(Number(rep.height)) ? Number(rep.height) : args.height;
    const width = Number.isFinite(Number(rep.width)) ? Number(rep.width) : Math.round(height * aspect);
    const category = rep.sample_category || rep.viewcell_category || 'unknown';
    const categoryId = Number.isFinite(Number(rep.sample_category_id))
      ? Number(rep.sample_category_id)
      : (CATEGORY_IDS[category] ?? CATEGORY_IDS.unknown);
    categoryCounts[category] = (categoryCounts[category] || 0) + 1;
    const split = rep.split || stableSplit(viewcellId, args.seed);
    const viewcellRadius = Math.max(args.halfRight, args.halfForward, args.halfUp);
    const explicitBackOffset = Number(rep.pvs_back_offset);
    const pvsBackOffset = Number.isFinite(explicitBackOffset) && explicitBackOffset > 0
      ? explicitBackOffset
      : backOffsetForRadius(viewcellRadius, args.baseFovY);
    const modelFovX = horizontalFovFromVertical(args.modelFovY, aspect);
    for (let subposeId = 0; subposeId < args.subposesPerViewcell; subposeId += 1) {
      const offset = sampleViewcellOffset(
        viewcellId,
        subposeId,
        args.halfRight,
        args.halfForward,
        args.halfUp,
        basis,
        args.seed,
      );
      const cameraPos = [
        center[0] + offset[0],
        center[1] + offset[1],
        center[2] + offset[2],
      ];
      const row = {
        pose_index: poseIndex,
        viewcell_id: viewcellId,
        subpose_id: subposeId,
        viewcell_category: category,
        viewcell_center: center,
        viewcell_shape: 'camera_aligned_box',
        viewcell_half_extent: [args.halfRight, args.halfForward, args.halfUp],
        viewcell_radius: viewcellRadius,
        viewcell_forward: forward,
        viewcell_yaw_deg: yawDeg,
        viewcell_pitch_deg: pitchDeg,
        pvs_fov_y: args.modelFovY,
        pvs_fov_x: modelFovX,
        pvs_back_offset: pvsBackOffset,
        split,
        sample_category: category,
        sample_category_id: categoryId,
        camera_pos: cameraPos,
        camera_forward: forward,
        yaw_deg: yawDeg,
        pitch_deg: pitchDeg,
        render_fov_y: args.renderFovY,
        fov_y: args.renderFovY,
        aspect,
        width,
        height,
      };
      out.write(`${JSON.stringify(row)}\n`);
      poseIndex += 1;
    }
  }
  out.end();
  const summary = {
    schema: 'neuralpvs-viewcell-subpose-plan-v2',
    input: args.input,
    output: args.output,
    scene: args.scene || null,
    viewcellCount: selected.length,
    subposesPerViewcell: args.subposesPerViewcell,
    poseCount: poseIndex,
    viewcellShape: 'camera_aligned_box',
    viewcellHalfExtent: [args.halfRight, args.halfForward, args.halfUp],
    protocol: NEURALPVS_FOV_PROTOCOL.schema,
    renderFovY: args.renderFovY,
    baseFovY: args.baseFovY,
    modelFovY: args.modelFovY,
    relatedWorkReferenceGroundTruthPositionsPerViewcell: Number(
      NEURALPVS_FOV_PROTOCOL.groundTruthPositionsPerViewcell,
    ),
    backOffsetFormula: NEURALPVS_FOV_PROTOCOL.backOffsetFormula,
    categoryCounts,
    semantics: 'Each viewcell emits K same-direction deterministic pseudo-random camera positions inside the viewcell; model sampling and candidate inference use 66 degrees, while the frontend display camera uses 60 degrees.',
  };
  const summaryPath = args.summary || args.output.replace(/\.jsonl$/i, '_summary.json');
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2), 'utf8');
  console.log(JSON.stringify(summary, null, 2));
}

main();
