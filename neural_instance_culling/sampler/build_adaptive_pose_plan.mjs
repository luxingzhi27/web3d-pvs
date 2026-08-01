#!/usr/bin/env node
/* 生成 Instance-PVS v3 自适应 rvcServer 采样 pose plan。
 *
 * 目标：
 *   - 不再在整个 sceneBounds 中均匀撒点。
 *   - 先用清洗后的实例 AABB footprint 找到建筑密集区。
 *   - 主要在楼间街道、建筑边缘、入口附近、建筑群近外围采样。
 *   - 低高度点必须避开任何实例 AABB，避免把相机放进楼体或构件内部。
 *
 * 输出：
 *   pose_plan.jsonl：每行一个已经展开方向、pitch、aspect 的 pose。
 *   pose_plan_summary.json：采样统计。
 *   pose_plan_preview.svg：俯视检查图，快速确认点是否集中在楼间空隙。
 */

import fs from 'node:fs';
import path from 'node:path';

const DEFAULT_RUNTIME_META = 'hkust-v3/assets/runtimeVisibilityMeta.json';
const DEFAULT_OUTPUT = 'neural_instance_culling/sampler/out/adaptive_pose_plan_v3.jsonl';
const RAD = Math.PI / 180;
const CATEGORY_IDS = {
  street_gap: 0,
  near_building: 1,
  plaza: 2,
  perimeter: 3,
  sky: 4,
  far: 5,
};

function parseArgs(argv) {
  const args = {
    runtimeMeta: DEFAULT_RUNTIME_META,
    output: DEFAULT_OUTPUT,
    summary: null,
    preview: null,
    gridStep: 15,
    maxBasePositions: 0,
    seed: 20260501,
    yawBins: 24,
    pitches: [-20, 0, 20],
    lowHeights: [2, 5, 10, 15, 25],
    skyHeights: [40, 60, 90, 120, 150],
    farHeights: [30, 80, 130],
    fovY: 66,
    captureHeight: 900,
    viewportUpscale: 1.2,
    aspects: [0.5625, 0.75, 1.0, 1.3333333333, 1.6, 1.7777777778, 2.1666666667, 2.3333333333],
    aspectWeights: [0.08, 0.08, 0.08, 0.12, 0.12, 0.24, 0.16, 0.12],
    aspectMode: 'weighted',
    cleanMaxFootprintArea: 200000,
    cleanMaxFootprintSpan: 450,
    cleanMinVerticalSize: 1.5,
    thinLargeArea: 2000,
    thinMaxHeight: 0.35,
    buildingDilateCells: 1,
    collisionVerticalPadding: 0.35,
    borderRejectCells: 6,
    streetMinDistCells: 1,
    streetMaxDistCells: 8,
    nearMaxDistCells: 4,
    plazaMaxDistCells: 16,
    perimeterMaxDistCells: 10,
    streetGapSearchCells: 12,
    categoryWeights: {
      street_gap: 0.42,
      near_building: 0.28,
      plaza: 0.12,
      perimeter: 0.10,
      sky: 0.05,
      far: 0.03,
    },
    yawJitterDeg: 5,
    pitchJitterDeg: 3,
    positionJitterM: 3,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    const value = argv[i + 1];
    i += 1;
    const name = key.slice(2);
    if (name === 'runtime-meta') args.runtimeMeta = path.resolve(value);
    else if (name === 'output') args.output = path.resolve(value);
    else if (name === 'summary') args.summary = path.resolve(value);
    else if (name === 'preview') args.preview = path.resolve(value);
    else if (name === 'grid-step') args.gridStep = Number(value);
    else if (name === 'max-base-positions') args.maxBasePositions = Number(value);
    else if (name === 'seed') args.seed = Number(value);
    else if (name === 'yaw-bins') args.yawBins = Number(value);
    else if (name === 'pitches') args.pitches = parseCsvNumbers(value);
    else if (name === 'low-heights') args.lowHeights = parseCsvNumbers(value);
    else if (name === 'sky-heights') args.skyHeights = parseCsvNumbers(value);
    else if (name === 'far-heights') args.farHeights = parseCsvNumbers(value);
    else if (name === 'fov-y') args.fovY = Number(value);
    else if (name === 'capture-height') args.captureHeight = Number(value);
    else if (name === 'viewport-upscale') args.viewportUpscale = Number(value);
    else if (name === 'aspects') args.aspects = parseCsvNumbers(value);
    else if (name === 'aspect-weights') args.aspectWeights = parseCsvNumbers(value);
    else if (name === 'aspect-mode') args.aspectMode = value;
    else if (name === 'clean-max-footprint-area') args.cleanMaxFootprintArea = Number(value);
    else if (name === 'clean-max-footprint-span') args.cleanMaxFootprintSpan = Number(value);
    else if (name === 'clean-min-vertical-size') args.cleanMinVerticalSize = Number(value);
    else if (name === 'thin-large-area') args.thinLargeArea = Number(value);
    else if (name === 'thin-max-height') args.thinMaxHeight = Number(value);
    else if (name === 'building-dilate-cells') args.buildingDilateCells = Number(value);
    else if (name === 'collision-vertical-padding') args.collisionVerticalPadding = Number(value);
    else if (name === 'border-reject-cells') args.borderRejectCells = Number(value);
    else if (name === 'street-min-dist-cells') args.streetMinDistCells = Number(value);
    else if (name === 'street-max-dist-cells') args.streetMaxDistCells = Number(value);
    else if (name === 'near-max-dist-cells') args.nearMaxDistCells = Number(value);
    else if (name === 'plaza-max-dist-cells') args.plazaMaxDistCells = Number(value);
    else if (name === 'perimeter-max-dist-cells') args.perimeterMaxDistCells = Number(value);
    else if (name === 'street-gap-search-cells') args.streetGapSearchCells = Number(value);
    else if (name === 'position-jitter-m') args.positionJitterM = Number(value);
    else if (name === 'yaw-jitter-deg') args.yawJitterDeg = Number(value);
    else if (name === 'pitch-jitter-deg') args.pitchJitterDeg = Number(value);
  }
  if (!args.summary) args.summary = args.output.replace(/\.jsonl$/i, '_summary.json');
  if (!args.preview) args.preview = args.output.replace(/\.jsonl$/i, '_preview.svg');
  return args;
}

function parseCsvNumbers(text) {
  return String(text).split(',').map((v) => Number(v.trim())).filter((v) => Number.isFinite(v));
}

function sceneMinMax(sceneBounds) {
  if (sceneBounds.min && sceneBounds.max) {
    return { min: sceneBounds.min.map(Number), max: sceneBounds.max.map(Number), size: sceneBounds.size.map(Number) };
  }
  const center = sceneBounds.center.map(Number);
  const size = sceneBounds.size.map(Number);
  return {
    min: center.map((v, i) => v - size[i] * 0.5),
    max: center.map((v, i) => v + size[i] * 0.5),
    size,
  };
}

function boundsMinMax(bounds) {
  if (bounds.min && bounds.max) {
    const min = bounds.min.map(Number);
    const max = bounds.max.map(Number);
    return { min, max, center: min.map((v, i) => (v + max[i]) * 0.5), size: min.map((v, i) => max[i] - v) };
  }
  const center = bounds.center.map(Number);
  const size = bounds.size.map(Number);
  return {
    min: center.map((v, i) => v - size[i] * 0.5),
    max: center.map((v, i) => v + size[i] * 0.5),
    center,
    size,
  };
}

function createRng(seed) {
  let state = seed >>> 0;
  return function rng() {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function shuffleInPlace(items, rng) {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function weightedPick(items, rng) {
  const total = items.reduce((sum, item) => sum + Math.max(0, item.weight), 0);
  if (!(total > 0)) return items[Math.floor(rng() * items.length)];
  let target = rng() * total;
  for (const item of items) {
    target -= Math.max(0, item.weight);
    if (target <= 0) return item;
  }
  return items[items.length - 1];
}

function pickAspect(args, poseIndex, rng) {
  if (args.aspectMode === 'cartesian') return args.aspects[poseIndex % args.aspects.length];
  const total = args.aspectWeights.reduce((sum, v) => sum + v, 0);
  let target = rng() * Math.max(total, 1e-6);
  for (let i = 0; i < args.aspects.length; i += 1) {
    target -= args.aspectWeights[i] || 0;
    if (target <= 0) return args.aspects[i];
  }
  return args.aspects[args.aspects.length - 1];
}

function cleanBuildingComponents(records, args) {
  const clean = [];
  const rejected = {
    zero: 0,
    tooLargeArea: 0,
    tooLargeSpan: 0,
    flatCampusPlane: 0,
    lowVertical: 0,
  };
  const collision = [];
  for (const record of records) {
    if (!record.bounds) continue;
    const b = boundsMinMax(record.bounds);
    const sx = Math.max(0, b.size[0]);
    const sy = Math.max(0, b.size[1]);
    const sz = Math.max(0, b.size[2]);
    const area = sx * sz;
    const span = Math.max(sx, sz);
    if (!(area > 1e-6) || !(span > 1e-6)) {
      rejected.zero += 1;
      continue;
    }
    collision.push({ id: Number(record.componentGlobalId), min: b.min, max: b.max, center: b.center, size: b.size });
    if (area > args.cleanMaxFootprintArea) {
      rejected.tooLargeArea += 1;
      continue;
    }
    if (span > args.cleanMaxFootprintSpan) {
      rejected.tooLargeSpan += 1;
      continue;
    }
    if (area > args.thinLargeArea && sy < args.thinMaxHeight) {
      rejected.flatCampusPlane += 1;
      continue;
    }
    if (sy < args.cleanMinVerticalSize && area > 16) {
      rejected.lowVertical += 1;
      continue;
    }
    clean.push({ id: Number(record.componentGlobalId), min: b.min, max: b.max, center: b.center, size: b.size, area });
  }
  return { clean, collision, rejected };
}

function buildGrid(clean, scene, args) {
  // Compute raw min/max by reduce (clean can hold 275k+ records; Math.min(...spread) overflows the call stack).
  let rawMinX = Infinity, rawMinZ = Infinity, rawMaxX = -Infinity, rawMaxZ = -Infinity;
  for (const b of clean) {
    if (b.min[0] < rawMinX) rawMinX = b.min[0];
    if (b.min[2] < rawMinZ) rawMinZ = b.min[2];
    if (b.max[0] > rawMaxX) rawMaxX = b.max[0];
    if (b.max[2] > rawMaxZ) rawMaxZ = b.max[2];
  }
  const rawMin = [rawMinX, rawMinZ];
  const rawMax = [rawMaxX, rawMaxZ];
  const pad = Math.max(150, args.gridStep * 12);
  const minX = Math.max(scene.min[0], rawMin[0] - pad);
  const maxX = Math.min(scene.max[0], rawMax[0] + pad);
  const minZ = Math.max(scene.min[2], rawMin[1] - pad);
  const maxZ = Math.min(scene.max[2], rawMax[1] + pad);
  const nx = Math.max(1, Math.ceil((maxX - minX) / args.gridStep));
  const nz = Math.max(1, Math.ceil((maxZ - minZ) / args.gridStep));
  const occupied = new Uint8Array(nx * nz);
  const mark = (x, z) => {
    if (x >= 0 && x < nx && z >= 0 && z < nz) occupied[x * nz + z] = 1;
  };
  for (const b of clean) {
    const x0 = Math.max(0, Math.floor((b.min[0] - minX) / args.gridStep) - args.buildingDilateCells);
    const x1 = Math.min(nx - 1, Math.floor((b.max[0] - minX) / args.gridStep) + args.buildingDilateCells);
    const z0 = Math.max(0, Math.floor((b.min[2] - minZ) / args.gridStep) - args.buildingDilateCells);
    const z1 = Math.min(nz - 1, Math.floor((b.max[2] - minZ) / args.gridStep) + args.buildingDilateCells);
    for (let x = x0; x <= x1; x += 1) {
      for (let z = z0; z <= z1; z += 1) mark(x, z);
    }
  }
  return { minX, maxX, minZ, maxZ, nx, nz, occupied };
}

function distanceToOccupied(grid, maxCells) {
  const { nx, nz, occupied } = grid;
  const dist = new Int16Array(nx * nz);
  dist.fill(32767);
  const queue = [];
  for (let x = 0; x < nx; x += 1) {
    for (let z = 0; z < nz; z += 1) {
      const index = x * nz + z;
      if (occupied[index]) {
        dist[index] = 0;
        queue.push(index);
      }
    }
  }
  let head = 0;
  const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  while (head < queue.length) {
    const index = queue[head++];
    const x = Math.floor(index / nz);
    const z = index - x * nz;
    const nextDist = dist[index] + 1;
    if (nextDist > maxCells) continue;
    for (const [dx, dz] of dirs) {
      const xx = x + dx;
      const zz = z + dz;
      if (xx < 0 || xx >= nx || zz < 0 || zz >= nz) continue;
      const ni = xx * nz + zz;
      if (dist[ni] <= nextDist) continue;
      dist[ni] = nextDist;
      queue.push(ni);
    }
  }
  return dist;
}

function hasBuildingsBothSides(grid, x, z, radius) {
  const { nx, nz, occupied } = grid;
  const hitDir = [false, false, false, false];
  for (let d = 1; d <= radius; d += 1) {
    if (x - d >= 0 && occupied[(x - d) * nz + z]) hitDir[0] = true;
    if (x + d < nx && occupied[(x + d) * nz + z]) hitDir[1] = true;
    if (z - d >= 0 && occupied[x * nz + (z - d)]) hitDir[2] = true;
    if (z + d < nz && occupied[x * nz + (z + d)]) hitDir[3] = true;
  }
  return (hitDir[0] && hitDir[1]) || (hitDir[2] && hitDir[3]) || hitDir.filter(Boolean).length >= 2;
}

function buildCollisionIndex(collision, grid, args) {
  const cells = new Map();
  const key = (x, z) => `${x},${z}`;
  for (const b of collision) {
    const x0 = Math.max(0, Math.floor((b.min[0] - grid.minX) / args.gridStep));
    const x1 = Math.min(grid.nx - 1, Math.floor((b.max[0] - grid.minX) / args.gridStep));
    const z0 = Math.max(0, Math.floor((b.min[2] - grid.minZ) / args.gridStep));
    const z1 = Math.min(grid.nz - 1, Math.floor((b.max[2] - grid.minZ) / args.gridStep));
    if (x1 < 0 || z1 < 0 || x0 >= grid.nx || z0 >= grid.nz) continue;
    for (let x = x0; x <= x1; x += 1) {
      for (let z = z0; z <= z1; z += 1) {
        const k = key(x, z);
        let arr = cells.get(k);
        if (!arr) {
          arr = [];
          cells.set(k, arr);
        }
        arr.push(b);
      }
    }
  }
  return { cells, key };
}

function collidesCamera(point, collisionIndex, grid, args) {
  const x = Math.floor((point[0] - grid.minX) / args.gridStep);
  const z = Math.floor((point[2] - grid.minZ) / args.gridStep);
  const list = collisionIndex.cells.get(collisionIndex.key(x, z)) || [];
  for (const b of list) {
    if (point[0] < b.min[0] || point[0] > b.max[0] || point[2] < b.min[2] || point[2] > b.max[2]) continue;
    if (point[1] >= b.min[1] - args.collisionVerticalPadding && point[1] <= b.max[1] + args.collisionVerticalPadding) {
      return true;
    }
  }
  return false;
}

function classifyCells(grid, dist, args) {
  const candidates = {
    street_gap: [],
    near_building: [],
    plaza: [],
    perimeter: [],
  };
  for (let x = args.borderRejectCells; x < grid.nx - args.borderRejectCells; x += 1) {
    for (let z = args.borderRejectCells; z < grid.nz - args.borderRejectCells; z += 1) {
      const index = x * grid.nz + z;
      if (grid.occupied[index]) continue;
      const d = dist[index];
      if (d === 32767) continue;
      const wx = grid.minX + (x + 0.5) * args.gridStep;
      const wz = grid.minZ + (z + 0.5) * args.gridStep;
      const base = { x, z, wx, wz, dist: d };
      if (d >= args.streetMinDistCells && d <= args.streetMaxDistCells && hasBuildingsBothSides(grid, x, z, args.streetGapSearchCells)) {
        candidates.street_gap.push(base);
      } else if (d >= 1 && d <= args.nearMaxDistCells) {
        candidates.near_building.push(base);
      } else if (d > args.nearMaxDistCells && d <= args.plazaMaxDistCells) {
        candidates.plaza.push(base);
      } else if (d <= args.perimeterMaxDistCells) {
        candidates.perimeter.push(base);
      }
    }
  }
  return candidates;
}

function chooseBasePositions(candidates, args, rng) {
  const pools = [];
  for (const [category, list] of Object.entries(candidates)) {
    for (const item of list) {
      const distWeight = category === 'street_gap'
        ? 1.0 + Math.max(0, args.streetMaxDistCells - item.dist) * 0.08
        : 1.0;
      pools.push({ ...item, category, weight: (args.categoryWeights[category] || 0.1) * distWeight });
    }
  }
  if (pools.length === 0) return [];
  const target = args.maxBasePositions > 0 ? Math.min(args.maxBasePositions, pools.length) : pools.length;
  const selected = [];
  const used = new Set();
  let attempts = 0;
  while (selected.length < target && attempts < target * 50) {
    attempts += 1;
    const item = weightedPick(pools, rng);
    const k = `${item.x},${item.z}`;
    if (used.has(k)) continue;
    used.add(k);
    selected.push(item);
  }
  if (selected.length < target) {
    shuffleInPlace(pools, rng);
    for (const item of pools) {
      const k = `${item.x},${item.z}`;
      if (used.has(k)) continue;
      used.add(k);
      selected.push(item);
      if (selected.length >= target) break;
    }
  }
  return selected;
}

function yawPitchForward(yawDeg, pitchDeg) {
  const yaw = yawDeg * RAD;
  const pitch = pitchDeg * RAD;
  const x = -Math.sin(yaw) * Math.cos(pitch);
  const y = Math.sin(pitch);
  const z = -Math.cos(yaw) * Math.cos(pitch);
  const len = Math.hypot(x, y, z) || 1;
  return [x / len, y / len, z / len];
}

function addJitter(value, amount, rng) {
  return value + (rng() * 2 - 1) * amount;
}

function expandPoses(basePositions, scene, grid, collisionIndex, args, rng) {
  const records = [];
  const rejected = { collision: 0 };
  let poseIndex = 0;
  const heightsFor = (category) => (category === 'sky' ? args.skyHeights : category === 'far' ? args.farHeights : args.lowHeights);

  for (const base of basePositions) {
    for (const height of heightsFor(base.category)) {
      const y = scene.min[1] + height;
      const jx = addJitter(0, args.positionJitterM, rng);
      const jz = addJitter(0, args.positionJitterM, rng);
      const position = [base.wx + jx, y, base.wz + jz];
      if ((base.category !== 'sky' && base.category !== 'far') && collidesCamera(position, collisionIndex, grid, args)) {
        rejected.collision += 1;
        continue;
      }
      for (let yawIndex = 0; yawIndex < args.yawBins; yawIndex += 1) {
        const yawBase = yawIndex * 360 / args.yawBins;
        for (const pitchBase of args.pitches) {
          const yawDeg = addJitter(yawBase, args.yawJitterDeg, rng);
          const pitchDeg = addJitter(pitchBase, args.pitchJitterDeg, rng);
          const aspect = pickAspect(args, poseIndex, rng);
          const heightPx = Math.round(args.captureHeight * args.viewportUpscale);
          const widthPx = Math.round(args.captureHeight * aspect * args.viewportUpscale);
          records.push({
            pose_index: poseIndex,
            sample_category: base.category,
            sample_category_id: CATEGORY_IDS[base.category] ?? 255,
            camera_pos: position,
            camera_forward: yawPitchForward(yawDeg, pitchDeg),
            yaw_deg: yawDeg,
            pitch_deg: pitchDeg,
            fov_y: args.fovY,
            aspect,
            width: widthPx,
            height: heightPx,
          });
          poseIndex += 1;
        }
      }
    }
  }
  return { records, rejected };
}

function addSkyAndFarBases(basePositions, clean, scene, args, rng) {
  // Reduce-based min/max (clean can hold 275k+ records; spreading overflows the call stack).
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const b of clean) {
    if (b.min[0] < minX) minX = b.min[0];
    if (b.max[0] > maxX) maxX = b.max[0];
    if (b.min[2] < minZ) minZ = b.min[2];
    if (b.max[2] > maxZ) maxZ = b.max[2];
  }
  const centerX = (minX + maxX) * 0.5;
  const centerZ = (minZ + maxZ) * 0.5;
  const span = Math.max(maxX - minX, maxZ - minZ);
  const primaryCount = Math.max(1, basePositions.length);
  const skyCount = Math.max(4, Math.round(primaryCount * args.categoryWeights.sky));
  for (let i = 0; i < skyCount; i += 1) {
    const b = clean[Math.floor(rng() * clean.length)];
    basePositions.push({ wx: b.center[0], wz: b.center[2], x: -1, z: -1, dist: 0, category: 'sky' });
  }
  const farRadii = [Math.max(250, span * 0.45), Math.max(500, span * 0.75), Math.max(1000, span * 1.1)];
  const farCount = Math.max(4, Math.round(primaryCount * args.categoryWeights.far));
  for (let i = 0; i < farCount; i += 1) {
    const radius = farRadii[i % farRadii.length];
    const a = i * Math.PI * 2 / farCount;
    const wx = Math.max(scene.min[0], Math.min(scene.max[0], centerX + Math.cos(a) * radius));
    const wz = Math.max(scene.min[2], Math.min(scene.max[2], centerZ + Math.sin(a) * radius));
    basePositions.push({ wx, wz, x: -1, z: -1, dist: 0, category: 'far' });
  }
}

function writePreview(filePath, grid, candidates, selected) {
  const w = 1200;
  const h = Math.max(300, Math.round(w * (grid.maxZ - grid.minZ) / Math.max(1, grid.maxX - grid.minX)));
  const sx = (x) => ((x - grid.minX) / Math.max(1, grid.maxX - grid.minX)) * w;
  const sz = (z) => ((z - grid.minZ) / Math.max(1, grid.maxZ - grid.minZ)) * h;
  const colors = {
    street_gap: '#e53935',
    near_building: '#fb8c00',
    plaza: '#43a047',
    perimeter: '#1e88e5',
    sky: '#8e24aa',
    far: '#546e7a',
  };
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`,
    '<rect width="100%" height="100%" fill="#111"/>'];
  const cellW = Math.max(0.5, w / grid.nx);
  const cellH = Math.max(0.5, h / grid.nz);
  for (let x = 0; x < grid.nx; x += 1) {
    for (let z = 0; z < grid.nz; z += 1) {
      if (!grid.occupied[x * grid.nz + z]) continue;
      parts.push(`<rect x="${(x * cellW).toFixed(2)}" y="${(z * cellH).toFixed(2)}" width="${cellW.toFixed(2)}" height="${cellH.toFixed(2)}" fill="#333"/>`);
    }
  }
  for (const [category, list] of Object.entries(candidates)) {
    const color = colors[category] || '#fff';
    for (let i = 0; i < list.length; i += Math.max(1, Math.floor(list.length / 2000))) {
      const p = list[i];
      parts.push(`<circle cx="${sx(p.wx).toFixed(2)}" cy="${sz(p.wz).toFixed(2)}" r="1" fill="${color}" opacity="0.45"/>`);
    }
  }
  for (const p of selected) {
    parts.push(`<circle cx="${sx(p.wx).toFixed(2)}" cy="${sz(p.wz).toFixed(2)}" r="2.2" fill="${colors[p.category] || '#fff'}" opacity="0.9"/>`);
  }
  parts.push('</svg>');
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, parts.join('\n'), 'utf8');
}

async function main() {
  const args = parseArgs(process.argv);
  const runtimeMeta = JSON.parse(fs.readFileSync(args.runtimeMeta, 'utf8'));
  const scene = sceneMinMax(runtimeMeta.sceneBounds);
  const rng = createRng(args.seed);
  const { clean, collision, rejected } = cleanBuildingComponents(runtimeMeta.componentRecords || [], args);
  if (clean.length === 0) throw new Error('No clean building components survived footprint filtering.');
  const grid = buildGrid(clean, scene, args);
  const dist = distanceToOccupied(grid, Math.max(args.plazaMaxDistCells, args.streetMaxDistCells) + 4);
  const candidates = classifyCells(grid, dist, args);
  const selected = chooseBasePositions(candidates, args, rng);
  addSkyAndFarBases(selected, clean, scene, args, rng);
  const collisionIndex = buildCollisionIndex(collision, grid, args);
  const { records, rejected: poseRejected } = expandPoses(selected, scene, grid, collisionIndex, args, rng);

  fs.mkdirSync(path.dirname(args.output), { recursive: true });
  const out = fs.createWriteStream(args.output, { encoding: 'utf8' });
  for (const record of records) out.write(`${JSON.stringify(record)}\n`);
  out.end();

  const categoryBaseCounts = {};
  const categoryPoseCounts = {};
  for (const item of selected) categoryBaseCounts[item.category] = (categoryBaseCounts[item.category] || 0) + 1;
  for (const record of records) categoryPoseCounts[record.sample_category] = (categoryPoseCounts[record.sample_category] || 0) + 1;
  const summary = {
    output: args.output,
    runtimeMeta: args.runtimeMeta,
    grid: { step: args.gridStep, nx: grid.nx, nz: grid.nz, minX: grid.minX, maxX: grid.maxX, minZ: grid.minZ, maxZ: grid.maxZ },
    cleanComponents: clean.length,
    totalComponents: (runtimeMeta.componentRecords || []).length,
    rejectedComponents: rejected,
    candidateCells: Object.fromEntries(Object.entries(candidates).map(([k, v]) => [k, v.length])),
    selectedBasePositions: selected.length,
    poseCount: records.length,
    categoryBaseCounts,
    categoryPoseCounts,
    rejectedPoses: poseRejected,
    args,
  };
  fs.mkdirSync(path.dirname(args.summary), { recursive: true });
  fs.writeFileSync(args.summary, JSON.stringify(summary, null, 2), 'utf8');
  writePreview(args.preview, grid, candidates, selected);
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
