#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export const POSE_PLAN_SCHEMA = 'pvs-standard-graphics-pose-plan-v1';
export const POSE_PLAN_AUDIT_SCHEMA = 'pvs-standard-graphics-pose-plan-audit-v1';

const RAD = Math.PI / 180;
const MODEL_FOV_Y = 66;
const RENDER_FOV_Y = 60;
const WIDTH = 512;
const HEIGHT = 288;
const ASPECT = WIDTH / HEIGHT;
const GRID_DIVISIONS = Object.freeze([16, 4, 16]);
const BLOCK_DIVISIONS = Object.freeze([4, 2, 4]);
const YAW_DEGREES = Object.freeze([0, 90, 180, 270]);
const PITCH_DEGREES = Object.freeze([-15, 0, 15]);
const VIEWCELL_HALF_EXTENTS = Object.freeze([0.5, 0.5, 0.25]);
const VIEWCELL_OUTER_RADIUS = Math.hypot(...VIEWCELL_HALF_EXTENTS);
const SURFACE_EPSILON = 0.05;
const CAMERA_CLEARANCE = VIEWCELL_OUTER_RADIUS + SURFACE_EPSILON;
const SPLIT_NAMES = Object.freeze(['train', 'calibration', 'validation', 'test']);
const SPLIT_RATIOS = Object.freeze([0.72, 0.08, 0.10, 0.10]);
const BVH_LEAF_SIZE = 16;

export class PosePlanBlockedError extends Error {
  constructor(message) {
    super(`pose-plan blocked: ${message}`);
    this.name = 'PosePlanBlockedError';
    this.code = 'POSE_PLAN_BLOCKED';
  }
}

function finiteNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new PosePlanBlockedError(`${label} must contain finite numbers`);
  return number;
}

function vector3(value, label) {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new PosePlanBlockedError(`${label} must be a three-element array`);
  }
  return value.map((item, axis) => finiteNumber(item, `${label}[${axis}]`));
}

function boundsFromValue(value, label) {
  if (!value || typeof value !== 'object') {
    throw new PosePlanBlockedError(`${label} is missing`);
  }
  let min;
  let max;
  let source;
  if (value.min !== undefined || value.max !== undefined) {
    min = vector3(value.min, `${label}.min`);
    max = vector3(value.max, `${label}.max`);
    source = 'min-max';
  } else if (value.center !== undefined || value.size !== undefined) {
    const center = vector3(value.center, `${label}.center`);
    const size = vector3(value.size, `${label}.size`);
    if (size.some((item) => item < 0)) throw new PosePlanBlockedError(`${label}.size must be non-negative`);
    min = center.map((item, axis) => item - size[axis] * 0.5);
    max = center.map((item, axis) => item + size[axis] * 0.5);
    source = 'center-size';
  } else {
    throw new PosePlanBlockedError(`${label} needs min/max or center/size`);
  }
  for (let axis = 0; axis < 3; axis += 1) {
    if (max[axis] < min[axis]) throw new PosePlanBlockedError(`${label} has max < min on axis ${axis}`);
  }
  return {
    min,
    max,
    center: min.map((item, axis) => (item + max[axis]) * 0.5),
    size: min.map((item, axis) => max[axis] - item),
    source,
  };
}

function sourceRecords(runtimeMeta) {
  if (!runtimeMeta || typeof runtimeMeta !== 'object') {
    throw new PosePlanBlockedError('runtime metadata must be a JSON object');
  }
  if (Array.isArray(runtimeMeta.componentRecords)) {
    return { records: runtimeMeta.componentRecords, source: 'componentRecords', idField: 'componentGlobalId' };
  }
  if (Array.isArray(runtimeMeta.globalGlbRecords)) {
    return { records: runtimeMeta.globalGlbRecords, source: 'globalGlbRecords', idField: 'globalGlbId' };
  }
  throw new PosePlanBlockedError('runtime metadata has no componentRecords or globalGlbRecords');
}

function recordBounds(record, index, source) {
  const value = record?.bounds || record?.aabb;
  return boundsFromValue(value, `${source}[${index}].bounds`);
}

function auditRecords(runtimeMeta) {
  const source = sourceRecords(runtimeMeta);
  if (source.records.length === 0) throw new PosePlanBlockedError(`${source.source} is empty`);
  const ids = new Set();
  const records = source.records.map((record, index) => {
    if (!record || typeof record !== 'object') {
      throw new PosePlanBlockedError(`${source.source}[${index}] is not an object`);
    }
    const rawId = record[source.idField];
    const id = rawId === undefined ? index : Number(rawId);
    if (!Number.isInteger(id) || id < 0) {
      throw new PosePlanBlockedError(`${source.source}[${index}] has an invalid resource id`);
    }
    if (ids.has(id)) throw new PosePlanBlockedError(`${source.source} contains duplicate resource id ${id}`);
    ids.add(id);
    return {
      id,
      bounds: recordBounds(record, index, source.source),
      staticPvsEligible: record.staticPvsEligible !== false,
      alwaysResident: record.alwaysResident === true,
      alphaMode: record.alphaMode || null,
    };
  });
  const expectedCount = source.source === 'componentRecords'
    ? runtimeMeta.instanceCount ?? runtimeMeta.componentCount
    : runtimeMeta.globalGlbCount;
  if (expectedCount !== undefined && Number(expectedCount) !== records.length) {
    throw new PosePlanBlockedError(
      `${source.source} count ${records.length} disagrees with runtime metadata count ${expectedCount}`,
    );
  }
  return { ...source, records };
}

function clearanceForScene(sceneBounds) {
  if (sceneBounds.size.some((value) => value <= 0)) {
    throw new PosePlanBlockedError('sceneBounds must have positive extent on all three axes');
  }
  return CAMERA_CLEARANCE;
}

function expandedBounds(bounds, clearance) {
  return {
    min: bounds.min.map((value) => value - clearance),
    max: bounds.max.map((value) => value + clearance),
  };
}

function compareNumbers(left, right) {
  return left - right;
}

function compareItemIndices(items, leftIndex, rightIndex, axis) {
  const left = items[leftIndex];
  const right = items[rightIndex];
  return compareNumbers(left.center[axis], right.center[axis])
    || compareNumbers(left.min[axis], right.min[axis])
    || compareNumbers(left.center[(axis + 1) % 3], right.center[(axis + 1) % 3])
    || compareNumbers(left.originalIndex, right.originalIndex);
}

function aggregateBounds(items, indices) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const index of indices) {
    const item = items[index];
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], item.min[axis]);
      max[axis] = Math.max(max[axis], item.max[axis]);
    }
  }
  return { min, max };
}

function longestAxis(bounds) {
  const sizes = bounds.max.map((value, axis) => value - bounds.min[axis]);
  let axis = 0;
  if (sizes[1] > sizes[axis]) axis = 1;
  if (sizes[2] > sizes[axis]) axis = 2;
  return axis;
}

function buildCollisionIndex(records, clearance) {
  const items = records.map((record, originalIndex) => {
    const bounds = expandedBounds(record.bounds, clearance);
    return {
      ...bounds,
      center: bounds.min.map((value, axis) => (value + bounds.max[axis]) * 0.5),
      originalIndex,
    };
  });
  const nodes = [];
  function build(indices) {
    const bounds = aggregateBounds(items, indices);
    const nodeIndex = nodes.length;
    nodes.push(null);
    if (indices.length <= BVH_LEAF_SIZE) {
      nodes[nodeIndex] = { ...bounds, indices: indices.slice().sort(compareNumbers) };
      return nodeIndex;
    }
    const axis = longestAxis(bounds);
    const ordered = indices.slice().sort((left, right) => compareItemIndices(items, left, right, axis));
    const middle = Math.floor(ordered.length * 0.5);
    const left = build(ordered.slice(0, middle));
    const right = build(ordered.slice(middle));
    nodes[nodeIndex] = { ...bounds, left, right };
    return nodeIndex;
  }
  return { mode: 'aabb', items, nodes, root: build(items.map((_item, index) => index)) };
}

function pointInsideBounds(point, bounds) {
  return point[0] >= bounds.min[0] && point[0] <= bounds.max[0]
    && point[1] >= bounds.min[1] && point[1] <= bounds.max[1]
    && point[2] >= bounds.min[2] && point[2] <= bounds.max[2];
}

function collides(point, index) {
  const stack = [index.root];
  while (stack.length > 0) {
    const node = index.nodes[stack.pop()];
    if (!pointInsideBounds(point, node)) continue;
    if (node.indices) {
      for (const itemIndex of node.indices) {
        if (pointInsideBounds(point, index.items[itemIndex])) return true;
      }
      continue;
    }
    stack.push(node.left, node.right);
  }
  return false;
}

function pointAabbDistanceSquared(point, bounds) {
  let distanceSquared = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const distance = point[axis] < bounds.min[axis]
      ? bounds.min[axis] - point[axis]
      : point[axis] > bounds.max[axis] ? point[axis] - bounds.max[axis] : 0;
    distanceSquared += distance * distance;
  }
  return distanceSquared;
}

function pointSegmentDistanceSquared(pointX, pointY, pointZ, ax, ay, az, bx, by, bz) {
  const dx = bx - ax;
  const dy = by - ay;
  const dz = bz - az;
  const lengthSquared = dx * dx + dy * dy + dz * dz;
  if (lengthSquared <= 1e-20) {
    const px = pointX - ax;
    const py = pointY - ay;
    const pz = pointZ - az;
    return px * px + py * py + pz * pz;
  }
  const px = pointX - ax;
  const py = pointY - ay;
  const pz = pointZ - az;
  const t = Math.max(0, Math.min(1, (px * dx + py * dy + pz * dz) / lengthSquared));
  const qx = ax + t * dx - pointX;
  const qy = ay + t * dy - pointY;
  const qz = az + t * dz - pointZ;
  return qx * qx + qy * qy + qz * qz;
}

function pointTriangleDistanceSquared(point, vertices, offset) {
  const pointX = point[0];
  const pointY = point[1];
  const pointZ = point[2];
  const ax = vertices[offset];
  const ay = vertices[offset + 1];
  const az = vertices[offset + 2];
  const bx = vertices[offset + 3];
  const by = vertices[offset + 4];
  const bz = vertices[offset + 5];
  const cx = vertices[offset + 6];
  const cy = vertices[offset + 7];
  const cz = vertices[offset + 8];
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;
  const crossX = aby * acz - abz * acy;
  const crossY = abz * acx - abx * acz;
  const crossZ = abx * acy - aby * acx;
  if (crossX * crossX + crossY * crossY + crossZ * crossZ <= 1e-20) {
    return Math.min(
      pointSegmentDistanceSquared(pointX, pointY, pointZ, ax, ay, az, bx, by, bz),
      pointSegmentDistanceSquared(pointX, pointY, pointZ, ax, ay, az, cx, cy, cz),
      pointSegmentDistanceSquared(pointX, pointY, pointZ, bx, by, bz, cx, cy, cz),
    );
  }

  const apx = pointX - ax;
  const apy = pointY - ay;
  const apz = pointZ - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) return apx * apx + apy * apy + apz * apz;

  const bpx = pointX - bx;
  const bpy = pointY - by;
  const bpz = pointZ - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return bpx * bpx + bpy * bpy + bpz * bpz;

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const qx = ax + v * abx - pointX;
    const qy = ay + v * aby - pointY;
    const qz = az + v * abz - pointZ;
    return qx * qx + qy * qy + qz * qz;
  }

  const cpx = pointX - cx;
  const cpy = pointY - cy;
  const cpz = pointZ - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return cpx * cpx + cpy * cpy + cpz * cpz;

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const qx = ax + w * acx - pointX;
    const qy = ay + w * acy - pointY;
    const qz = az + w * acz - pointZ;
    return qx * qx + qy * qy + qz * qz;
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + d5 - d6);
    const qx = bx + w * (cx - bx) - pointX;
    const qy = by + w * (cy - by) - pointY;
    const qz = bz + w * (cz - bz) - pointZ;
    return qx * qx + qy * qy + qz * qz;
  }

  const inverseDenominator = 1 / (va + vb + vc);
  const v = vb * inverseDenominator;
  const w = vc * inverseDenominator;
  const qx = ax + abx * v + acx * w - pointX;
  const qy = ay + aby * v + acy * w - pointY;
  const qz = az + abz * v + acz * w - pointZ;
  return qx * qx + qy * qy + qz * qz;
}

function triangleRangeBounds(bounds, order, start, end) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let orderIndex = start; orderIndex < end; orderIndex += 1) {
    const triangle = order[orderIndex];
    const offset = triangle * 6;
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], bounds[offset + axis]);
      max[axis] = Math.max(max[axis], bounds[offset + 3 + axis]);
    }
  }
  return { min, max };
}

function triangleCenterCompare(centers, left, right, axis) {
  const leftValue = centers[left * 3 + axis];
  const rightValue = centers[right * 3 + axis];
  return leftValue - rightValue || left - right;
}

function buildTriangleCollisionIndex(sourceScene) {
  const renderables = Array.isArray(sourceScene?.renderables) ? sourceScene.renderables : [];
  const triangleCount = renderables.reduce((sum, renderable) => (
    sum + Math.floor((renderable?.indices?.length || 0) / 3)
  ), 0);
  if (triangleCount === 0) throw new PosePlanBlockedError('source scene has no static triangles');
  const vertices = new Float32Array(triangleCount * 9);
  const bounds = new Float32Array(triangleCount * 6);
  const centers = new Float32Array(triangleCount * 3);
  let triangleIndex = 0;
  for (const renderable of renderables) {
    const positions = renderable.positions;
    const indices = renderable.indices;
    if (!positions || !indices || indices.length % 3 !== 0) {
      throw new PosePlanBlockedError('source scene contains an invalid triangle primitive');
    }
    for (let index = 0; index < indices.length; index += 3) {
      const vertexOffset = triangleIndex * 9;
      const boundOffset = triangleIndex * 6;
      const centerOffset = triangleIndex * 3;
      const vertexIndices = [indices[index], indices[index + 1], indices[index + 2]];
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (let corner = 0; corner < 3; corner += 1) {
        const sourceIndex = Number(vertexIndices[corner]);
        if (!Number.isInteger(sourceIndex) || sourceIndex < 0 || sourceIndex * 3 + 2 >= positions.length) {
          throw new PosePlanBlockedError('source scene contains an out-of-range triangle index');
        }
        for (let axis = 0; axis < 3; axis += 1) {
          const value = finiteNumber(positions[sourceIndex * 3 + axis], 'source triangle position');
          vertices[vertexOffset + corner * 3 + axis] = value;
          min[axis] = Math.min(min[axis], value);
          max[axis] = Math.max(max[axis], value);
        }
      }
      for (let axis = 0; axis < 3; axis += 1) {
        bounds[boundOffset + axis] = min[axis];
        bounds[boundOffset + 3 + axis] = max[axis];
        centers[centerOffset + axis] = (min[axis] + max[axis]) * 0.5;
      }
      triangleIndex += 1;
    }
  }

  const order = new Uint32Array(triangleCount);
  for (let index = 0; index < triangleCount; index += 1) order[index] = index;
  const nodes = [];
  function build(start, end) {
    const nodeBounds = triangleRangeBounds(bounds, order, start, end);
    const nodeIndex = nodes.length;
    nodes.push(null);
    if (end - start <= BVH_LEAF_SIZE) {
      nodes[nodeIndex] = {
        min: nodeBounds.min,
        max: nodeBounds.max,
        indices: Array.from(order.slice(start, end)),
      };
      return nodeIndex;
    }
    const axis = longestAxis(nodeBounds);
    const sorted = Array.from(order.slice(start, end));
    sorted.sort((left, right) => triangleCenterCompare(centers, left, right, axis));
    order.set(sorted, start);
    const middle = start + Math.floor((end - start) * 0.5);
    const left = build(start, middle);
    const right = build(middle, end);
    nodes[nodeIndex] = { min: nodeBounds.min, max: nodeBounds.max, left, right };
    return nodeIndex;
  }
  return {
    mode: 'geometry',
    vertices,
    bounds,
    centers,
    triangleCount,
    nodes,
    root: build(0, triangleCount),
  };
}

function nearestTriangleDistanceSquared(point, index, maximumDistanceSquared) {
  let nearest = Infinity;
  const stack = [index.root];
  while (stack.length > 0) {
    const node = index.nodes[stack.pop()];
    if (pointAabbDistanceSquared(point, node) > Math.min(nearest, maximumDistanceSquared)) continue;
    if (node.indices) {
      for (const triangle of node.indices) {
        const distanceSquared = pointTriangleDistanceSquared(point, index.vertices, triangle * 9);
        if (distanceSquared < nearest) nearest = distanceSquared;
        if (nearest <= maximumDistanceSquared) return nearest;
      }
      continue;
    }
    stack.push(node.left, node.right);
  }
  return nearest;
}

function collidesWithGeometry(point, index, clearance) {
  const threshold = clearance * clearance;
  return nearestTriangleDistanceSquared(point, index, threshold) <= threshold;
}

function collisionAtPoint(point, index, clearance) {
  return index.mode === 'geometry'
    ? collidesWithGeometry(point, index, clearance)
    : collides(point, index);
}

function morton3(x, y, z) {
  let result = 0;
  for (let bit = 0; bit < 8; bit += 1) {
    result |= ((x >> bit) & 1) << (bit * 3);
    result |= ((y >> bit) & 1) << (bit * 3 + 1);
    result |= ((z >> bit) & 1) << (bit * 3 + 2);
  }
  return result >>> 0;
}

function splitBlockCounts(blockCount) {
  if (blockCount < SPLIT_NAMES.length) {
    throw new PosePlanBlockedError(
      `only ${blockCount} occupied spatial blocks are available; four disjoint splits need at least four`,
    );
  }
  const ideal = SPLIT_RATIOS.map((ratio) => ratio * blockCount);
  const counts = ideal.map((value) => Math.max(1, Math.floor(value)));
  while (counts.reduce((sum, value) => sum + value, 0) > blockCount) {
    let selected = -1;
    let excess = -Infinity;
    for (let index = 0; index < counts.length; index += 1) {
      if (counts[index] <= 1) continue;
      const value = counts[index] - ideal[index];
      if (value > excess) {
        excess = value;
        selected = index;
      }
    }
    counts[selected] -= 1;
  }
  while (counts.reduce((sum, value) => sum + value, 0) < blockCount) {
    let selected = 0;
    let deficit = -Infinity;
    for (let index = 0; index < counts.length; index += 1) {
      const value = ideal[index] - counts[index];
      if (value > deficit) {
        deficit = value;
        selected = index;
      }
    }
    counts[selected] += 1;
  }
  return counts;
}

function blockCoordinate(gridIndex) {
  return gridIndex.map((value, axis) => Math.floor(value * BLOCK_DIVISIONS[axis] / GRID_DIVISIONS[axis]));
}

function gridPosition(sceneBounds, gridIndex) {
  return gridIndex.map((value, axis) => (
    sceneBounds.min[axis] + ((value + 0.5) / GRID_DIVISIONS[axis]) * sceneBounds.size[axis]
  ));
}

function directionFor(yawDeg, pitchDeg) {
  const yaw = yawDeg * RAD;
  const pitch = pitchDeg * RAD;
  const direction = [
    -Math.sin(yaw) * Math.cos(pitch),
    Math.sin(pitch),
    -Math.cos(yaw) * Math.cos(pitch),
  ];
  const length = Math.hypot(...direction);
  return direction.map((value) => {
    const normalized = value / length;
    return normalized === 0 ? 0 : normalized;
  });
}

function collectLegalPositions(sceneBounds, collisionIndex, clearance) {
  const positions = [];
  let collisionRejected = 0;
  let candidateIndex = 0;
  for (let x = 0; x < GRID_DIVISIONS[0]; x += 1) {
    for (let y = 0; y < GRID_DIVISIONS[1]; y += 1) {
      for (let z = 0; z < GRID_DIVISIONS[2]; z += 1) {
        const gridIndex = [x, y, z];
        const position = gridPosition(sceneBounds, gridIndex);
        if (collisionAtPoint(position, collisionIndex, clearance)) {
          collisionRejected += 1;
        } else {
          const block = blockCoordinate(gridIndex);
          positions.push({
            candidateIndex,
            gridIndex,
            position,
            block,
            blockMorton: morton3(block[0], block[1], block[2]),
          });
        }
        candidateIndex += 1;
      }
    }
  }
  return {
    positions,
    gridCandidateCount: candidateIndex,
    collisionRejected,
  };
}

function groupPositionsByBlock(positions) {
  const groups = new Map();
  for (const position of positions) {
    const key = position.block.join(',');
    let group = groups.get(key);
    if (!group) {
      group = {
        block: position.block,
        blockMorton: position.blockMorton,
        positions: [],
      };
      groups.set(key, group);
    }
    group.positions.push(position);
  }
  return [...groups.values()].sort((left, right) => (
    compareNumbers(left.blockMorton, right.blockMorton)
    || compareNumbers(left.block[0], right.block[0])
    || compareNumbers(left.block[1], right.block[1])
    || compareNumbers(left.block[2], right.block[2])
  ));
}

function assignSplits(groups) {
  const counts = splitBlockCounts(groups.length);
  const splitByBlock = new Map();
  let offset = 0;
  for (let splitIndex = 0; splitIndex < SPLIT_NAMES.length; splitIndex += 1) {
    for (let index = 0; index < counts[splitIndex]; index += 1) {
      const group = groups[offset + index];
      splitByBlock.set(group.block.join(','), SPLIT_NAMES[splitIndex]);
    }
    offset += counts[splitIndex];
  }
  return { counts, splitByBlock };
}

function createRows(groups, splitByBlock) {
  const rows = [];
  for (const group of groups) {
    const split = splitByBlock.get(group.block.join(','));
    for (const position of group.positions.sort((left, right) => (
      compareNumbers(left.gridIndex[0], right.gridIndex[0])
      || compareNumbers(left.gridIndex[1], right.gridIndex[1])
      || compareNumbers(left.gridIndex[2], right.gridIndex[2])
    ))) {
      for (const yawDeg of YAW_DEGREES) {
        for (const pitchDeg of PITCH_DEGREES) {
          rows.push({
            pose_index: rows.length,
            split,
            sample_category: 'free_space_grid',
            sample_category_id: 0,
            camera_pos: position.position.slice(),
            camera_forward: directionFor(yawDeg, pitchDeg),
            viewcell_shape: 'camera_aligned_box',
            viewcell_half_extent: VIEWCELL_HALF_EXTENTS.slice(),
            viewcell_radius: VIEWCELL_OUTER_RADIUS,
            yaw_deg: yawDeg,
            pitch_deg: pitchDeg,
            fov_y: MODEL_FOV_Y,
            pvs_fov_y: MODEL_FOV_Y,
            render_fov_y: RENDER_FOV_Y,
            aspect: ASPECT,
            width: WIDTH,
            height: HEIGHT,
            position_index: position.candidateIndex,
            grid_index: position.gridIndex.slice(),
            spatial_block: position.block.slice(),
            spatial_block_morton: position.blockMorton,
          });
        }
      }
    }
  }
  return rows;
}

function countBySplit(rows) {
  const counts = Object.fromEntries(SPLIT_NAMES.map((name) => [name, 0]));
  for (const row of rows) counts[row.split] += 1;
  return counts;
}

function sourceAuditFrom(runtimeMeta, conversionManifest) {
  return conversionManifest?.sourceAudit || runtimeMeta.sourceAudit || null;
}

export function generatePosePlan(runtimeMeta, options = {}) {
  const sceneBounds = boundsFromValue(runtimeMeta?.sceneBounds, 'sceneBounds');
  const audited = auditRecords(runtimeMeta);
  const clearance = clearanceForScene(sceneBounds);
  const legalityMode = options.sourceScene ? 'geometry' : 'aabb';
  const collisionIndex = legalityMode === 'geometry'
    ? buildTriangleCollisionIndex(options.sourceScene)
    : buildCollisionIndex(audited.records, clearance);
  const legal = collectLegalPositions(sceneBounds, collisionIndex, clearance);
  if (legal.positions.length === 0) {
    throw new PosePlanBlockedError(
      `no collision-free grid positions remain after rejecting ${legal.collisionRejected} positions`,
    );
  }
  const groups = groupPositionsByBlock(legal.positions);
  const { counts: splitBlockCount, splitByBlock } = assignSplits(groups);
  const rows = createRows(groups, splitByBlock);
  const splitPoseCount = countBySplit(rows);
  if (SPLIT_NAMES.some((name) => splitPoseCount[name] === 0)) {
    throw new PosePlanBlockedError('one or more spatial splits has no pose');
  }
  const staticPvsEligibleCount = audited.records.filter((record) => record.staticPvsEligible).length;
  const alwaysResidentCount = audited.records.filter((record) => record.alwaysResident).length;
  const audit = {
    schema: POSE_PLAN_AUDIT_SCHEMA,
    generatedBy: 'slm-graphics-scene-importer-g1',
    runtimeMeta: options.runtimeMetaPath || null,
    conversionManifest: options.conversionManifestPath || null,
    sceneName: runtimeMeta.sceneName || null,
    sourceSchema: runtimeMeta.schemaVersion ?? null,
    sceneBounds,
    legalRegion: {
      semantics: legalityMode === 'geometry'
        ? 'sceneBounds interior grid points whose nearest source-scene triangle surface is farther than the fixed clearance; no volume or navigability is claimed'
        : 'sceneBounds interior grid points outside expanded runtime resource AABBs; navigability is not claimed',
      mode: legalityMode,
      collisionInput: legalityMode === 'geometry' ? 'source-scene-triangles' : audited.source,
      sourceScene: options.sourceScenePath || null,
      sourceSceneTriangleCount: legalityMode === 'geometry' ? collisionIndex.triangleCount : null,
      closedVolumeClassification: 'not performed; non-watertight shells are surface-only',
      viewcellShape: 'camera_aligned_box',
      viewcellHalfExtent: VIEWCELL_HALF_EXTENTS.slice(),
      viewcellOuterRadius: VIEWCELL_OUTER_RADIUS,
      surfaceEpsilon: SURFACE_EPSILON,
      safetyRadius: CAMERA_CLEARANCE,
      safetyRadiusSemantics: 'center-to-surface distance must cover the view-cell box outer radius plus surface epsilon',
      resourceRecordCount: audited.records.length,
      staticPvsEligibleResourceCount: staticPvsEligibleCount,
      staticPvsIneligibleResourceCount: audited.records.length - staticPvsEligibleCount,
      alwaysResidentResourceCount: alwaysResidentCount,
      alphaModeCounts: audited.records.reduce((counts, record) => {
        const mode = record.alphaMode || 'UNKNOWN';
        counts[mode] = (counts[mode] || 0) + 1;
        return counts;
      }, {}),
      cameraClearance: clearance,
      expandedAabbCollision: legalityMode === 'aabb',
      nearestTriangleSurfaceDistance: legalityMode === 'geometry',
      gridDivisions: GRID_DIVISIONS.slice(),
      gridCandidateCount: legal.gridCandidateCount,
      collisionRejectedPositionCount: legal.collisionRejected,
      legalPositionCount: legal.positions.length,
      occupiedSpatialBlockCount: groups.length,
      totalSpatialBlockCount: BLOCK_DIVISIONS.reduce((product, value) => product * value, 1),
    },
    protocol: {
      modelFovY: MODEL_FOV_Y,
      renderFovY: RENDER_FOV_Y,
      width: WIDTH,
      height: HEIGHT,
      aspect: ASPECT,
      yaws: YAW_DEGREES.slice(),
      pitches: PITCH_DEGREES.slice(),
      poseCountPerPosition: YAW_DEGREES.length * PITCH_DEGREES.length,
      viewcellShape: 'camera_aligned_box',
      viewcellHalfExtent: VIEWCELL_HALF_EXTENTS.slice(),
      viewcellOuterRadius: VIEWCELL_OUTER_RADIUS,
      surfaceEpsilon: SURFACE_EPSILON,
      cameraSafetyRadius: CAMERA_CLEARANCE,
    },
    spatialSplit: {
      blockDivisions: BLOCK_DIVISIONS.slice(),
      ordering: '3D Morton block order, occupied blocks only',
      ratios: Object.fromEntries(SPLIT_NAMES.map((name, index) => [name, SPLIT_RATIOS[index]])),
      splitBlockCount: Object.fromEntries(SPLIT_NAMES.map((name, index) => [name, splitBlockCount[index]])),
      splitPoseCount,
      noBlockAppearsInMultipleSplits: true,
    },
    poseCount: rows.length,
    sourceAudit: sourceAuditFrom(runtimeMeta, options.conversionManifest),
    semantics: 'Raw model-sampling poses; fov_y is the 66 degree model query FOV and render_fov_y records the 60 degree display protocol.',
  };
  return { rows, audit };
}

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`${filePath}: cannot read JSON: ${error.message}`);
  }
}

function writePlan(filePath, rows) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`, 'utf8');
}

function parseArgs(argv) {
  const args = { runtimeMeta: '', output: '', summary: '', conversionManifest: '' };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--runtime-meta') args.runtimeMeta = path.resolve(argv[++index] || '');
    else if (argument === '--source-scene') args.sourceScene = path.resolve(argv[++index] || '');
    else if (argument === '--output') args.output = path.resolve(argv[++index] || '');
    else if (argument === '--summary') args.summary = path.resolve(argv[++index] || '');
    else if (argument === '--conversion-manifest') args.conversionManifest = path.resolve(argv[++index] || '');
    else if (argument === '--help') args.help = true;
    else throw new Error(`unknown argument: ${argument}`);
  }
  if (args.help) return args;
  if (!args.runtimeMeta || !args.output) {
    throw new Error(
      'Usage: generate_pose_plan.mjs --runtime-meta runtimeVisibilityMeta.json --output pose_plan.jsonl [--source-scene scene.gltf|scene.glb] [--summary audit.json]',
    );
  }
  if (!args.summary) args.summary = args.output.replace(/\.jsonl$/i, '_summary.json');
  if (!args.conversionManifest) {
    const sibling = path.join(path.dirname(args.runtimeMeta), 'conversionManifest.json');
    if (fs.existsSync(sibling)) args.conversionManifest = sibling;
  }
  return args;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log('generate_pose_plan.mjs --runtime-meta runtimeVisibilityMeta.json --output pose_plan.jsonl [--source-scene scene.gltf|scene.glb] [--summary audit.json]');
    return null;
  }
  const runtimeMeta = readJson(args.runtimeMeta);
  const conversionManifest = args.conversionManifest ? readJson(args.conversionManifest) : null;
  if (conversionManifest?.unitCount !== undefined
    && runtimeMeta.instanceCount !== undefined
    && Number(conversionManifest.unitCount) !== Number(runtimeMeta.instanceCount)) {
    throw new PosePlanBlockedError('conversion manifest and runtime metadata unit counts disagree');
  }
  let sourceScene = null;
  if (args.sourceScene) {
    const { readGltfScene } = await import('./read_gltf.mjs');
    sourceScene = await readGltfScene(args.sourceScene);
  }
  const result = generatePosePlan(runtimeMeta, {
    runtimeMetaPath: args.runtimeMeta,
    conversionManifestPath: args.conversionManifest || null,
    sourceScenePath: args.sourceScene || null,
    conversionManifest,
    sourceScene,
  });
  writePlan(args.output, result.rows);
  fs.mkdirSync(path.dirname(args.summary), { recursive: true });
  fs.writeFileSync(args.summary, `${JSON.stringify(result.audit, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({ status: 'ok', output: args.output, summary: args.summary, poseCount: result.rows.length }, null, 2));
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
