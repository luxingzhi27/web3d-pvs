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
const GROUND_GRID_DIVISIONS = Object.freeze([24, 24]);
const YAW_DEGREES = Object.freeze([0, 90, 180, 270]);
const PITCH_DEGREES = Object.freeze([-15, 0, 15]);
const VIEWCELL_RADIUS = 0.75;
const SURFACE_EPSILON = 0.05;
const CAMERA_CLEARANCE = VIEWCELL_RADIUS + SURFACE_EPSILON;
const PVS_BACK_OFFSET = VIEWCELL_RADIUS / Math.tan(RENDER_FOV_Y * 0.5 * RAD);
const SPLIT_SEED = 20260911;
const SPLIT_NAMES = Object.freeze(['train', 'calibration', 'validation', 'test']);
const BVH_LEAF_SIZE = 16;
const GROUND_CAMERA_HEIGHT = 1.7;
const TERRAIN_NEAR_TOKEN = 'terrain_near';
const TERRAIN_FAR_TOKEN = 'terrain_far';

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

function integerVector(value, length, label) {
  if (!Array.isArray(value) || value.length !== length) {
    throw new PosePlanBlockedError(`${label} must contain ${length} integers`);
  }
  return value.map((item, axis) => {
    const number = Number(item);
    if (!Number.isInteger(number) || number <= 0) {
      throw new PosePlanBlockedError(`${label}[${axis}] must be a positive integer`);
    }
    return number;
  });
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
      sourceNodePath: String(record.sourceNodePath || ''),
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

function pathContains(record, token) {
  return String(record?.sourceNodePath || '').toLowerCase().includes(token);
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
  return buildTriangleIndex(Array.isArray(sourceScene?.renderables) ? sourceScene.renderables : []);
}

function buildTriangleIndex(renderables) {
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

function groundHeightAt(x, z, index) {
  let highest = -Infinity;
  const stack = [index.root];
  while (stack.length > 0) {
    const node = index.nodes[stack.pop()];
    if (x < node.min[0] || x > node.max[0] || z < node.min[2] || z > node.max[2]) continue;
    if (node.indices) {
      for (const triangle of node.indices) {
        const offset = triangle * 9;
        const ax = index.vertices[offset];
        const ay = index.vertices[offset + 1];
        const az = index.vertices[offset + 2];
        const bx = index.vertices[offset + 3];
        const by = index.vertices[offset + 4];
        const bz = index.vertices[offset + 5];
        const cx = index.vertices[offset + 6];
        const cy = index.vertices[offset + 7];
        const cz = index.vertices[offset + 8];
        const denominator = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
        if (Math.abs(denominator) <= 1e-12) continue;
        const u = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / denominator;
        const v = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / denominator;
        const w = 1 - u - v;
        if (u < -1e-8 || v < -1e-8 || w < -1e-8) continue;
        highest = Math.max(highest, u * ay + v * by + w * cy);
      }
      continue;
    }
    stack.push(node.left, node.right);
  }
  return highest;
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

function hash32(value) {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

function gridPosition(sceneBounds, gridIndex, gridDivisions) {
  return gridIndex.map((value, axis) => (
    sceneBounds.min[axis] + ((value + 0.5) / gridDivisions[axis]) * sceneBounds.size[axis]
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

function collectLegalPositions(sceneBounds, collisionIndex, clearance, gridDivisions) {
  const positions = [];
  let collisionRejected = 0;
  let candidateIndex = 0;
  for (let x = 0; x < gridDivisions[0]; x += 1) {
    for (let y = 0; y < gridDivisions[1]; y += 1) {
      for (let z = 0; z < gridDivisions[2]; z += 1) {
        const gridIndex = [x, y, z];
        const position = gridPosition(sceneBounds, gridIndex, gridDivisions);
        if (collisionAtPoint(position, collisionIndex, clearance)) {
          collisionRejected += 1;
        } else {
          positions.push({
            candidateIndex,
            gridIndex,
            position,
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

function collectGroundSurfacePositions(runtimeRecords, sourceScene, clearance, groundGridDivisions) {
  const renderables = Array.isArray(sourceScene?.renderables) ? sourceScene.renderables : [];
  const groundRenderables = renderables.filter((record) => pathContains(record, TERRAIN_NEAR_TOKEN));
  if (groundRenderables.length === 0) {
    throw new PosePlanBlockedError(`ground_surface_grid needs source nodes containing ${TERRAIN_NEAR_TOKEN}`);
  }
  const domainRenderables = renderables.filter((record) => !pathContains(record, TERRAIN_FAR_TOKEN));
  const domain = aggregateBounds(
    domainRenderables.map((renderable, originalIndex) => ({
      min: renderable.bounds.min,
      max: renderable.bounds.max,
      originalIndex,
    })),
    domainRenderables.map((_renderable, index) => index),
  );
  const groundIndex = buildTriangleIndex(groundRenderables);
  const obstacleRecords = runtimeRecords.filter((record) => (
    !pathContains(record, TERRAIN_NEAR_TOKEN) && !pathContains(record, TERRAIN_FAR_TOKEN)
  ));
  const obstacleIndex = obstacleRecords.length > 0 ? buildCollisionIndex(obstacleRecords, clearance) : null;
  const positions = [];
  let noGroundRejected = 0;
  let collisionRejected = 0;
  let candidateIndex = 0;
  for (let xIndex = 0; xIndex < groundGridDivisions[0]; xIndex += 1) {
    for (let zIndex = 0; zIndex < groundGridDivisions[1]; zIndex += 1) {
      const x = domain.min[0] + ((xIndex + 0.5) / groundGridDivisions[0]) * (domain.max[0] - domain.min[0]);
      const z = domain.min[2] + ((zIndex + 0.5) / groundGridDivisions[1]) * (domain.max[2] - domain.min[2]);
      const groundY = groundHeightAt(x, z, groundIndex);
      if (!Number.isFinite(groundY)) {
        noGroundRejected += 1;
      } else {
        const position = [x, groundY + GROUND_CAMERA_HEIGHT, z];
        const hitsGroundSurface = collidesWithGeometry(position, groundIndex, clearance);
        const hitsObstacle = obstacleIndex ? collides(position, obstacleIndex) : false;
        if (hitsGroundSurface || hitsObstacle) {
          collisionRejected += 1;
        } else {
          positions.push({
            candidateIndex,
            gridIndex: [xIndex, 0, zIndex],
            position,
            sampleCategory: 'ground_surface_grid',
          });
        }
      }
      candidateIndex += 1;
    }
  }
  return {
    positions,
    gridCandidateCount: candidateIndex,
    collisionRejected,
    noGroundRejected,
    groundTriangleCount: groundIndex.triangleCount,
    groundRenderableCount: groundRenderables.length,
    obstacleRecordCount: obstacleRecords.length,
    domainBounds: boundsFromValue(domain, 'ground surface domain'),
  };
}

function assignCenterGroupSplits(positions, splitSeed) {
  if (positions.length < 10) {
    throw new PosePlanBlockedError('at least ten legal camera centers are required for four splits');
  }
  const order = positions.slice().sort((left, right) => (
    compareNumbers(
      hash32(splitSeed ^ Math.imul(left.candidateIndex + 1, 0x9e3779b1)),
      hash32(splitSeed ^ Math.imul(right.candidateIndex + 1, 0x9e3779b1)),
    ) || compareNumbers(left.candidateIndex, right.candidateIndex)
  ));
  const validationCount = Math.round(positions.length * 0.10);
  const testCount = Math.round(positions.length * 0.10);
  const historicalTrainCount = positions.length - validationCount - testCount;
  const calibrationCount = Math.round(historicalTrainCount * 0.10);
  const trainPool = order.slice(0, historicalTrainCount).sort((left, right) => (
    compareNumbers(
      hash32((splitSeed + 1) ^ Math.imul(left.candidateIndex + 1, 0x85ebca6b)),
      hash32((splitSeed + 1) ^ Math.imul(right.candidateIndex + 1, 0x85ebca6b)),
    ) || compareNumbers(left.candidateIndex, right.candidateIndex)
  ));
  const splitByCenter = new Map();
  for (const position of trainPool.slice(0, calibrationCount)) {
    splitByCenter.set(position.candidateIndex, 'calibration');
  }
  for (const position of trainPool.slice(calibrationCount)) {
    splitByCenter.set(position.candidateIndex, 'train');
  }
  for (const position of order.slice(historicalTrainCount, historicalTrainCount + validationCount)) {
    splitByCenter.set(position.candidateIndex, 'validation');
  }
  for (const position of order.slice(historicalTrainCount + validationCount)) {
    splitByCenter.set(position.candidateIndex, 'test');
  }
  return {
    splitByCenter,
    centerCounts: {
      train: historicalTrainCount - calibrationCount,
      calibration: calibrationCount,
      validation: validationCount,
      test: testCount,
    },
  };
}

function createRows(positions, splitByCenter) {
  const rows = [];
  for (const position of positions.slice().sort((left, right) => (
    compareNumbers(left.gridIndex[0], right.gridIndex[0])
    || compareNumbers(left.gridIndex[1], right.gridIndex[1])
    || compareNumbers(left.gridIndex[2], right.gridIndex[2])
  ))) {
    const split = splitByCenter.get(position.candidateIndex);
      for (const yawDeg of YAW_DEGREES) {
        for (const pitchDeg of PITCH_DEGREES) {
          rows.push({
            pose_index: rows.length,
            split,
            sample_category: position.sampleCategory || 'free_space_grid',
            sample_category_id: 0,
            camera_pos: position.position.slice(),
            camera_forward: directionFor(yawDeg, pitchDeg),
            viewcell_shape: 'horizontal_disk',
            viewcell_half_extent: [VIEWCELL_RADIUS, VIEWCELL_RADIUS, 0],
            viewcell_radius: VIEWCELL_RADIUS,
            pvs_back_offset: PVS_BACK_OFFSET,
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
          });
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
  const placementMode = options.placementMode || 'volume_grid';
  const gridDivisions = integerVector(
    options.gridDivisions || GRID_DIVISIONS,
    3,
    'gridDivisions',
  );
  const groundGridDivisions = integerVector(
    options.groundGridDivisions || GROUND_GRID_DIVISIONS,
    2,
    'groundGridDivisions',
  );
  const splitSeed = Number(options.splitSeed ?? SPLIT_SEED);
  if (!Number.isInteger(splitSeed) || splitSeed < 0) {
    throw new PosePlanBlockedError('splitSeed must be a non-negative integer');
  }
  if (!['volume_grid', 'ground_surface_grid'].includes(placementMode)) {
    throw new PosePlanBlockedError(`unsupported placement mode ${placementMode}`);
  }
  if (placementMode === 'ground_surface_grid' && !options.sourceScene) {
    throw new PosePlanBlockedError('ground_surface_grid requires --source-scene');
  }
  const legalityMode = placementMode === 'ground_surface_grid'
    ? 'ground_surface_grid'
    : options.sourceScene ? 'geometry' : 'aabb';
  let collisionIndex = null;
  let legal;
  if (placementMode === 'ground_surface_grid') {
    legal = collectGroundSurfacePositions(
      audited.records, options.sourceScene, clearance, groundGridDivisions,
    );
  } else {
    collisionIndex = legalityMode === 'geometry'
      ? buildTriangleCollisionIndex(options.sourceScene)
      : buildCollisionIndex(audited.records, clearance);
    legal = collectLegalPositions(sceneBounds, collisionIndex, clearance, gridDivisions);
  }
  if (legal.positions.length === 0) {
    throw new PosePlanBlockedError(
      `no collision-free grid positions remain after rejecting ${legal.collisionRejected} positions`,
    );
  }
  const { centerCounts: splitCenterCount, splitByCenter } = assignCenterGroupSplits(
    legal.positions, splitSeed,
  );
  const rows = createRows(legal.positions, splitByCenter);
  const splitPoseCount = countBySplit(rows);
  if (SPLIT_NAMES.some((name) => splitPoseCount[name] === 0)) {
    throw new PosePlanBlockedError('one or more center-group splits has no pose');
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
      semantics: legalityMode === 'ground_surface_grid'
        ? 'fixed XZ grid over the non-far scene domain, projected downward onto terrain_near; camera centers are offset upward and clearance-tested'
        : legalityMode === 'geometry'
          ? 'sceneBounds interior grid points whose nearest source-scene triangle surface is farther than the fixed clearance; no volume or navigability is claimed'
          : 'sceneBounds interior grid points outside expanded runtime resource AABBs; navigability is not claimed',
      mode: legalityMode,
      collisionInput: legalityMode === 'geometry'
        ? 'source-scene-triangles'
        : legalityMode === 'ground_surface_grid' ? 'terrain_near triangles plus non-terrain runtime AABBs' : audited.source,
      sourceScene: options.sourceScenePath || null,
      sourceSceneTriangleCount: legalityMode === 'geometry' ? collisionIndex.triangleCount : null,
      groundRenderableCount: legal.groundRenderableCount ?? null,
      groundTriangleCount: legal.groundTriangleCount ?? null,
      groundCameraHeight: legalityMode === 'ground_surface_grid' ? GROUND_CAMERA_HEIGHT : null,
      groundNodeToken: legalityMode === 'ground_surface_grid' ? TERRAIN_NEAR_TOKEN : null,
      excludedDomainNodeToken: legalityMode === 'ground_surface_grid' ? TERRAIN_FAR_TOKEN : null,
      cameraDomainBounds: legal.domainBounds ?? sceneBounds,
      obstacleRecordCount: legal.obstacleRecordCount ?? null,
      closedVolumeClassification: 'not performed; non-watertight shells are surface-only',
      viewcellShape: 'horizontal_disk',
      viewcellRadius: VIEWCELL_RADIUS,
      surfaceEpsilon: SURFACE_EPSILON,
      safetyRadius: CAMERA_CLEARANCE,
      safetyRadiusSemantics: 'center-to-surface distance must cover the horizontal-disk radius plus surface epsilon',
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
      expandedAabbCollision: legalityMode === 'aabb' || legalityMode === 'ground_surface_grid',
      nearestTriangleSurfaceDistance: legalityMode === 'geometry' || legalityMode === 'ground_surface_grid',
      gridDivisions: legalityMode === 'ground_surface_grid'
        ? [groundGridDivisions[0], 1, groundGridDivisions[1]] : gridDivisions.slice(),
      gridCandidateCount: legal.gridCandidateCount,
      noGroundIntersectionPositionCount: legal.noGroundRejected ?? 0,
      collisionRejectedPositionCount: legal.collisionRejected,
      legalPositionCount: legal.positions.length,
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
      viewcellShape: 'horizontal_disk',
      viewcellRadius: VIEWCELL_RADIUS,
      verticalDisplacement: 0,
      pvsBackOffset: PVS_BACK_OFFSET,
      backOffsetFormula: 'viewcellRadius / tan(renderFovY / 2)',
      surfaceEpsilon: SURFACE_EPSILON,
      cameraSafetyRadius: CAMERA_CLEARANCE,
    },
    centerGroupSplit: {
      seed: splitSeed,
      assignment: 'deterministic seeded random assignment of physical camera centers',
      procedure: '80/10/10 train/validation/test, then 10% of the initial train centers become calibration',
      ratios: { train: 0.72, calibration: 0.08, validation: 0.10, test: 0.10 },
      splitCenterCount,
      splitPoseCount,
      allOrientationsAtOneCenterStayInOneSplit: true,
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

function parseIntegerList(value, length, label) {
  const values = String(value || '').split(',').map((item) => Number(item.trim()));
  if (values.length !== length || values.some((item) => !Number.isInteger(item) || item <= 0)) {
    throw new Error(`${label} must contain ${length} comma-separated positive integers`);
  }
  return values;
}

function parseArgs(argv) {
  const args = {
    runtimeMeta: '',
    output: '',
    summary: '',
    conversionManifest: '',
    placementMode: 'volume_grid',
    gridDivisions: GRID_DIVISIONS.slice(),
    groundGridDivisions: GROUND_GRID_DIVISIONS.slice(),
    splitSeed: SPLIT_SEED,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--runtime-meta') args.runtimeMeta = path.resolve(argv[++index] || '');
    else if (argument === '--source-scene') args.sourceScene = path.resolve(argv[++index] || '');
    else if (argument === '--output') args.output = path.resolve(argv[++index] || '');
    else if (argument === '--summary') args.summary = path.resolve(argv[++index] || '');
    else if (argument === '--conversion-manifest') args.conversionManifest = path.resolve(argv[++index] || '');
    else if (argument === '--placement-mode') args.placementMode = String(argv[++index] || '');
    else if (argument === '--grid-divisions') args.gridDivisions = parseIntegerList(argv[++index], 3, '--grid-divisions');
    else if (argument === '--ground-grid-divisions') args.groundGridDivisions = parseIntegerList(argv[++index], 2, '--ground-grid-divisions');
    else if (argument === '--split-seed') args.splitSeed = Number(argv[++index]);
    else if (argument === '--help') args.help = true;
    else throw new Error(`unknown argument: ${argument}`);
  }
  if (args.help) return args;
  if (!args.runtimeMeta || !args.output) {
    throw new Error(
      'Usage: generate_pose_plan.mjs --runtime-meta runtimeVisibilityMeta.json --output pose_plan.jsonl [--source-scene scene.gltf|scene.glb] [--placement-mode volume_grid|ground_surface_grid] [--grid-divisions 16,4,16] [--ground-grid-divisions 24,24] [--split-seed N] [--summary audit.json]',
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
    console.log('generate_pose_plan.mjs --runtime-meta runtimeVisibilityMeta.json --output pose_plan.jsonl [--source-scene scene.gltf|scene.glb] [--placement-mode volume_grid|ground_surface_grid] [--grid-divisions 16,4,16] [--ground-grid-divisions 24,24] [--split-seed N] [--summary audit.json]');
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
    placementMode: args.placementMode,
    gridDivisions: args.gridDivisions,
    groundGridDivisions: args.groundGridDivisions,
    splitSeed: args.splitSeed,
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
