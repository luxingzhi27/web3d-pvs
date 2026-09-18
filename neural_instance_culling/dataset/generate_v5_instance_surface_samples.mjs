#!/usr/bin/env node
/* Build the V5 local surface asset from real component/renderable units.
 *
 * The canonical output directory contains the same dense binaries, NumPy
 * arrays, and strict surface_manifest.json written by dataset/v5's Python
 * implementation. A component is never represented by the complete GLB when
 * a GLB contains multiple components: usable triangles are assigned from
 * their world-space AABB and centroid to one component AABB.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const DEFAULT_ASSETS_DIR = path.join(REPO_ROOT, 'hkust-v3', 'assets');
const DEFAULT_RUNTIME_META = path.join(DEFAULT_ASSETS_DIR, 'runtimeVisibilityMeta.json');
const DEFAULT_GLB_INDEX = path.join(DEFAULT_ASSETS_DIR, 'glbIndex.json');
const DEFAULT_OUTPUT_DIR = path.join(
  REPO_ROOT,
  'neural_instance_culling',
  'dataset',
  'out',
  'v5_instance_surface_samples',
);

const THREE_URL = pathToFileURL(
  path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'build', 'three.module.js'),
).href;
const GLTF_LOADER_URL = pathToFileURL(
  path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'loaders', 'GLTFLoader.js'),
).href;
const DRACO_LOADER_URL = pathToFileURL(
  path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'loaders', 'DRACOLoader.js'),
).href;
const MESHOPT_URL = pathToFileURL(
  path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'three', 'examples', 'jsm', 'libs', 'meshopt_decoder.module.js'),
).href;

if (!globalThis.self) globalThis.self = globalThis;

export const SURFACE_SCHEMA = 'gcof-pvs-v5-local-surface-v1';
export const SURFACE_BINARY_MAGIC = 'GPV5';
export const SURFACE_BINARY_VERSION = 1;
export const SURFACE_BINARY_HEADER_BYTES = 16;
export const POINTS_PER_UNIT = 256;
export const POINT_FEATURE_DIM = 6;
export const POINT_RECORD_BYTES = POINTS_PER_UNIT * POINT_FEATURE_DIM * Float32Array.BYTES_PER_ELEMENT;
export const DEFAULT_SAMPLING_SEED = 20260918;

const CANONICAL_FILES = Object.freeze({
  points: 'surface_points_fp32.bin',
  sizeRatios: 'size_ratios_fp32.bin',
  aabbMin: 'aabb_min_fp32.npy',
  aabbMax: 'aabb_max_fp32.npy',
  unitIds: 'unit_ids_uint64.npy',
});

const AREA_EPSILON = 1e-14;
const AABB_EPSILON = 1e-7;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function asFiniteNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${name} must be finite`);
  return number;
}

function asNonNegativeInteger(value, name) {
  if (typeof value === 'boolean') throw new Error(`${name} must be a non-negative integer`);
  const number = Number(value);
  if (!Number.isInteger(number) || number < 0) throw new Error(`${name} must be a non-negative integer`);
  return number;
}

function asPositiveInteger(value, name) {
  const number = asNonNegativeInteger(value, name);
  if (number <= 0) throw new Error(`${name} must be positive`);
  return number;
}

function asVec3(value, name) {
  assert(Array.isArray(value) || ArrayBuffer.isView(value), `${name} must be a length-three array`);
  assert(value.length === 3, `${name} must be a length-three array`);
  const result = [
    asFiniteNumber(value[0], `${name}[0]`),
    asFiniteNumber(value[1], `${name}[1]`),
    asFiniteNumber(value[2], `${name}[2]`),
  ];
  return result;
}

function sortedUniqueIntegers(values, name) {
  assert(Array.isArray(values), `${name} must be an array`);
  const result = values.map((value, index) => asNonNegativeInteger(value, `${name}[${index}]`));
  const unique = [...new Set(result)].sort((a, b) => a - b);
  assert(unique.length === result.length, `${name} must not contain duplicates`);
  return unique;
}

function sameIntegerList(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function boundsFromRecord(record, name) {
  const raw = record?.bounds || record?.aabb || record;
  assert(raw && typeof raw === 'object', `${name} must contain bounds`);
  let min;
  let max;
  if (raw.min !== undefined || raw.max !== undefined) {
    assert(raw.min !== undefined && raw.max !== undefined, `${name} must contain both min and max`);
    min = asVec3(raw.min, `${name}.min`);
    max = asVec3(raw.max, `${name}.max`);
  } else {
    const center = asVec3(raw.center, `${name}.center`);
    const size = asVec3(raw.size, `${name}.size`);
    assert(size.every((value) => value >= 0), `${name}.size must be non-negative`);
    min = size.map((value, index) => center[index] - value * 0.5);
    max = size.map((value, index) => center[index] + value * 0.5);
  }
  assert(min.every((value, index) => value <= max[index] + AABB_EPSILON), `${name} has inverted bounds`);
  const size = min.map((value, index) => Math.max(0, max[index] - value));
  const center = min.map((value, index) => (value + max[index]) * 0.5);
  const halfDiagonal = Math.hypot(size[0], size[1], size[2]) * 0.5;
  return { min, max, center, size, halfDiagonal };
}

function normalizeComponentRecord(record, index) {
  assert(record && typeof record === 'object', `componentRecords[${index}] must be an object`);
  const componentGlobalId = asNonNegativeInteger(
    record.componentGlobalId,
    `componentRecords[${index}].componentGlobalId`,
  );
  const globalGlbId = asNonNegativeInteger(
    record.globalGlbId,
    `componentRecords[${index}].globalGlbId`,
  );
  return {
    ...record,
    componentGlobalId,
    globalGlbId,
    bounds: boundsFromRecord(record, `componentRecords[${index}].bounds`),
  };
}

function normalizeTriangle(triangle, index = 0) {
  const a = asVec3(triangle.a, `triangle[${index}].a`);
  const b = asVec3(triangle.b, `triangle[${index}].b`);
  const c = asVec3(triangle.c, `triangle[${index}].c`);
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const cross = [
    ab[1] * ac[2] - ab[2] * ac[1],
    ab[2] * ac[0] - ab[0] * ac[2],
    ab[0] * ac[1] - ab[1] * ac[0],
  ];
  const crossLength = Math.hypot(cross[0], cross[1], cross[2]);
  const area = Number.isFinite(triangle.area) ? Number(triangle.area) : crossLength * 0.5;
  if (!Number.isFinite(area) || area <= AREA_EPSILON || crossLength <= AREA_EPSILON) {
    throw new Error(`triangle[${index}] must have positive finite area`);
  }
  const normal = Array.isArray(triangle.normal) || ArrayBuffer.isView(triangle.normal)
    ? asVec3(triangle.normal, `triangle[${index}].normal`)
    : cross.map((value) => value / crossLength);
  const normalLength = Math.hypot(normal[0], normal[1], normal[2]);
  assert(normalLength > AREA_EPSILON, `triangle[${index}] normal must be non-zero`);
  const unitNormal = normal.map((value) => value / normalLength);
  const min = [
    Math.min(a[0], b[0], c[0]),
    Math.min(a[1], b[1], c[1]),
    Math.min(a[2], b[2], c[2]),
  ];
  const max = [
    Math.max(a[0], b[0], c[0]),
    Math.max(a[1], b[1], c[1]),
    Math.max(a[2], b[2], c[2]),
  ];
  const centroid = [
    (a[0] + b[0] + c[0]) / 3,
    (a[1] + b[1] + c[1]) / 3,
    (a[2] + b[2] + c[2]) / 3,
  ];
  return {
    ...triangle,
    a,
    b,
    c,
    area,
    normal: unitNormal,
    min,
    max,
    centroid,
  };
}

function pointInsideAabb(point, bounds, tolerance) {
  return point.every((value, axis) => (
    value >= bounds.min[axis] - tolerance && value <= bounds.max[axis] + tolerance
  ));
}

function pointAabbDistance(point, bounds) {
  let squared = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const distance = point[axis] < bounds.min[axis]
      ? bounds.min[axis] - point[axis]
      : point[axis] > bounds.max[axis]
        ? point[axis] - bounds.max[axis]
        : 0;
    squared += distance * distance;
  }
  return Math.sqrt(squared);
}

function aabbIntersects(min, max, bounds, tolerance) {
  return [0, 1, 2].every((axis) => (
    max[axis] >= bounds.min[axis] - tolerance && min[axis] <= bounds.max[axis] + tolerance
  ));
}

function triangleAabbOverlapFractions(triangle, bounds, tolerance) {
  const fractions = [];
  for (let axis = 0; axis < 3; axis += 1) {
    const triangleExtent = Math.max(0, triangle.max[axis] - triangle.min[axis]);
    if (triangleExtent <= tolerance) {
      fractions.push(
        triangle.min[axis] >= bounds.min[axis] - tolerance
        && triangle.min[axis] <= bounds.max[axis] + tolerance ? 1 : 0,
      );
      continue;
    }
    const overlap = Math.max(
      0,
      Math.min(triangle.max[axis], bounds.max[axis])
      - Math.max(triangle.min[axis], bounds.min[axis]),
    );
    fractions.push(Math.min(1, overlap / triangleExtent));
  }
  return fractions;
}

function candidateForTriangle(triangle, component, sceneScale) {
  const componentScale = Math.max(component.bounds.halfDiagonal * 2, sceneScale, 1);
  const tolerance = AABB_EPSILON * componentScale;
  const centroidInside = pointInsideAabb(triangle.centroid, component.bounds, tolerance);
  const intersects = aabbIntersects(triangle.min, triangle.max, component.bounds, tolerance);
  if (!centroidInside && !intersects) return null;
  const fractions = triangleAabbOverlapFractions(triangle, component.bounds, tolerance);
  const overlapMin = Math.min(...fractions);
  const overlapMean = (fractions[0] + fractions[1] + fractions[2]) / 3;
  const triangleContained = [0, 1, 2].every((axis) => (
    triangle.min[axis] >= component.bounds.min[axis] - tolerance
    && triangle.max[axis] <= component.bounds.max[axis] + tolerance
  ));
  const normalizedDistance = pointAabbDistance(triangle.centroid, component.bounds)
    / Math.max(component.bounds.halfDiagonal, sceneScale, 1);
  // Centroid containment is the strongest signal.  The overlap terms resolve
  // adjacent AABBs; the distance term only breaks otherwise close matches.
  const score = (centroidInside ? 1_000_000 : 0)
    + (triangleContained ? 100_000 : 0)
    + overlapMin * 1_000
    + overlapMean * 100
    - normalizedDistance;
  return {
    component,
    score,
    centroidInside,
    triangleContained,
    overlapMin,
    overlapMean,
    normalizedDistance,
  };
}

function nearestComponentFallback(triangle, components, sceneScale) {
  const ranked = components.map((component) => ({
    component,
    distance: pointAabbDistance(triangle.centroid, component.bounds),
  })).sort((left, right) => (
    left.distance - right.distance
    || left.component.componentGlobalId - right.component.componentGlobalId
  ));
  if (!ranked.length) return null;
  const nearest = ranked[0];
  const second = ranked[1];
  const componentScale = Math.max(nearest.component.bounds.halfDiagonal, sceneScale, 1);
  const triangleScale = Math.max(
    Math.hypot(
      triangle.max[0] - triangle.min[0],
      triangle.max[1] - triangle.min[1],
      triangle.max[2] - triangle.min[2],
    ),
    sceneScale,
    1,
  );
  const threshold = Math.max(AABB_EPSILON * sceneScale * 10, componentScale * 0.05, triangleScale * 0.05);
  const clearlyNearest = !second || second.distance - nearest.distance > threshold * 0.25;
  if (nearest.distance <= threshold && clearlyNearest) {
    return nearest.component;
  }
  return null;
}

/**
 * Assign world-space triangles to component records without ever duplicating a
 * triangle into two components.  The returned Maps are intentionally kept out
 * of JSON and are useful for focused unit tests.
 */
export function matchTrianglesToComponents(triangles, componentRecords, options = {}) {
  assert(Array.isArray(triangles), 'triangles must be an array');
  assert(Array.isArray(componentRecords) && componentRecords.length > 0, 'componentRecords must be non-empty');
  const components = componentRecords
    .map((record, index) => normalizeComponentRecord(record, index))
    .sort((left, right) => left.componentGlobalId - right.componentGlobalId);
  const sceneScale = Math.max(
    Number(options.sceneScale) > 0 ? Number(options.sceneScale) : 0,
    ...components.map((component) => component.bounds.halfDiagonal),
    1,
  );
  const normalizedTriangles = triangles.map((triangle, index) => normalizeTriangle(triangle, index));
  const assignments = new Map(components.map((component) => [component.componentGlobalId, []]));
  const componentStats = new Map(components.map((component) => [component.componentGlobalId, {
    assignedTriangleCount: 0,
    assignedArea: 0,
    modes: new Set(),
  }]));
  const stats = {
    triangleCount: normalizedTriangles.length,
    matchedTriangleCount: 0,
    fallbackTriangleCount: 0,
    unmatchedTriangleCount: 0,
    ambiguousTriangleCount: 0,
    modeCounts: {},
  };

  const note = (mode) => {
    stats.modeCounts[mode] = (stats.modeCounts[mode] || 0) + 1;
  };

  for (const triangle of normalizedTriangles) {
    const candidates = components
      .map((component) => candidateForTriangle(triangle, component, sceneScale))
      .filter(Boolean)
      .sort((left, right) => (
        right.score - left.score
        || left.component.componentGlobalId - right.component.componentGlobalId
      ));
    const hinted = triangle.instanceComponentGlobalId == null
      ? null
      : components.find((component) => component.componentGlobalId === Number(triangle.instanceComponentGlobalId));
    let chosen = null;
    let mode = null;

    // InstancedMesh carries an explicit instance-to-component ordering.  It is
    // used only to disambiguate an overlapping AABB, or to repair a tiny bounds
    // mismatch; ordinary multi-mesh GLBs still rely on geometric matching.
    if (hinted && (candidates.length === 0 || candidates[0].component.componentGlobalId !== hinted.componentGlobalId)) {
      const hintedCandidate = candidates.find(
        (candidate) => candidate.component.componentGlobalId === hinted.componentGlobalId,
      );
      if (hintedCandidate && candidates[0] && Math.abs(hintedCandidate.score - candidates[0].score) <= 1e-6) {
        chosen = hinted;
        mode = 'instance-index-geometric';
      } else if (candidates.length === 0) {
        chosen = hinted;
        mode = 'instance-index-fallback';
      }
    }
    if (!chosen && candidates.length > 0) {
      const best = candidates[0];
      const tied = candidates.filter((candidate) => Math.abs(candidate.score - best.score) <= 1e-6);
      if (tied.length === 1) {
        chosen = best.component;
        mode = 'geometric-aabb-centroid';
      } else if (hinted && tied.some((candidate) => candidate.component.componentGlobalId === hinted.componentGlobalId)) {
        chosen = hinted;
        mode = 'instance-index-geometric';
      } else {
        stats.ambiguousTriangleCount += 1;
        note('ambiguous');
      }
    }
    if (!chosen && !mode) {
      if (components.length === 1) {
        chosen = components[0];
        mode = 'single-component-fallback';
      } else {
        chosen = nearestComponentFallback(triangle, components, sceneScale);
        mode = chosen ? 'nearest-aabb-fallback' : null;
      }
    }

    if (!chosen) {
      stats.unmatchedTriangleCount += 1;
      note('unmatched');
      continue;
    }
    assignments.get(chosen.componentGlobalId).push(triangle);
    const perComponent = componentStats.get(chosen.componentGlobalId);
    perComponent.assignedTriangleCount += 1;
    perComponent.assignedArea += triangle.area;
    perComponent.modes.add(mode);
    stats.matchedTriangleCount += 1;
    if (mode.includes('fallback')) stats.fallbackTriangleCount += 1;
    note(mode);
  }

  return {
    components,
    assignments,
    componentStats,
    stats,
  };
}

export function createRng(seed) {
  const initial = asNonNegativeInteger(seed, 'sampling seed');
  let state = initial >>> 0;
  return function rng() {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export function deriveUnitSeed(seed, componentGlobalId) {
  let state = (asNonNegativeInteger(seed, 'sampling seed') ^ Math.imul(
    (asNonNegativeInteger(componentGlobalId, 'componentGlobalId') + 1) >>> 0,
    0x9e3779b1,
  )) >>> 0;
  state ^= state >>> 16;
  state = Math.imul(state, 0x85ebca6b) >>> 0;
  state ^= state >>> 13;
  state = Math.imul(state, 0xc2b2ae35) >>> 0;
  state ^= state >>> 16;
  return state >>> 0;
}

function zeroFeatureBlock() {
  return new Float32Array(POINTS_PER_UNIT * POINT_FEATURE_DIM);
}

/** Sample one component's assigned triangles with world-area weighting. */
export function sampleUnitSurfaceFeatures(
  triangles,
  componentBounds,
  { seed = DEFAULT_SAMPLING_SEED, componentGlobalId = 0, pointsPerUnit = POINTS_PER_UNIT } = {},
) {
  assert(pointsPerUnit === POINTS_PER_UNIT, 'V5 surface sampling is fixed at 256 points per unit');
  const bounds = boundsFromRecord(componentBounds, 'component bounds');
  const normalizedTriangles = (triangles || []).map((triangle, index) => normalizeTriangle(triangle, index));
  const features = zeroFeatureBlock();
  if (!normalizedTriangles.length) {
    return {
      features,
      emptyGeometry: true,
      sampledArea: 0,
      normalization: bounds,
      outOfBoundsPointCount: 0,
    };
  }
  assert(bounds.halfDiagonal > AREA_EPSILON, 'component AABB half diagonal must be positive for surface sampling');
  const cumulative = [];
  let totalArea = 0;
  for (const triangle of normalizedTriangles) {
    totalArea += triangle.area;
    cumulative.push(totalArea);
  }
  assert(Number.isFinite(totalArea) && totalArea > AREA_EPSILON, 'surface triangle area must be positive and finite');
  const rng = createRng(deriveUnitSeed(seed, componentGlobalId));
  let outOfBoundsPointCount = 0;
  for (let pointIndex = 0; pointIndex < POINTS_PER_UNIT; pointIndex += 1) {
    const target = rng() * totalArea;
    let low = 0;
    let high = cumulative.length - 1;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (target < cumulative[middle]) high = middle;
      else low = middle + 1;
    }
    const triangle = normalizedTriangles[low];
    const sqrtU = Math.sqrt(rng());
    const v = rng();
    const wa = 1 - sqrtU;
    const wb = sqrtU * (1 - v);
    const wc = sqrtU * v;
    const worldPoint = [
      triangle.a[0] * wa + triangle.b[0] * wb + triangle.c[0] * wc,
      triangle.a[1] * wa + triangle.b[1] * wb + triangle.c[1] * wc,
      triangle.a[2] * wa + triangle.b[2] * wb + triangle.c[2] * wc,
    ];
    const normalized = [
      (worldPoint[0] - bounds.center[0]) / bounds.halfDiagonal,
      (worldPoint[1] - bounds.center[1]) / bounds.halfDiagonal,
      (worldPoint[2] - bounds.center[2]) / bounds.halfDiagonal,
    ];
    if (normalized.some((value) => Math.abs(value) > 1 + 1e-5)) outOfBoundsPointCount += 1;
    const offset = pointIndex * POINT_FEATURE_DIM;
    features[offset] = normalized[0];
    features[offset + 1] = normalized[1];
    features[offset + 2] = normalized[2];
    features[offset + 3] = triangle.normal[0];
    features[offset + 4] = triangle.normal[1];
    features[offset + 5] = triangle.normal[2];
  }
  assert(Number.isFinite(features[features.length - 1]), 'surface feature block contains non-finite values');
  return {
    features,
    emptyGeometry: false,
    sampledArea: totalArea,
    normalization: bounds,
    outOfBoundsPointCount,
  };
}

function vectorFromAttribute(THREE, attribute, index) {
  return [attribute.getX(index), attribute.getY(index), attribute.getZ(index)];
}

function transformPoint(THREE, point, matrix) {
  const vector = new THREE.Vector3(point[0], point[1], point[2]).applyMatrix4(matrix);
  return [vector.x, vector.y, vector.z];
}

function collectObjectTriangles(THREE, object, instanceIndex, sourceIndexStart, instanceComponentGlobalId) {
  const geometry = object.geometry;
  if (!geometry || !geometry.attributes || !geometry.attributes.position) {
    return { triangles: [], sourceTriangleCount: 0, degenerateTriangleCount: 0 };
  }
  const position = geometry.attributes.position;
  const index = geometry.index;
  const indexCount = index ? index.count : position.count;
  const sourceTriangleCount = Math.floor(indexCount / 3);
  const instanceMatrix = new THREE.Matrix4().identity();
  if (instanceIndex !== null) object.getMatrixAt(instanceIndex, instanceMatrix);
  const finalMatrix = new THREE.Matrix4().multiplyMatrices(
    object.matrixWorld,
    instanceIndex === null ? new THREE.Matrix4().identity() : instanceMatrix,
  );
  const triangles = [];
  let degenerateTriangleCount = 0;
  for (let triangleIndex = 0; triangleIndex < sourceTriangleCount; triangleIndex += 1) {
    const ia = index ? index.getX(triangleIndex * 3) : triangleIndex * 3;
    const ib = index ? index.getX(triangleIndex * 3 + 1) : triangleIndex * 3 + 1;
    const ic = index ? index.getX(triangleIndex * 3 + 2) : triangleIndex * 3 + 2;
    const a = transformPoint(THREE, vectorFromAttribute(THREE, position, ia), finalMatrix);
    const b = transformPoint(THREE, vectorFromAttribute(THREE, position, ib), finalMatrix);
    const c = transformPoint(THREE, vectorFromAttribute(THREE, position, ic), finalMatrix);
    try {
      triangles.push(normalizeTriangle({
        a,
        b,
        c,
        sourceIndex: sourceIndexStart + triangleIndex,
        instanceComponentGlobalId,
      }, sourceIndexStart + triangleIndex));
    } catch (error) {
      if (String(error?.message || error).includes('positive finite area')) {
        degenerateTriangleCount += 1;
        continue;
      }
      throw error;
    }
  }
  return { triangles, sourceTriangleCount, degenerateTriangleCount };
}

/** Collect transformed face triangles from every renderable object in a GLB. */
export function collectWorldTriangles(THREE, gltf, componentGlobalIds = []) {
  assert(gltf?.scene, 'GLB has no scene');
  const triangles = [];
  let sourceTriangleCount = 0;
  let degenerateTriangleCount = 0;
  let renderableObjectCount = 0;
  let sourceIndex = 0;
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((object) => {
    if (!object.isMesh && !object.isInstancedMesh) return;
    renderableObjectCount += 1;
    if (object.isInstancedMesh) {
      for (let instanceIndex = 0; instanceIndex < object.count; instanceIndex += 1) {
        const hintedComponent = componentGlobalIds.length === object.count
          ? componentGlobalIds[instanceIndex]
          : null;
        const result = collectObjectTriangles(
          THREE,
          object,
          instanceIndex,
          sourceIndex,
          hintedComponent,
        );
        for (const triangle of result.triangles) triangles.push(triangle);
        sourceTriangleCount += result.sourceTriangleCount;
        degenerateTriangleCount += result.degenerateTriangleCount;
        sourceIndex += result.sourceTriangleCount;
      }
    } else {
      const result = collectObjectTriangles(THREE, object, null, sourceIndex, null);
      for (const triangle of result.triangles) triangles.push(triangle);
      sourceTriangleCount += result.sourceTriangleCount;
      degenerateTriangleCount += result.degenerateTriangleCount;
      sourceIndex += result.sourceTriangleCount;
    }
  });
  return {
    triangles,
    sourceTriangleCount,
    degenerateTriangleCount,
    renderableObjectCount,
  };
}

function disposeLoadedGltf(gltf) {
  gltf?.scene?.traverse((object) => {
    if (object.geometry?.dispose) object.geometry.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!material) continue;
      for (const value of Object.values(material)) {
        if (!value?.isTexture) continue;
        const source = value.source?.data?.src || value.image?.src;
        if (typeof source === 'string' && source.startsWith('blob:')) URL.revokeObjectURL(source);
        value.dispose?.();
      }
      material.dispose?.();
    }
  });
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

/** Validate and normalize the two runtime mapping inputs. */
export function validateRuntimeInputs(runtimeMeta, glbIndex) {
  assert(runtimeMeta && typeof runtimeMeta === 'object', 'runtimeVisibilityMeta must be an object');
  assert(Array.isArray(runtimeMeta.componentRecords), 'runtimeVisibilityMeta.componentRecords must be an array');
  assert(Array.isArray(runtimeMeta.globalGlbRecords), 'runtimeVisibilityMeta.globalGlbRecords must be an array');
  assert(glbIndex && typeof glbIndex === 'object', 'glbIndex must be an object');
  assert(Array.isArray(glbIndex.entries), 'glbIndex.entries must be an array');
  assert(glbIndex.entries.length > 0, 'glbIndex.entries must not be empty');

  const components = runtimeMeta.componentRecords
    .map((record, index) => normalizeComponentRecord(record, index))
    .sort((left, right) => left.componentGlobalId - right.componentGlobalId);
  assert(components.length > 0, 'runtimeVisibilityMeta must contain at least one component');
  for (let index = 1; index < components.length; index += 1) {
    assert(
      components[index - 1].componentGlobalId !== components[index].componentGlobalId,
      'componentRecords contain duplicate componentGlobalId values',
    );
  }
  const componentById = new Map(components.map((component) => [component.componentGlobalId, component]));
  const componentIdsByGlb = new Map();
  for (const component of components) {
    const ids = componentIdsByGlb.get(component.globalGlbId) || [];
    ids.push(component.componentGlobalId);
    componentIdsByGlb.set(component.globalGlbId, ids);
  }
  for (const ids of componentIdsByGlb.values()) ids.sort((left, right) => left - right);

  const globalRecordsById = new Map();
  for (let index = 0; index < runtimeMeta.globalGlbRecords.length; index += 1) {
    const record = runtimeMeta.globalGlbRecords[index];
    assert(record && typeof record === 'object', `globalGlbRecords[${index}] must be an object`);
    const globalGlbId = asNonNegativeInteger(record.globalGlbId, `globalGlbRecords[${index}].globalGlbId`);
    assert(!globalRecordsById.has(globalGlbId), `duplicate globalGlbId ${globalGlbId}`);
    const componentGlobalIds = record.componentGlobalIds === undefined
      ? undefined
      : sortedUniqueIntegers(record.componentGlobalIds, `globalGlbRecords[${index}].componentGlobalIds`);
    globalRecordsById.set(globalGlbId, { ...record, globalGlbId, componentGlobalIds });
  }

  const entries = glbIndex.entries.map((entry, index) => {
    assert(entry && typeof entry === 'object', `glbIndex.entries[${index}] must be an object`);
    const globalId = asNonNegativeInteger(entry.globalId, `glbIndex.entries[${index}].globalId`);
    assert(typeof entry.path === 'string' && entry.path.length > 0, `glbIndex.entries[${index}].path must be non-empty`);
    const componentGlobalIds = entry.componentGlobalIds === undefined
      ? undefined
      : sortedUniqueIntegers(entry.componentGlobalIds, `glbIndex.entries[${index}].componentGlobalIds`);
    return { ...entry, globalId, componentGlobalIds };
  }).sort((left, right) => left.globalId - right.globalId);
  for (let index = 1; index < entries.length; index += 1) {
    assert(entries[index - 1].globalId !== entries[index].globalId, 'glbIndex contains duplicate globalId values');
  }
  if (glbIndex.total !== undefined) {
    assert(Number(glbIndex.total) === entries.length, 'glbIndex.total does not match entries length');
  }

  const entryByGlb = new Map(entries.map((entry) => [entry.globalId, entry]));
  for (const [globalGlbId, record] of globalRecordsById) {
    assert(entryByGlb.has(globalGlbId), `globalGlbId ${globalGlbId} is missing from glbIndex`);
    const derived = componentIdsByGlb.get(globalGlbId) || [];
    if (record.componentGlobalIds !== undefined) {
      assert(
        sameIntegerList(record.componentGlobalIds, derived),
        `runtime globalGlbRecords mapping disagrees for globalGlbId ${globalGlbId}`,
      );
    }
  }
  for (const entry of entries) {
    const derived = componentIdsByGlb.get(entry.globalId) || [];
    if (entry.componentGlobalIds !== undefined) {
      assert(
        sameIntegerList(entry.componentGlobalIds, derived),
        `glbIndex component mapping disagrees for globalGlbId ${entry.globalId}`,
      );
    }
  }
  for (const component of components) {
    assert(entryByGlb.has(component.globalGlbId), `component ${component.componentGlobalId} has no GLB index entry`);
    assert(componentById.has(component.componentGlobalId), 'internal component mapping error');
  }
  if (runtimeMeta.componentCount !== undefined) {
    assert(Number(runtimeMeta.componentCount) >= components.length, 'runtime componentCount is smaller than componentRecords');
  }
  if (runtimeMeta.globalGlbCount !== undefined) {
    assert(Number(runtimeMeta.globalGlbCount) >= entries.length, 'runtime globalGlbCount is smaller than glbIndex entries');
  }

  return {
    components,
    componentById,
    componentIdsByGlb,
    globalRecordsById,
    entries,
    entryByGlb,
  };
}

async function importThreeDependencies() {
  const THREE = await import(THREE_URL);
  const { GLTFLoader } = await import(GLTF_LOADER_URL);
  const { DRACOLoader } = await import(DRACO_LOADER_URL);
  const { MeshoptDecoder } = await import(MESHOPT_URL);
  return { THREE, GLTFLoader, DRACOLoader, MeshoptDecoder };
}

export function createGltfLoader({ GLTFLoader, DRACOLoader, MeshoptDecoder }) {
  const dracoLoader = new DRACOLoader().setDecoderPath(
    `${pathToFileURL(path.join(
      REPO_ROOT,
      'slm2viewer',
      'node_modules',
      'three',
      'examples',
      'jsm',
      'libs',
      'draco',
      'gltf',
    )).href}/`,
  );
  const loader = new GLTFLoader().setDRACOLoader(dracoLoader).setMeshoptDecoder(MeshoptDecoder);
  return { loader, dracoLoader };
}

export async function loadGltf(loader, filePath) {
  const data = fs.readFileSync(filePath);
  const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return new Promise((resolve, reject) => {
    loader.parse(arrayBuffer, `${path.dirname(filePath).replace(/\\/g, '/')}/`, resolve, reject);
  });
}

function createDenseFp32Header(featureDim, numUnits, itemsPerUnit) {
  asPositiveInteger(featureDim, 'featureDim');
  asPositiveInteger(numUnits, 'numUnits');
  asPositiveInteger(itemsPerUnit, 'itemsPerUnit');
  assert(featureDim <= 0xffff, 'featureDim exceeds the canonical uint16 header field');
  const header = Buffer.alloc(SURFACE_BINARY_HEADER_BYTES);
  header.write(SURFACE_BINARY_MAGIC, 0, 'ascii');
  header.writeUInt16LE(SURFACE_BINARY_VERSION, 4);
  header.writeUInt16LE(featureDim, 6);
  header.writeUInt32LE(numUnits, 8);
  header.writeUInt32LE(itemsPerUnit, 12);
  return header;
}

export function parseDenseFp32Header(value) {
  const bytes = Buffer.isBuffer(value)
    ? value
    : value instanceof ArrayBuffer
      ? Buffer.from(value)
      : Buffer.from(value.buffer, value.byteOffset || 0, value.byteLength);
  assert(bytes.length >= SURFACE_BINARY_HEADER_BYTES, 'dense FP32 asset is missing its 16-byte header');
  const header = {
    magic: bytes.subarray(0, 4).toString('ascii'),
    version: bytes.readUInt16LE(4),
    featureDim: bytes.readUInt16LE(6),
    numUnits: bytes.readUInt32LE(8),
    itemsPerUnit: bytes.readUInt32LE(12),
  };
  assert(header.magic === SURFACE_BINARY_MAGIC, 'dense FP32 asset magic must be GPV5');
  assert(header.version === SURFACE_BINARY_VERSION, 'unsupported dense FP32 asset version');
  return header;
}

function float32BufferLittleEndian(values) {
  if (os.endianness() === 'LE') {
    return Buffer.from(values.buffer, values.byteOffset, values.byteLength);
  }
  const buffer = Buffer.alloc(values.length * Float32Array.BYTES_PER_ELEMENT);
  for (let index = 0; index < values.length; index += 1) buffer.writeFloatLE(values[index], index * 4);
  return buffer;
}

function sizeRatios(bounds) {
  const maximum = Math.max(...bounds.size);
  return maximum > AREA_EPSILON ? bounds.size.map((value) => value / maximum) : [0, 0, 0];
}

function writeNpyV1(filePath, values, { descr, shape }) {
  const shapeText = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')},)`;
  const dictionary = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeText}, }`;
  const preambleBytes = 10;
  const padding = (16 - ((preambleBytes + dictionary.length + 1) % 16)) % 16;
  const headerText = `${dictionary}${' '.repeat(padding)}\n`;
  const header = Buffer.alloc(preambleBytes + headerText.length);
  header[0] = 0x93;
  header.write('NUMPY', 1, 'ascii');
  header[6] = 1;
  header[7] = 0;
  header.writeUInt16LE(headerText.length, 8);
  header.write(headerText, preambleBytes, 'ascii');
  const payload = descr === '<f4'
    ? float32BufferLittleEndian(values)
    : (() => {
      assert(descr === '<u8', `unsupported NumPy dtype ${descr}`);
      const bytes = Buffer.alloc(values.length * 8);
      for (let index = 0; index < values.length; index += 1) {
        bytes.writeBigUInt64LE(BigInt(asNonNegativeInteger(values[index], `unitIds[${index}]`)), index * 8);
      }
      return bytes;
    })();
  fs.writeFileSync(filePath, Buffer.concat([header, payload]));
}

function writeDenseFp32(filePath, rows, featureDim, itemsPerUnit) {
  const fd = fs.openSync(filePath, 'w');
  try {
    fs.writeSync(fd, createDenseFp32Header(featureDim, rows.length, itemsPerUnit));
    for (const row of rows) {
      assert(row.length === featureDim * itemsPerUnit, `${path.basename(filePath)} row has the wrong shape`);
      fs.writeSync(fd, float32BufferLittleEndian(row));
    }
  } finally {
    fs.closeSync(fd);
  }
}

function makeManifest(components, degenerateUnitIds, seed) {
  return {
    schema: SURFACE_SCHEMA,
    version: 1,
    pointsPerUnit: POINTS_PER_UNIT,
    pointFeatureDim: POINT_FEATURE_DIM,
    pointFeatureOrder: 'normalized_xyz_unit_normal',
    sizeRatioDim: 3,
    storage: 'dense_fp32_header16',
    numUnits: components.length,
    unitIds: components.map((component) => component.componentGlobalId),
    samplingSeed: asNonNegativeInteger(seed, 'sampling seed'),
    transformPolicy: 'world_triangle_area_inverse_transpose_normal',
    degenerateUnitIds,
    files: { ...CANONICAL_FILES },
  };
}

export function validateSurfaceManifest(manifest) {
  const required = [
    'schema',
    'version',
    'pointsPerUnit',
    'pointFeatureDim',
    'pointFeatureOrder',
    'sizeRatioDim',
    'storage',
    'numUnits',
    'unitIds',
    'samplingSeed',
    'transformPolicy',
    'degenerateUnitIds',
    'files',
  ];
  assert(manifest && typeof manifest === 'object', 'surface manifest must be an object');
  assert(Object.keys(manifest).sort().join(',') === required.sort().join(','), 'surface manifest keys do not match the canonical schema');
  assert(manifest.schema === SURFACE_SCHEMA && manifest.version === 1, 'unsupported surface manifest schema');
  assert(manifest.pointsPerUnit === POINTS_PER_UNIT, 'surface manifest pointsPerUnit must be 256');
  assert(manifest.pointFeatureDim === 6 && manifest.pointFeatureOrder === 'normalized_xyz_unit_normal', 'surface manifest point feature contract is invalid');
  assert(manifest.sizeRatioDim === 3, 'surface manifest sizeRatioDim must be 3');
  assert(manifest.storage === 'dense_fp32_header16', 'surface manifest storage is not canonical');
  assert(Number.isInteger(manifest.numUnits) && manifest.numUnits > 0, 'surface manifest numUnits must be positive');
  assert(Array.isArray(manifest.unitIds) && manifest.unitIds.length === manifest.numUnits, 'surface manifest unitIds shape is invalid');
  assert(manifest.unitIds.every((value, index) => Number.isInteger(value) && value >= 0
    && (index === 0 || value > manifest.unitIds[index - 1])), 'surface manifest unitIds must be sorted and unique');
  assert(Number.isInteger(manifest.samplingSeed) && manifest.samplingSeed >= 0, 'surface manifest samplingSeed is invalid');
  assert(manifest.transformPolicy === 'world_triangle_area_inverse_transpose_normal', 'surface manifest transformPolicy is invalid');
  assert(Array.isArray(manifest.degenerateUnitIds)
    && manifest.degenerateUnitIds.every((value) => manifest.unitIds.includes(value)), 'surface manifest degenerateUnitIds are invalid');
  assert(Object.keys(manifest.files).sort().join(',') === Object.keys(CANONICAL_FILES).sort().join(','), 'surface manifest files do not match the canonical schema');
  for (const [key, value] of Object.entries(CANONICAL_FILES)) {
    assert(manifest.files[key] === value, `surface manifest file ${key} is not canonical`);
  }
  return manifest;
}

function createFailureState(component, entry, reason, sourceTriangleCount = 0) {
  return {
    component,
    glbPath: entry.path,
    features: zeroFeatureBlock(),
    fallback: true,
    fallbackReason: reason,
    emptyGeometry: false,
    analysis: {
      sourceTriangleCount,
      usableTriangleCount: 0,
      assignedTriangleCount: 0,
      assignedArea: 0,
      outOfBoundsPointCount: 0,
      matchMode: 'unmatched',
      unmatchedTriangleCount: sourceTriangleCount,
      ambiguousTriangleCount: 0,
    },
  };
}

function createEmptyState(component, entry, geometryStats) {
  return {
    component,
    glbPath: entry.path,
    features: zeroFeatureBlock(),
    fallback: false,
    fallbackReason: null,
    emptyGeometry: true,
    analysis: {
      sourceTriangleCount: geometryStats.sourceTriangleCount,
      usableTriangleCount: 0,
      assignedTriangleCount: 0,
      assignedArea: 0,
      outOfBoundsPointCount: 0,
      matchMode: 'empty-geometry',
      unmatchedTriangleCount: 0,
      ambiguousTriangleCount: 0,
    },
  };
}

async function processGlbEntry({ THREE, loader, entry, componentIds, stateById, seed, totals }) {
  let gltf;
  try {
    gltf = await loadGltf(loader, entry.absPath);
  } catch (error) {
    totals.decodeFailureCount += 1;
    for (const componentId of componentIds) {
      stateById.set(componentId, createFailureState(
        stateById.get(componentId).component,
        entry,
        `glb-decode-error: ${String(error?.message || error)}`,
      ));
    }
    return;
  }

  const geometry = collectWorldTriangles(THREE, gltf, componentIds);
  disposeLoadedGltf(gltf);
  gltf = null;
  totals.glbCountProcessed += 1;
  totals.sourceTriangleCount += geometry.sourceTriangleCount;
  totals.usableTriangleCount += geometry.triangles.length;
  totals.degenerateTriangleCount += geometry.degenerateTriangleCount;
  totals.renderableObjectCount += geometry.renderableObjectCount;
  if (geometry.triangles.length === 0) {
    totals.emptyGeometryGlbCount += 1;
    for (const componentId of componentIds) {
      stateById.set(componentId, createEmptyState(
        stateById.get(componentId).component,
        entry,
        geometry,
      ));
    }
    return;
  }

  const componentRecords = componentIds.map((componentId) => stateById.get(componentId).component);
  const matched = matchTrianglesToComponents(geometry.triangles, componentRecords, {
    sceneScale: Math.max(...componentRecords.map((component) => component.bounds.halfDiagonal), 1),
  });
  totals.matchedTriangleCount += matched.stats.matchedTriangleCount;
  totals.fallbackTriangleCount += matched.stats.fallbackTriangleCount;
  totals.unmatchedTriangleCount += matched.stats.unmatchedTriangleCount;
  totals.ambiguousTriangleCount += matched.stats.ambiguousTriangleCount;
  for (const [mode, count] of Object.entries(matched.stats.modeCounts)) {
    totals.matchModeCounts[mode] = (totals.matchModeCounts[mode] || 0) + count;
  }

  for (const componentId of componentIds) {
    const state = stateById.get(componentId);
    const assigned = matched.assignments.get(componentId) || [];
    const componentStats = matched.componentStats.get(componentId);
    if (!assigned.length) {
      stateById.set(componentId, createFailureState(
        state.component,
        entry,
        matched.stats.ambiguousTriangleCount > 0
          ? 'component-aabb-match-ambiguous-or-missing'
          : 'component-aabb-no-geometric-match',
        geometry.sourceTriangleCount,
      ));
      stateById.get(componentId).analysis.usableTriangleCount = geometry.triangles.length;
      stateById.get(componentId).analysis.unmatchedTriangleCount = geometry.triangles.length;
      continue;
    }
    const modes = [...componentStats.modes].sort();
    if (modes.some((mode) => mode.includes('fallback'))) {
      stateById.set(componentId, createFailureState(
        state.component,
        entry,
        modes.join(','),
        geometry.sourceTriangleCount,
      ));
      stateById.get(componentId).analysis.usableTriangleCount = geometry.triangles.length;
      continue;
    }
    let sampled;
    try {
      sampled = sampleUnitSurfaceFeatures(assigned, state.component.bounds, {
        seed,
        componentGlobalId: componentId,
      });
    } catch (error) {
      stateById.set(componentId, createFailureState(
        state.component,
        entry,
        `surface-sampling-error: ${String(error?.message || error)}`,
        geometry.sourceTriangleCount,
      ));
      stateById.get(componentId).analysis.usableTriangleCount = geometry.triangles.length;
      continue;
    }
    const incomplete = matched.stats.unmatchedTriangleCount > 0;
    stateById.set(componentId, {
      component: state.component,
      glbPath: entry.path,
      features: sampled.features,
      fallback: false,
      fallbackReason: null,
      emptyGeometry: false,
      analysis: {
        sourceTriangleCount: geometry.sourceTriangleCount,
        usableTriangleCount: geometry.triangles.length,
        assignedTriangleCount: assigned.length,
        assignedArea: componentStats.assignedArea,
        outOfBoundsPointCount: sampled.outOfBoundsPointCount,
        matchMode: modes.join(','),
        unmatchedTriangleCount: incomplete ? matched.stats.unmatchedTriangleCount : 0,
        ambiguousTriangleCount: matched.stats.ambiguousTriangleCount,
      },
    });
  }
}

function normalizeOptions(options = {}) {
  const assetsDir = path.resolve(options.assetsDir || DEFAULT_ASSETS_DIR);
  const runtimeMetaPath = path.resolve(options.runtimeMetaPath || options.runtimeMetaFile || DEFAULT_RUNTIME_META);
  const glbIndexPath = path.resolve(options.glbIndexPath || options.glbIndexFile || DEFAULT_GLB_INDEX);
  const outputDir = path.resolve(options.outputDir || DEFAULT_OUTPUT_DIR);
  const seed = asNonNegativeInteger(
    options.seed === undefined ? DEFAULT_SAMPLING_SEED : options.seed,
    'sampling seed',
  );
  const maxGlbs = options.maxGlbs === undefined || options.maxGlbs === null
    ? null
    : asPositiveInteger(options.maxGlbs, 'maxGlbs');
  const progressEvery = options.progressEvery === undefined
    ? 50
    : asNonNegativeInteger(options.progressEvery, 'progressEvery');
  return {
    assetsDir,
    runtimeMetaPath,
    glbIndexPath,
    outputDir,
    seed,
    maxGlbs,
    progressEvery,
    runtimeMeta: options.runtimeMeta,
    glbIndex: options.glbIndex,
    dependencies: options.dependencies,
  };
}

/** Generate the canonical V5 local-surface asset directory. */
export async function generateV5InstanceSurfaceSamples(options = {}) {
  const args = normalizeOptions(options);
  const runtimeMeta = args.runtimeMeta || readJson(args.runtimeMetaPath);
  const glbIndex = args.glbIndex || readJson(args.glbIndexPath);
  const input = validateRuntimeInputs(runtimeMeta, glbIndex);
  const selectedEntries = args.maxGlbs === null ? input.entries : input.entries.slice(0, args.maxGlbs);
  const selectedGlbIds = new Set(selectedEntries.map((entry) => entry.globalId));
  const components = input.components.filter((component) => selectedGlbIds.has(component.globalGlbId));
  assert(components.length > 0, 'selected GLB entries contain no mapped components');
  const stateById = new Map();
  for (const component of components) {
    const entry = input.entryByGlb.get(component.globalGlbId);
    stateById.set(component.componentGlobalId, {
      component,
      glbPath: entry.path,
      features: null,
      fallback: true,
      fallbackReason: 'not-processed',
      emptyGeometry: false,
      analysis: null,
    });
  }

  fs.mkdirSync(args.outputDir, { recursive: true });
  const totals = {
    glbCountSelected: selectedEntries.length,
    glbCountProcessed: 0,
    decodeFailureCount: 0,
    emptyGeometryGlbCount: 0,
    renderableObjectCount: 0,
    sourceTriangleCount: 0,
    usableTriangleCount: 0,
    degenerateTriangleCount: 0,
    matchedTriangleCount: 0,
    fallbackTriangleCount: 0,
    unmatchedTriangleCount: 0,
    ambiguousTriangleCount: 0,
    matchModeCounts: {},
  };
  let loaderBundle = null;
  try {
    const dependencies = args.dependencies || await importThreeDependencies();
    loaderBundle = createGltfLoader(dependencies);
    for (let index = 0; index < selectedEntries.length; index += 1) {
      const entry = selectedEntries[index];
      const componentIds = (input.componentIdsByGlb.get(entry.globalId) || [])
        .filter((componentId) => stateById.has(componentId));
      if (componentIds.length > 0) {
        await processGlbEntry({
          THREE: dependencies.THREE,
          loader: loaderBundle.loader,
          entry: { ...entry, absPath: path.resolve(args.assetsDir, entry.path) },
          componentIds,
          stateById,
          seed: args.seed,
          totals,
        });
      }
      if (args.progressEvery > 0 && (index + 1) % args.progressEvery === 0) {
        console.log(`[v5-surface] ${index + 1}/${selectedEntries.length} GLBs processed`);
      }
    }

    const pointRows = [];
    const sizeRatioRows = [];
    const aabbMin = new Float32Array(components.length * 3);
    const aabbMax = new Float32Array(components.length * 3);
    const unitIds = components.map((component) => component.componentGlobalId);
    const degenerateUnitIds = [];
    for (let unitIndex = 0; unitIndex < components.length; unitIndex += 1) {
      const component = components[unitIndex];
      const state = stateById.get(component.componentGlobalId);
      if (!state.features) {
        state.features = zeroFeatureBlock();
        state.fallback = true;
        state.fallbackReason = 'not-processed';
      }
      assert(state.features.length === POINTS_PER_UNIT * POINT_FEATURE_DIM, `unit ${component.componentGlobalId} has invalid feature shape`);
      pointRows.push(state.features);
      sizeRatioRows.push(Float32Array.from(sizeRatios(component.bounds)));
      aabbMin.set(component.bounds.min, unitIndex * 3);
      aabbMax.set(component.bounds.max, unitIndex * 3);
      if (state.fallback || state.emptyGeometry) degenerateUnitIds.push(component.componentGlobalId);
    }

    const pointsPath = path.join(args.outputDir, CANONICAL_FILES.points);
    const sizeRatiosPath = path.join(args.outputDir, CANONICAL_FILES.sizeRatios);
    writeDenseFp32(pointsPath, pointRows, POINT_FEATURE_DIM, POINTS_PER_UNIT);
    writeDenseFp32(sizeRatiosPath, sizeRatioRows, 3, 1);
    writeNpyV1(path.join(args.outputDir, CANONICAL_FILES.aabbMin), aabbMin, {
      descr: '<f4',
      shape: [components.length, 3],
    });
    writeNpyV1(path.join(args.outputDir, CANONICAL_FILES.aabbMax), aabbMax, {
      descr: '<f4',
      shape: [components.length, 3],
    });
    writeNpyV1(path.join(args.outputDir, CANONICAL_FILES.unitIds), unitIds, {
      descr: '<u8',
      shape: [components.length],
    });

    const manifest = validateSurfaceManifest(makeManifest(components, degenerateUnitIds, args.seed));
    const manifestPath = path.join(args.outputDir, 'surface_manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), 'utf8');
    const expectedPointBytes = SURFACE_BINARY_HEADER_BYTES + components.length * POINT_RECORD_BYTES;
    const expectedSizeRatioBytes = SURFACE_BINARY_HEADER_BYTES + components.length * 3 * 4;
    assert(fs.statSync(pointsPath).size === expectedPointBytes, 'surface point binary size mismatch');
    assert(fs.statSync(sizeRatiosPath).size === expectedSizeRatioBytes, 'size-ratio binary size mismatch');
    const stats = {
      ...totals,
      unitCount: components.length,
      fallbackUnitCount: components.filter((component) => stateById.get(component.componentGlobalId).fallback).length,
      emptyGeometryUnitCount: components.filter((component) => stateById.get(component.componentGlobalId).emptyGeometry).length,
      degenerateUnitCount: degenerateUnitIds.length,
      outOfBoundsPointCount: components.reduce(
        (sum, component) => sum + Number(stateById.get(component.componentGlobalId).analysis?.outOfBoundsPointCount || 0),
        0,
      ),
      pointBytes: expectedPointBytes,
      sizeRatioBytes: expectedSizeRatioBytes,
    };
    return { outputDir: args.outputDir, manifestPath, manifest, stats };
  } finally {
    if (loaderBundle?.dracoLoader) loaderBundle.dracoLoader.dispose();
  }
}

export function parseArgs(argv = process.argv) {
  const args = {
    assetsDir: DEFAULT_ASSETS_DIR,
    runtimeMetaPath: DEFAULT_RUNTIME_META,
    glbIndexPath: DEFAULT_GLB_INDEX,
    outputDir: DEFAULT_OUTPUT_DIR,
    seed: DEFAULT_SAMPLING_SEED,
    maxGlbs: null,
    progressEvery: 50,
    help: false,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === '--help' || key === '-h') {
      args.help = true;
      continue;
    }
    assert(key.startsWith('--'), `unknown argument ${key}`);
    const value = argv[index + 1];
    assert(value !== undefined && !value.startsWith('--'), `${key} requires a value`);
    index += 1;
    if (key === '--assets-dir') args.assetsDir = path.resolve(value);
    else if (key === '--runtime-meta') args.runtimeMetaPath = path.resolve(value);
    else if (key === '--glb-index') args.glbIndexPath = path.resolve(value);
    else if (key === '--output-dir') args.outputDir = path.resolve(value);
    else if (key === '--seed') args.seed = asNonNegativeInteger(value, 'seed');
    else if (key === '--max-glbs') args.maxGlbs = asPositiveInteger(value, 'max-glbs');
    else if (key === '--progress-every') args.progressEvery = asNonNegativeInteger(value, 'progress-every');
    else throw new Error(`unknown argument ${key}`);
  }
  return args;
}

function printHelp() {
  console.log(`Usage: node generate_v5_instance_surface_samples.mjs [options]

Options:
  --assets-dir DIR       Scene asset root containing the indexed GLBs.
  --runtime-meta FILE    runtimeVisibilityMeta.json path.
  --glb-index FILE       glbIndex.json path.
  --output-dir DIR       Canonical V5 surface asset directory.
  --seed N               Deterministic non-negative sampling seed.
  --max-glbs N           Process only the first N indexed GLBs.
  --progress-every N     Print progress every N GLBs; zero disables it.
`);
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  try {
    const args = parseArgs(process.argv);
    if (args.help) {
      printHelp();
    } else {
      const result = await generateV5InstanceSurfaceSamples(args);
      console.log(JSON.stringify({
        schema: result.manifest.schema,
        outputDir: result.outputDir,
        manifest: result.manifestPath,
        numUnits: result.manifest.numUnits,
        degenerateUnitCount: result.stats.degenerateUnitCount,
      }, null, 2));
    }
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  }
}
