#!/usr/bin/env node
/* Generate the V5 source-only external-hit field from real GLB triangles.
 *
 * A row is one (unit, surface-start, direction) ray.  The six column files
 * are deliberately headerless little-endian arrays so training can mmap them
 * directly.  The manifest carries the shape and dtype contract.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { MeshBVH } from 'three-mesh-bvh';

import {
  createGltfLoader,
  loadGltf,
  matchTrianglesToComponents,
  validateRuntimeInputs,
} from '../generate_v5_instance_surface_samples.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const DEFAULT_ASSETS_DIR = path.join(REPO_ROOT, 'hkust-v3', 'assets');
const DEFAULT_RUNTIME_META = path.join(DEFAULT_ASSETS_DIR, 'runtimeVisibilityMeta.json');
const DEFAULT_GLB_INDEX = path.join(DEFAULT_ASSETS_DIR, 'glbIndex.json');
const DEFAULT_SURFACE_MANIFEST = path.join(
  REPO_ROOT,
  'neural_instance_culling',
  'dataset',
  'out',
  'v5_instance_surface_samples',
  'surface_manifest.json',
);
const DEFAULT_OUTPUT_DIR = path.join(
  REPO_ROOT,
  'neural_instance_culling',
  'dataset',
  'out',
  'v5_external_hit_probes',
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

export const EXTERNAL_HIT_PROBE_SCHEMA = 'parallel_external_hit_current_status-v1';
export const RAY_SCHEMA = 'surface_origin_first_external_hit_right_censor_v1';
export const SURFACE_STARTS_PER_UNIT = 16;
export const DIRECTIONS_PER_UNIT = 36;
export const RAYS_PER_UNIT = SURFACE_STARTS_PER_UNIT * DIRECTIONS_PER_UNIT;
export const POINTS_PER_UNIT = 256;
export const MAX_TRACE_DISTANCE_RATIO = 1024.0;
export const RAY_ORIGIN_OFFSET_RATIO = 1e-5;
export const DISTANCE_RATIOS = Object.freeze([
  0.25,
  0.5,
  1.0,
  2.0,
  4.0,
  8.0,
  16.0,
  32.0,
  64.0,
  128.0,
  256.0,
  512.0,
  1024.0,
]);
export const COLUMN_FILES = Object.freeze({
  unitIds: 'unit_ids_uint32.bin',
  directions: 'directions_float32.bin',
  hitDistances: 'hit_distances_float32.bin',
  maxDistances: 'max_distances_float32.bin',
  startIds: 'start_ids_uint8.bin',
  directionIds: 'direction_ids_uint8.bin',
});

const SURFACE_BINARY_MAGIC = 'GPV5';
const SURFACE_BINARY_VERSION = 1;
const SURFACE_BINARY_HEADER_BYTES = 16;
const AREA_EPSILON = 1e-14;
const UINT32_MAX = 0xffffffff;
const FLOAT_EPSILON = 1e-6;

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

function asUint32(value, name) {
  const number = asNonNegativeInteger(value, name);
  if (number > UINT32_MAX) throw new Error(`${name} must fit uint32`);
  return number;
}

function asVec3(value, name) {
  assert(Array.isArray(value) || ArrayBuffer.isView(value), `${name} must be a length-three vector`);
  assert(value.length === 3, `${name} must be a length-three vector`);
  return [
    asFiniteNumber(value[0], `${name}[0]`),
    asFiniteNumber(value[1], `${name}[1]`),
    asFiniteNumber(value[2], `${name}[2]`),
  ];
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJsonAtomically(filePath, value) {
  const temporary = `${filePath}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  fs.renameSync(temporary, filePath);
}

function finiteVectorLength(value) {
  return Math.hypot(value[0], value[1], value[2]);
}

function normalizeDirection(value, name) {
  const vector = asVec3(value, name);
  const length = finiteVectorLength(vector);
  assert(length > FLOAT_EPSILON, `${name} must be non-zero`);
  return vector.map((component) => component / length);
}

function boundsFromRecord(record, name, allowDegenerate = false) {
  const raw = record?.bounds || record?.aabb || record;
  assert(raw && typeof raw === 'object', `${name} must contain bounds`);
  let min;
  let max;
  if (raw.min !== undefined || raw.max !== undefined) {
    assert(raw.min !== undefined && raw.max !== undefined, `${name} must contain min and max`);
    min = asVec3(raw.min, `${name}.min`);
    max = asVec3(raw.max, `${name}.max`);
  } else {
    const center = asVec3(raw.center, `${name}.center`);
    const size = asVec3(raw.size, `${name}.size`);
    assert(size.every((value) => value >= 0), `${name}.size must be non-negative`);
    min = size.map((value, axis) => center[axis] - value * 0.5);
    max = size.map((value, axis) => center[axis] + value * 0.5);
  }
  assert(min.every((value, axis) => value <= max[axis]), `${name} has inverted bounds`);
  const size = min.map((value, axis) => max[axis] - value);
  const center = min.map((value, axis) => (value + max[axis]) * 0.5);
  const radius = Math.hypot(size[0], size[1], size[2]) * 0.5;
  assert(Number.isFinite(radius), `${name} must have a finite diagonal radius`);
  if (!allowDegenerate) {
    assert(radius > AREA_EPSILON, `${name} must have a positive diagonal radius`);
  }
  return { min, max, size, center, radius };
}

function normalizeUnitRecords(runtimeMeta, glbIndex, degenerateUnitIds = []) {
  const input = validateRuntimeInputs(runtimeMeta, glbIndex);
  const allowedDegenerate = new Set(degenerateUnitIds);
  const components = input.components.map((component) => ({
    ...component,
    componentGlobalId: asUint32(component.componentGlobalId, 'componentGlobalId'),
    bounds: boundsFromRecord(
      component.bounds,
      `component ${component.componentGlobalId} bounds`,
      allowedDegenerate.has(component.componentGlobalId),
    ),
  }));
  return {
    ...input,
    components,
    componentById: new Map(components.map((component) => [component.componentGlobalId, component])),
  };
}

function makeIcosahedronDirections() {
  const phi = (1 + Math.sqrt(5)) * 0.5;
  const coordinates = [
    [0, 1, phi],
    [0, 1, -phi],
    [0, -1, phi],
    [0, -1, -phi],
    [1, phi, 0],
    [1, -phi, 0],
    [-1, phi, 0],
    [-1, -phi, 0],
    [phi, 0, 1],
    [phi, 0, -1],
    [-phi, 0, 1],
    [-phi, 0, -1],
  ];
  return coordinates.map((value) => normalizeDirection(value, 'icosahedron direction'));
}

function makeFibonacciDirections() {
  const values = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let index = 0; index < 24; index += 1) {
    const y = 1 - 2 * (index + 0.5) / 24;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = goldenAngle * index;
    values.push([
      radius * Math.cos(theta),
      y,
      radius * Math.sin(theta),
    ]);
  }
  return values.map((value) => normalizeDirection(value, 'Fibonacci direction'));
}

export const FIXED_DIRECTIONS = Object.freeze([
  ...makeIcosahedronDirections(),
  ...makeFibonacciDirections(),
].map((direction) => Object.freeze(direction)));

assert(FIXED_DIRECTIONS.length === DIRECTIONS_PER_UNIT, 'V5 direction table must contain 36 directions');

function parseDenseSurfaceHeader(value) {
  const bytes = Buffer.isBuffer(value)
    ? value
    : Buffer.from(value.buffer, value.byteOffset || 0, value.byteLength);
  assert(bytes.length >= SURFACE_BINARY_HEADER_BYTES, 'surface point asset is shorter than its header');
  const header = {
    magic: bytes.subarray(0, 4).toString('ascii'),
    version: bytes.readUInt16LE(4),
    featureDim: bytes.readUInt16LE(6),
    numUnits: bytes.readUInt32LE(8),
    itemsPerUnit: bytes.readUInt32LE(12),
  };
  assert(header.magic === SURFACE_BINARY_MAGIC, 'surface point asset must use GPV5 binary magic');
  assert(header.version === SURFACE_BINARY_VERSION, 'unsupported surface point asset version');
  assert(header.featureDim === 6, 'surface point features must have dimension 6');
  assert(header.itemsPerUnit === POINTS_PER_UNIT, 'surface point asset must contain 256 points per unit');
  return header;
}

function readDenseSurfacePoints(filePath) {
  const bytes = fs.readFileSync(filePath);
  const header = parseDenseSurfaceHeader(bytes);
  const expectedBytes = SURFACE_BINARY_HEADER_BYTES
    + header.numUnits * header.itemsPerUnit * header.featureDim * Float32Array.BYTES_PER_ELEMENT;
  assert(bytes.byteLength === expectedBytes, `surface point asset size mismatch: ${filePath}`);
  const values = new Float32Array(
    bytes.buffer,
    bytes.byteOffset + SURFACE_BINARY_HEADER_BYTES,
    header.numUnits * header.itemsPerUnit * header.featureDim,
  );
  for (const value of values) assert(Number.isFinite(value), 'surface points contain non-finite values');
  return { header, values };
}

function parseNpy(filePath) {
  const bytes = fs.readFileSync(filePath);
  assert(bytes.length >= 10, `NPY file is too short: ${filePath}`);
  assert(bytes.subarray(0, 6).equals(Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])), `invalid NPY magic: ${filePath}`);
  const major = bytes[6];
  const minor = bytes[7];
  const headerBytes = major === 1
    ? bytes.readUInt16LE(8)
    : major === 2 || major === 3
      ? bytes.readUInt32LE(8)
      : 0;
  assert(headerBytes > 0, `unsupported NPY version ${major}.${minor}: ${filePath}`);
  const headerStart = major === 1 ? 10 : 12;
  const header = bytes.subarray(headerStart, headerStart + headerBytes).toString('latin1');
  const descrMatch = header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/);
  const shapeMatch = header.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/);
  const fortranMatch = header.match(/['"]fortran_order['"]\s*:\s*(True|False)/);
  assert(descrMatch && shapeMatch && fortranMatch, `invalid NPY header: ${filePath}`);
  assert(fortranMatch[1] === 'False', `Fortran-order NPY is not supported: ${filePath}`);
  const shape = shapeMatch[1]
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => Number(value));
  assert(shape.length > 0 && shape.every((value) => Number.isInteger(value) && value >= 0), `invalid NPY shape: ${filePath}`);
  const descr = descrMatch[1];
  const typeCode = descr.slice(1);
  const littleEndian = descr[0] === '<' || descr[0] === '|';
  assert(littleEndian, `big-endian NPY is not supported: ${filePath}`);
  const elementBytes = Number(typeCode.slice(1));
  assert(Number.isInteger(elementBytes) && elementBytes > 0, `invalid NPY dtype: ${descr}`);
  const count = shape.reduce((product, value) => product * value, 1);
  const dataOffset = headerStart + headerBytes;
  assert(dataOffset + count * elementBytes <= bytes.length, `NPY payload is truncated: ${filePath}`);
  const data = new Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < count; index += 1) {
    const offset = dataOffset + index * elementBytes;
    if (descr.endsWith('f4')) data[index] = view.getFloat32(offset, true);
    else if (descr.endsWith('f8')) data[index] = view.getFloat64(offset, true);
    else if (descr.endsWith('u1')) data[index] = view.getUint8(offset);
    else if (descr.endsWith('u4')) data[index] = view.getUint32(offset, true);
    else if (descr.endsWith('u8')) data[index] = Number(view.getBigUint64(offset, true));
    else throw new Error(`unsupported NPY dtype: ${descr}`);
  }
  return { descr, shape, data };
}

function resolveManifestFile(manifestPath, value, fallbackName) {
  const file = value || fallbackName;
  assert(typeof file === 'string' && file.length > 0, 'surface manifest file name must be non-empty');
  return path.resolve(path.dirname(manifestPath), file);
}

function boundsFromSurfaceManifest(manifestPath, manifest, unitIds, requestedUnitIds = unitIds) {
  const requested = new Set(requestedUnitIds);
  const byUnit = new Map();
  if (Array.isArray(manifest.records)) {
    for (const record of manifest.records) {
      const unitId = record.componentGlobalId ?? record.unitId;
      const normalization = record.normalization;
      if (unitId === undefined || !normalization || !requested.has(Number(unitId))) continue;
      const center = asVec3(normalization.center, `surface record ${unitId} center`);
      const radius = asFiniteNumber(normalization.halfDiagonal, `surface record ${unitId} halfDiagonal`);
      assert(radius > AREA_EPSILON, `surface record ${unitId} has invalid halfDiagonal`);
      byUnit.set(asUint32(unitId, `surface record ${unitId}`), { center, radius });
    }
  }
  if (byUnit.size < unitIds.length && manifest.files?.aabbMin && manifest.files?.aabbMax) {
    const mins = parseNpy(resolveManifestFile(manifestPath, manifest.files.aabbMin)).data;
    const maxs = parseNpy(resolveManifestFile(manifestPath, manifest.files.aabbMax)).data;
    assert(mins.length === unitIds.length * 3 && maxs.length === unitIds.length * 3, 'surface AABB arrays have an invalid shape');
    for (let index = 0; index < unitIds.length; index += 1) {
      if (!requested.has(unitIds[index])) continue;
      const min = mins.slice(index * 3, index * 3 + 3);
      const max = maxs.slice(index * 3, index * 3 + 3);
      const center = min.map((value, axis) => (value + max[axis]) * 0.5);
      const radius = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) * 0.5;
      assert(radius > AREA_EPSILON, `surface AABB ${unitIds[index]} has invalid diagonal radius`);
      byUnit.set(unitIds[index], { center, radius });
    }
  }
  return byUnit;
}

function normalizeSurfaceStarts(options, units) {
  const direct = options.surfaceStartsByUnit || options.surfaceStarts || options.surfacePoints;
  if (direct) {
    const result = new Map();
    const getValue = (unitId, index) => {
      if (direct instanceof Map) return direct.get(unitId);
      if (Array.isArray(direct)) {
        if (direct.length === units.length && direct[index] !== undefined) return direct[index];
        return direct[unitId];
      }
      if (typeof direct === 'object') return direct[unitId] ?? direct[String(unitId)];
      return undefined;
    };
    for (let index = 0; index < units.length; index += 1) {
      const unit = units[index];
      const value = getValue(unit.componentGlobalId, index);
      assert(value !== undefined, `surface starts are missing unit ${unit.componentGlobalId}`);
      const points = value.points || value.worldPoints || value;
      assert(Array.isArray(points) || ArrayBuffer.isView(points), `surface starts for unit ${unit.componentGlobalId} must be an array`);
      const pointList = ArrayBuffer.isView(points) && points.length >= SURFACE_STARTS_PER_UNIT * 3
        ? Array.from({ length: SURFACE_STARTS_PER_UNIT }, (_, pointIndex) => points.slice(pointIndex * 3, pointIndex * 3 + 3))
        : points;
      assert(pointList.length >= SURFACE_STARTS_PER_UNIT, `surface starts for unit ${unit.componentGlobalId} must contain 16 points`);
      const normalized = [];
      for (let pointIndex = 0; pointIndex < SURFACE_STARTS_PER_UNIT; pointIndex += 1) {
        normalized.push(asVec3(pointList[pointIndex], `surface start ${unit.componentGlobalId}/${pointIndex}`));
      }
      result.set(unit.componentGlobalId, normalized);
    }
    return result;
  }

  const manifestPath = path.resolve(
    options.surfaceManifestPath
      || options.surfaceManifest
      || options.surfaceMetaPath
      || DEFAULT_SURFACE_MANIFEST,
  );
  const manifest = options.surfaceManifestObject || readJson(manifestPath);
  const pointFile = options.surfacePointsPath || options.surfacePointsFile
    || manifest.files?.points || manifest.binary?.file;
  assert(pointFile, 'surface manifest must expose a points file');
  const pointsPath = path.isAbsolute(pointFile)
    ? pointFile
    : resolveManifestFile(manifestPath, pointFile);
  const { header, values } = readDenseSurfacePoints(pointsPath);
  assert(header.numUnits === (manifest.numUnits ?? header.numUnits), 'surface point count disagrees with surface manifest');
  const manifestUnitIds = manifest.unitIds
    ? manifest.unitIds.map((value, index) => asUint32(value, `surface unitIds[${index}]`))
    : Array.isArray(manifest.records)
      ? manifest.records.map((record, index) => asUint32(record.componentGlobalId ?? record.unitId, `surface record ${index}`))
      : manifest.files?.unitIds
        ? parseNpy(resolveManifestFile(manifestPath, manifest.files.unitIds)).data.map((value, index) => asUint32(value, `surface unitIds[${index}]`))
        : null;
  assert(manifestUnitIds && manifestUnitIds.length === header.numUnits, 'surface manifest must expose unit order');
  const requestedUnitIds = units.map((unit) => unit.componentGlobalId);
  const normalizationByUnit = boundsFromSurfaceManifest(
    manifestPath,
    manifest,
    manifestUnitIds,
    requestedUnitIds,
  );
  const result = new Map();
  const unitIndexById = new Map(manifestUnitIds.map((unitId, index) => [unitId, index]));
  for (const unit of units) {
    const unitId = unit.componentGlobalId;
    const surfaceIndex = unitIndexById.get(unitId);
    assert(surfaceIndex !== undefined, `surface asset is missing unit ${unitId}`);
    const normalization = normalizationByUnit.get(unitId) || unit.bounds;
    const rowOffset = surfaceIndex * POINTS_PER_UNIT * 6;
    const starts = [];
    for (let pointIndex = 0; pointIndex < SURFACE_STARTS_PER_UNIT; pointIndex += 1) {
      const offset = rowOffset + pointIndex * 6;
      const localPoint = [values[offset], values[offset + 1], values[offset + 2]];
      assert(localPoint.every((value) => Number.isFinite(value)), `surface point ${unitId}/${pointIndex} is not finite`);
      starts.push(localPoint.map((value, axis) => normalization.center[axis] + value * normalization.radius));
    }
    result.set(unitId, starts);
  }
  return result;
}

function vectorFromAttribute(attribute, index) {
  return [attribute.getX(index), attribute.getY(index), attribute.getZ(index)];
}

function transformPoint(THREE, point, matrix) {
  const value = new THREE.Vector3(point[0], point[1], point[2]).applyMatrix4(matrix);
  return [value.x, value.y, value.z];
}

function objectIsVisible(object) {
  for (let current = object; current; current = current.parent) {
    if (current.visible === false) return false;
  }
  return true;
}

function collectObjectTriangles(THREE, object, instanceIndex, sourceIndexStart, hintedUnitId) {
  const geometry = object.geometry;
  const position = geometry?.attributes?.position;
  if (!position) return { triangles: [], sourceTriangleCount: 0, degenerateTriangleCount: 0 };
  const index = geometry.index;
  const indexCount = index ? index.count : position.count;
  const sourceTriangleCount = Math.floor(indexCount / 3);
  const instanceMatrix = new THREE.Matrix4().identity();
  if (instanceIndex !== null) object.getMatrixAt(instanceIndex, instanceMatrix);
  const finalMatrix = new THREE.Matrix4().copy(object.matrixWorld);
  if (instanceIndex !== null) finalMatrix.multiply(instanceMatrix);
  const triangles = [];
  let degenerateTriangleCount = 0;
  for (let triangleIndex = 0; triangleIndex < sourceTriangleCount; triangleIndex += 1) {
    const ia = index ? index.getX(triangleIndex * 3) : triangleIndex * 3;
    const ib = index ? index.getX(triangleIndex * 3 + 1) : triangleIndex * 3 + 1;
    const ic = index ? index.getX(triangleIndex * 3 + 2) : triangleIndex * 3 + 2;
    const a = transformPoint(THREE, vectorFromAttribute(position, ia), finalMatrix);
    const b = transformPoint(THREE, vectorFromAttribute(position, ib), finalMatrix);
    const c = transformPoint(THREE, vectorFromAttribute(position, ic), finalMatrix);
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const cross = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    ];
    const crossLength = Math.hypot(...cross);
    if (
      ![...a, ...b, ...c].every(Number.isFinite)
      || !Number.isFinite(crossLength)
      || crossLength <= AREA_EPSILON
      || crossLength * 0.5 <= AREA_EPSILON
    ) {
      degenerateTriangleCount += 1;
      continue;
    }
    // The formal DoubleSide/alpha rule is conservative here: every decoded
    // triangle is a potential hit, so retaining material objects adds no
    // information to the probe result.
    triangles.push({
      a,
      b,
      c,
      sourceIndex: sourceIndexStart + triangleIndex,
      instanceComponentGlobalId: hintedUnitId,
    });
  }
  return { triangles, sourceTriangleCount, degenerateTriangleCount };
}

/**
 * Visit one temporary triangle batch at a time.
 *
 * The callback owns the batch only for the duration of the call.  Keeping the
 * callback boundary at a renderable object (and at one instance for an
 * InstancedMesh) prevents a large GLB from becoming one long-lived JS object
 * array before component matching.
 */
export function visitRenderableTriangleBatches(THREE, gltf, componentGlobalIds = [], onBatch) {
  assert(gltf?.scene, 'GLB has no scene');
  assert(typeof onBatch === 'function', 'onBatch must be a function');
  let sourceTriangleCount = 0;
  let degenerateTriangleCount = 0;
  let renderableObjectCount = 0;
  let usableTriangleCount = 0;
  let sourceIndex = 0;
  let instanceCursor = 0;
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((object) => {
    if ((!object.isMesh && !object.isInstancedMesh) || !objectIsVisible(object)) return;
    renderableObjectCount += 1;
    if (object.isInstancedMesh) {
      for (let instanceIndex = 0; instanceIndex < object.count; instanceIndex += 1) {
        const hintedUnitId = componentGlobalIds.length === object.count
          ? componentGlobalIds[instanceIndex]
          : componentGlobalIds[instanceCursor + instanceIndex] ?? null;
        const result = collectObjectTriangles(
          THREE,
          object,
          instanceIndex,
          sourceIndex,
          hintedUnitId,
        );
        sourceTriangleCount += result.sourceTriangleCount;
        degenerateTriangleCount += result.degenerateTriangleCount;
        usableTriangleCount += result.triangles.length;
        onBatch(result.triangles);
        sourceIndex += result.sourceTriangleCount;
      }
      instanceCursor += object.count;
    } else {
      const result = collectObjectTriangles(THREE, object, null, sourceIndex, null);
      sourceTriangleCount += result.sourceTriangleCount;
      degenerateTriangleCount += result.degenerateTriangleCount;
      usableTriangleCount += result.triangles.length;
      onBatch(result.triangles);
      sourceIndex += result.sourceTriangleCount;
    }
  });
  return {
    sourceTriangleCount,
    degenerateTriangleCount,
    renderableObjectCount,
    usableTriangleCount,
  };
}

/** Collect actual world-space render triangles. Intended for focused tests only. */
export function collectRenderableTriangles(THREE, gltf, componentGlobalIds = []) {
  const triangles = [];
  const stats = visitRenderableTriangleBatches(THREE, gltf, componentGlobalIds, (batch) => {
    for (const triangle of batch) triangles.push(triangle);
  });
  return { triangles, ...stats };
}

/**
 * Compact scene-wide triangle storage.
 *
 * Positions and unit ownership are kept in typed-array chunks.  No per-triangle
 * JS object survives a batch callback, and chunks are flattened exactly once
 * when the final BufferGeometry is created.
 */
export class CompactTriangleStore {
  constructor({ chunkTriangleCapacity = 262144 } = {}) {
    assert(Number.isInteger(chunkTriangleCapacity) && chunkTriangleCapacity > 0,
      'chunkTriangleCapacity must be a positive integer');
    this.chunkTriangleCapacity = chunkTriangleCapacity;
    this.chunks = [];
    this.triangleCount = 0;
  }

  get triangleObjectCount() {
    return 0;
  }

  _ensureChunk() {
    const last = this.chunks[this.chunks.length - 1];
    if (last && last.count < this.chunkTriangleCapacity) return last;
    const chunk = {
      positions: new Float32Array(this.chunkTriangleCapacity * 9),
      unitIds: new Uint32Array(this.chunkTriangleCapacity),
      count: 0,
    };
    this.chunks.push(chunk);
    return chunk;
  }

  appendTriangle(unitId, triangle) {
    const normalizedUnitId = asUint32(unitId, 'triangle unitId');
    const a = asVec3(triangle.a, `triangle ${this.triangleCount}.a`);
    const b = asVec3(triangle.b, `triangle ${this.triangleCount}.b`);
    const c = asVec3(triangle.c, `triangle ${this.triangleCount}.c`);
    assertNonDegenerateTriangle(a, b, c, this.triangleCount);
    const chunk = this._ensureChunk();
    const offset = chunk.count * 9;
    chunk.positions.set(a, offset);
    chunk.positions.set(b, offset + 3);
    chunk.positions.set(c, offset + 6);
    chunk.unitIds[chunk.count] = normalizedUnitId;
    chunk.count += 1;
    this.triangleCount += 1;
  }

  appendTriangles(unitId, triangles) {
    assert(Array.isArray(triangles), 'triangle batch must be an array');
    for (const triangle of triangles) this.appendTriangle(unitId, triangle);
  }

  storageStats() {
    const positionBytes = this.triangleCount * 9 * Float32Array.BYTES_PER_ELEMENT;
    const unitIdBytes = this.triangleCount * Uint32Array.BYTES_PER_ELEMENT;
    return {
      triangleCount: this.triangleCount,
      chunkCount: this.chunks.length,
      positionBytes,
      unitIdBytes,
      typedArrayBytes: positionBytes + unitIdBytes,
      triangleObjectCount: this.triangleObjectCount,
      storage: 'chunked_float32_positions_uint32_unit_ids',
    };
  }

  /** Flatten chunks and release them immediately after the copy. */
  finalize() {
    assert(this.triangleCount > 0, 'at least one real triangle is required');
    const positions = new Float32Array(this.triangleCount * 9);
    const triangleUnitIds = new Uint32Array(this.triangleCount);
    let triangleOffset = 0;
    for (const chunk of this.chunks) {
      positions.set(chunk.positions.subarray(0, chunk.count * 9), triangleOffset * 9);
      triangleUnitIds.set(chunk.unitIds.subarray(0, chunk.count), triangleOffset);
      triangleOffset += chunk.count;
    }
    const stats = this.storageStats();
    this.chunks = [];
    return { positions, triangleUnitIds, stats };
  }
}

function compactDirectTriangles(trianglesByUnit, units) {
  const valuesByUnit = new Map(units.map((unit) => [unit.componentGlobalId, undefined]));
  if (trianglesByUnit instanceof Map) {
    for (const [rawUnitId, triangles] of trianglesByUnit) {
      const unitId = asUint32(rawUnitId, 'triangle unitId');
      assert(valuesByUnit.has(unitId), `triangle input contains unknown unit ${unitId}`);
      valuesByUnit.set(unitId, triangles);
    }
  } else if (trianglesByUnit && typeof trianglesByUnit === 'object') {
    for (const [rawUnitId, triangles] of Object.entries(trianglesByUnit)) {
      const unitId = asUint32(rawUnitId, 'triangle unitId');
      assert(valuesByUnit.has(unitId), `triangle input contains unknown unit ${unitId}`);
      valuesByUnit.set(unitId, triangles);
    }
  } else {
    throw new Error('trianglesByUnit must be a Map or object');
  }

  const store = new CompactTriangleStore();
  for (const unit of units) {
    const triangles = valuesByUnit.get(unit.componentGlobalId);
    assert(Array.isArray(triangles) && triangles.length > 0, `triangle input is missing unit ${unit.componentGlobalId}`);
    store.appendTriangles(unit.componentGlobalId, triangles);
  }
  return store;
}

async function importDefaultDependencies() {
  const [THREE, { GLTFLoader }, { DRACOLoader }, { MeshoptDecoder }] = await Promise.all([
    import(THREE_URL),
    import(GLTF_LOADER_URL),
    import(DRACO_LOADER_URL),
    import(MESHOPT_URL),
  ]);
  return { THREE, GLTFLoader, DRACOLoader, MeshoptDecoder };
}

function assertNonDegenerateTriangle(a, b, c, index) {
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const cross = [
    ab[1] * ac[2] - ab[2] * ac[1],
    ab[2] * ac[0] - ab[0] * ac[2],
    ab[0] * ac[1] - ab[1] * ac[0],
  ];
  assert(Math.hypot(...cross) > AREA_EPSILON, `triangle ${index} is degenerate`);
}

/** Build the exact triangle acceleration structure used by every probe. */
export function buildTriangleBvh(
  THREE,
  trianglesByUnit,
  MeshBvh = MeshBVH,
  { targetLeafSize = 10 } = {},
) {
  assert(THREE && THREE.BufferGeometry, 'buildTriangleBvh requires Three.js');
  const leafSize = asPositiveInteger(targetLeafSize, 'BVH targetLeafSize');
  const store = trianglesByUnit instanceof CompactTriangleStore
    ? trianglesByUnit
    : compactDirectTriangles(
      trianglesByUnit,
      (trianglesByUnit instanceof Map
        ? [...trianglesByUnit.keys()].map((value, index) => asUint32(value, `triangle unitId[${index}]`))
        : Object.keys(trianglesByUnit || {}).map((value, index) => asUint32(value, `triangle unitId[${index}]`)))
        .map((componentGlobalId) => ({ componentGlobalId })),
    );
  const { positions, triangleUnitIds, stats } = store.finalize();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const bvh = new MeshBvh(geometry, { indirect: false, targetLeafSize: leafSize });
  return {
    geometry,
    bvh,
    triangleUnitIds,
    triangleCount: stats.triangleCount,
    memoryStats: {
      ...stats,
      retainedTriangleObjects: 0,
      geometryPositionBytes: positions.byteLength,
      geometryUnitIdBytes: triangleUnitIds.byteLength,
      storage: 'buffer_geometry_float32_positions_uint32_unit_ids',
    },
  };
}

/** Trace one ray against real triangles, excluding every triangle of its unit. */
export function traceNearestExternalHit({
  THREE,
  bvhIndex,
  startPoint,
  direction,
  unitId,
  radius,
  maxTraceDistanceRatio = MAX_TRACE_DISTANCE_RATIO,
  originOffsetRatio = RAY_ORIGIN_OFFSET_RATIO,
}) {
  assert(THREE && THREE.Ray, 'traceNearestExternalHit requires Three.js');
  assert(bvhIndex?.bvh && bvhIndex.triangleUnitIds, 'traceNearestExternalHit requires a BVH index');
  const start = asVec3(startPoint, 'startPoint');
  const dir = normalizeDirection(direction, 'direction');
  const rayRadius = asFiniteNumber(radius, 'radius');
  assert(rayRadius > AREA_EPSILON, 'radius must be positive');
  const maxDistance = asFiniteNumber(maxTraceDistanceRatio, 'maxTraceDistanceRatio') * rayRadius;
  const originOffset = asFiniteNumber(originOffsetRatio, 'originOffsetRatio') * rayRadius;
  const origin = start.map((value, axis) => value + dir[axis] * originOffset);
  const ray = new THREE.Ray(
    new THREE.Vector3(origin[0], origin[1], origin[2]),
    new THREE.Vector3(dir[0], dir[1], dir[2]),
  );
  // The Color-ID sampler and the viewer's formal default both render
  // DoubleSide. Passing it here is the ray equivalent of that protocol.
  const intersections = bvhIndex.bvh.raycast(ray, THREE.DoubleSide);
  let nearest = null;
  for (const intersection of intersections) {
    const triangleIndex = Number(intersection.faceIndex);
    if (!Number.isInteger(triangleIndex) || triangleIndex < 0) continue;
    if (bvhIndex.triangleUnitIds[triangleIndex] === asUint32(unitId, 'unitId')) continue;
    const distance = originOffset + Number(intersection.distance);
    if (!Number.isFinite(distance) || distance <= 0 || distance > maxDistance + FLOAT_EPSILON) continue;
    if (!nearest || distance < nearest.distance) {
      nearest = {
        distance,
        triangleIndex,
        point: intersection.point,
      };
    }
  }
  return nearest;
}

/** Trace the fixed 16x36 ray block for one unit. */
export function traceUnitRays({ THREE, bvhIndex, unitId, starts, radius }) {
  assert(Array.isArray(starts) && starts.length === SURFACE_STARTS_PER_UNIT, 'a unit must have exactly 16 starts');
  const hitDistances = new Float32Array(RAYS_PER_UNIT);
  hitDistances.fill(Number.NaN);
  const maxDistances = new Float32Array(RAYS_PER_UNIT);
  const directions = new Float32Array(RAYS_PER_UNIT * 3);
  const startIds = new Uint8Array(RAYS_PER_UNIT);
  const directionIds = new Uint8Array(RAYS_PER_UNIT);
  const maxDistance = MAX_TRACE_DISTANCE_RATIO * radius;
  for (let startId = 0; startId < SURFACE_STARTS_PER_UNIT; startId += 1) {
    const start = asVec3(starts[startId], `surface start ${unitId}/${startId}`);
    for (let directionId = 0; directionId < DIRECTIONS_PER_UNIT; directionId += 1) {
      const row = directionId * SURFACE_STARTS_PER_UNIT + startId;
      const direction = FIXED_DIRECTIONS[directionId];
      directions.set(direction, row * 3);
      startIds[row] = startId;
      directionIds[row] = directionId;
      maxDistances[row] = maxDistance;
      const hit = traceNearestExternalHit({
        THREE,
        bvhIndex,
        startPoint: start,
        direction,
        unitId,
        radius,
      });
      if (hit) hitDistances[row] = hit.distance;
    }
  }
  return { hitDistances, maxDistances, directions, startIds, directionIds };
}

function columnByteLength(name, rowCount) {
  if (name === 'unitIds' || name === 'maxDistances' || name === 'hitDistances' || name === 'startIds' || name === 'directionIds') {
    return rowCount * (name === 'unitIds' || name === 'maxDistances' || name === 'hitDistances' ? 4 : 1);
  }
  if (name === 'directions') return rowCount * 3 * 4;
  throw new Error(`unknown probe column ${name}`);
}

function encodeColumn(name, values) {
  if (name === 'unitIds') {
    const output = Buffer.alloc(values.length * 4);
    for (let index = 0; index < values.length; index += 1) output.writeUInt32LE(values[index], index * 4);
    return output;
  }
  if (name === 'directions' || name === 'hitDistances' || name === 'maxDistances') {
    const output = Buffer.alloc(values.length * 4);
    for (let index = 0; index < values.length; index += 1) output.writeFloatLE(values[index], index * 4);
    return output;
  }
  if (name === 'startIds' || name === 'directionIds') return Buffer.from(values);
  throw new Error(`unknown probe column ${name}`);
}

function decodeColumn(name, bytes, rowCount) {
  assert(bytes.byteLength === columnByteLength(name, rowCount), `${name} column byte length is invalid`);
  if (name === 'unitIds') {
    const values = new Uint32Array(rowCount);
    for (let index = 0; index < rowCount; index += 1) values[index] = bytes.readUInt32LE(index * 4);
    return values;
  }
  if (name === 'directions' || name === 'hitDistances' || name === 'maxDistances') {
    const count = name === 'directions' ? rowCount * 3 : rowCount;
    const values = new Float32Array(count);
    for (let index = 0; index < count; index += 1) values[index] = bytes.readFloatLE(index * 4);
    return values;
  }
  return new Uint8Array(bytes);
}

function makeColumnManifestFiles(outputDir) {
  return Object.fromEntries(Object.entries(COLUMN_FILES).map(([name, file]) => [name, path.relative(outputDir, path.join(outputDir, file)) || file]));
}

export function validateProbeManifest(manifest) {
  assert(manifest && typeof manifest === 'object', 'probe manifest must be an object');
  assert(manifest.schema === EXTERNAL_HIT_PROBE_SCHEMA, 'probe manifest has an unexpected schema');
  assert(manifest.version === 1, 'unsupported external-hit probe manifest version');
  assert(manifest.assetKind === 'external_hit_probe', 'probe manifest assetKind is invalid');
  assert(typeof manifest.sceneId === 'string' && manifest.sceneId.length > 0, 'probe manifest sceneId is required');
  assert(manifest.split === 'train' && manifest.sourceRole === 'source_train', 'external-hit probes are source-train-only');
  assert(manifest.containsVisibilityLabels === false, 'external-hit probes cannot contain visibility labels');
  assert(manifest.raySchema === RAY_SCHEMA, 'probe ray schema is invalid');
  assert(manifest.surfaceStartsPerUnit === SURFACE_STARTS_PER_UNIT, 'surfaceStartsPerUnit is not 16');
  assert(manifest.directionsPerUnit === DIRECTIONS_PER_UNIT, 'directionsPerUnit is not 36');
  assert(manifest.directionSet === 'icosahedron12_plus_fibonacci24', 'probe direction set is invalid');
  assert(JSON.stringify(manifest.distanceRatios) === JSON.stringify(DISTANCE_RATIOS), 'probe distance grid is invalid');
  assert(manifest.maxTraceDistanceRatio === MAX_TRACE_DISTANCE_RATIO, 'maxTraceDistanceRatio is invalid');
  assert(Number.isInteger(manifest.numUnits) && manifest.numUnits > 0, 'probe numUnits must be positive');
  assert(manifest.rowCount === manifest.numUnits * RAYS_PER_UNIT, 'probe rowCount is invalid');
  assert(Array.isArray(manifest.unitIds) && manifest.unitIds.length === manifest.numUnits, 'probe unitIds are invalid');
  assert(manifest.unitIds.every((value, index) => Number.isInteger(value) && value >= 0
    && value <= UINT32_MAX && (index === 0 || value > manifest.unitIds[index - 1])), 'probe unitIds must be sorted and unique');
  if (manifest.sceneNumUnits !== undefined) {
    assert(Number.isInteger(manifest.sceneNumUnits) && manifest.sceneNumUnits >= manifest.numUnits, 'probe sceneNumUnits is invalid');
  }
  assert(manifest.storage === 'columnar_memmap_little_endian_v1', 'probe storage is invalid');
  assert(manifest.eventEncoding === 'finite_hit_distance_is_event', 'probe event encoding is invalid');
  const expectedFiles = Object.keys(COLUMN_FILES).sort();
  assert(manifest.files && JSON.stringify(Object.keys(manifest.files).sort()) === JSON.stringify(expectedFiles), 'probe column file set is invalid');
  assert(manifest.columnDtypes?.unitIds === 'uint32', 'unitIds dtype must be uint32');
  assert(manifest.columnDtypes?.directions === 'float32[3]', 'directions dtype must be float32[3]');
  assert(manifest.columnDtypes?.hitDistances === 'float32', 'hitDistances dtype must be float32');
  assert(manifest.columnDtypes?.maxDistances === 'float32', 'maxDistances dtype must be float32');
  assert(manifest.columnDtypes?.startIds === 'uint8', 'startIds dtype must be uint8');
  assert(manifest.columnDtypes?.directionIds === 'uint8', 'directionIds dtype must be uint8');
  assert(manifest.permissions?.assetKind === 'external_hit_probe', 'probe permissions are missing assetKind');
  assert(manifest.permissions?.access === 'source_train_only', 'probe permissions must be source_train_only');
  for (const name of expectedFiles) assert(typeof manifest.files[name] === 'string' && manifest.files[name].length > 0, `probe file ${name} is invalid`);
  return manifest;
}

export function makeProbeManifest({ sceneId, unitIds, totalUnitCount, outputDir, shardIndex = null, shardCount = null }) {
  const selectedUnitIds = unitIds.map((value, index) => asUint32(value, `unitIds[${index}]`));
  const rowCount = selectedUnitIds.length * RAYS_PER_UNIT;
  const manifest = {
    schema: EXTERNAL_HIT_PROBE_SCHEMA,
    version: 1,
    assetKind: 'external_hit_probe',
    sceneId,
    split: 'train',
    sourceRole: 'source_train',
    containsVisibilityLabels: false,
    permissions: {
      assetKind: 'external_hit_probe',
      access: 'source_train_only',
      heldOutReadable: false,
      visibilityLabelsIncluded: false,
    },
    raySchema: RAY_SCHEMA,
    surfaceStartsPerUnit: SURFACE_STARTS_PER_UNIT,
    directionsPerUnit: DIRECTIONS_PER_UNIT,
    directionSet: 'icosahedron12_plus_fibonacci24',
    distanceRatios: [...DISTANCE_RATIOS],
    maxTraceDistanceRatio: MAX_TRACE_DISTANCE_RATIO,
    recordFields: [
      'sceneId',
      'unitId',
      'probeId',
      'startPointWorld',
      'directionWorld',
      'hitDistance',
      'maxTraceDistance',
      'event',
    ],
    recordsFile: 'external_hit_probes.columnar',
    storage: 'columnar_memmap_little_endian_v1',
    rowCount,
    rowLayout: '[unit][directionId][startId]',
    eventEncoding: 'finite_hit_distance_is_event',
    hitDistanceOrigin: 'original_surface_start_world',
    rayOriginOffset: 'direction_world_times_1e-5_times_unit_radius',
    unitRadius: 'component_aabb_half_diagonal',
    unitOrder: 'unitIds_column_order_matches_source_unit_order',
    numUnits: selectedUnitIds.length,
    sceneNumUnits: totalUnitCount,
    unitIds: selectedUnitIds,
    columnDtypes: {
      unitIds: 'uint32',
      directions: 'float32[3]',
      hitDistances: 'float32',
      maxDistances: 'float32',
      startIds: 'uint8',
      directionIds: 'uint8',
    },
    columnShapes: {
      unitIds: [rowCount],
      directions: [rowCount, 3],
      hitDistances: [rowCount],
      maxDistances: [rowCount],
      startIds: [rowCount],
      directionIds: [rowCount],
    },
    byteOrder: 'little-endian',
    files: makeColumnManifestFiles(outputDir),
    materialRule: 'formal_color_id_double_side_with_source_alpha_test_provenance',
    acceleration: 'three-mesh-bvh_exact_world_triangles_compact_typed_storage',
  };
  if (shardIndex !== null) {
    manifest.shard = {
      index: shardIndex,
      count: shardCount,
      selectedUnitIds,
      complete: true,
    };
  }
  return validateProbeManifest(manifest);
}

function selectedShardUnits(units, shardIndex, shardCount) {
  if (shardIndex === null || shardIndex === undefined) return units;
  assert(Number.isInteger(shardCount) && shardCount > 0, 'shardCount must be positive when shardIndex is provided');
  assert(Number.isInteger(shardIndex) && shardIndex >= 0 && shardIndex < shardCount, 'shardIndex is outside shardCount');
  const start = Math.floor(units.length * shardIndex / shardCount);
  const end = Math.floor(units.length * (shardIndex + 1) / shardCount);
  return units.slice(start, end);
}

function sameStringArray(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function openColumnOutputs(outputDir, manifest, resume) {
  fs.mkdirSync(outputDir, { recursive: true });
  const rowCount = manifest.rowCount;
  const descriptors = {};
  for (const [name, relative] of Object.entries(manifest.files)) {
    const filePath = path.resolve(outputDir, relative);
    const exists = fs.existsSync(filePath);
    const expected = columnByteLength(name, rowCount);
    if (exists && resume) {
      assert(fs.statSync(filePath).size === expected, `${name} column cannot be resumed: byte length mismatch`);
    }
    const fd = fs.openSync(filePath, exists && resume ? 'r+' : 'w+');
    fs.ftruncateSync(fd, expected);
    descriptors[name] = { fd, filePath };
  }
  return descriptors;
}

function closeColumnOutputs(descriptors) {
  for (const descriptor of Object.values(descriptors || {})) {
    if (descriptor?.fd !== undefined) fs.closeSync(descriptor.fd);
  }
}

function writeUnitColumns(descriptors, localUnitIndex, unitId, rows) {
  const rowOffset = localUnitIndex * RAYS_PER_UNIT;
  const values = {
    unitIds: new Uint32Array(RAYS_PER_UNIT).fill(unitId),
    directions: rows.directions,
    hitDistances: rows.hitDistances,
    maxDistances: rows.maxDistances,
    startIds: rows.startIds,
    directionIds: rows.directionIds,
  };
  for (const name of Object.keys(COLUMN_FILES)) {
    const encoded = encodeColumn(name, values[name]);
    const offset = name === 'directions' ? rowOffset * 3 * 4 : rowOffset * (name === 'unitIds' || name === 'hitDistances' || name === 'maxDistances' ? 4 : 1);
    fs.writeSync(descriptors[name].fd, encoded, 0, encoded.length, offset);
  }
}

function readProgress(progressPath, selectedUnitIds, manifest) {
  if (!fs.existsSync(progressPath)) return new Set();
  const progress = readJson(progressPath);
  if (progress.schema !== EXTERNAL_HIT_PROBE_SCHEMA || progress.rowCount !== manifest.rowCount) return new Set();
  if (!Array.isArray(progress.selectedUnitIds) || !sameStringArray(progress.selectedUnitIds.map(String), selectedUnitIds.map(String))) return new Set();
  return new Set((progress.completedUnitIds || []).map((value) => asUint32(value, 'completedUnitId')));
}

function writeProgress(progressPath, selectedUnitIds, completedUnitIds, manifest) {
  writeJsonAtomically(progressPath, {
    schema: EXTERNAL_HIT_PROBE_SCHEMA,
    version: 1,
    rowCount: manifest.rowCount,
    selectedUnitIds,
    completedUnitIds: [...completedUnitIds].sort((left, right) => left - right),
  });
}

function manifestIsComplete(outputDir, manifest) {
  if (!manifest || manifest.storage !== 'columnar_memmap_little_endian_v1') return false;
  try {
    validateProbeManifest(manifest);
    return Object.values(manifest.files).every((relative) => (
      fs.existsSync(path.resolve(outputDir, relative))
      && fs.statSync(path.resolve(outputDir, relative)).size === columnByteLength(
        Object.entries(manifest.files).find(([, value]) => value === relative)?.[0],
        manifest.rowCount,
      )
    ));
  } catch {
    return false;
  }
}

function disposeLoadedGltf(gltf) {
  const textures = new Set();
  const materials = new Set();
  gltf?.scene?.traverse((object) => {
    object.geometry?.dispose?.();
    const objectMaterials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of objectMaterials) {
      if (!material || materials.has(material)) continue;
      materials.add(material);
      for (const value of Object.values(material)) {
        if (!value?.isTexture || textures.has(value)) continue;
        textures.add(value);
        const source = value.source?.data?.src || value.image?.src;
        if (typeof source === 'string' && source.startsWith('blob:')) {
          globalThis.URL?.revokeObjectURL?.(source);
        }
        value.dispose?.();
      }
      material.dispose?.();
    }
  });
  // GLTFParser keeps decoded buffers, textures and dependency records alive
  // even after the scene is no longer referenced.  Clear those registries
  // before the next GLB is parsed.
  const parser = gltf?.parser;
  parser?.cache?.removeAll?.();
  parser?.associations?.clear?.();
  if (parser) {
    parser.json = null;
    parser.extensions = {};
    parser.plugins = {};
    parser.primitiveCache = {};
    parser.nodeCache = {};
    parser.meshCache = { refs: {}, uses: {} };
    parser.cameraCache = { refs: {}, uses: {} };
    parser.lightCache = { refs: {}, uses: {} };
    parser.sourceCache = {};
    parser.textureCache = {};
    parser.nodeNamesUsed = {};
    parser.options = {};
    parser.textureLoader = null;
    parser.fileLoader = null;
  }
  gltf?.scene?.clear?.();
}

function collectGarbageIfAvailable() {
  if (typeof globalThis.gc !== 'function') return false;
  globalThis.gc();
  return true;
}

async function loadTrianglesFromGlbs({
  dependencies,
  input,
  assetsDir,
  progressEvery = 50,
  gcEvery = 0,
  triangleChunkCapacity = 262144,
}) {
  const triangleStore = new CompactTriangleStore({ chunkTriangleCapacity: triangleChunkCapacity });
  const assignedTriangleCounts = new Map(input.components.map((component) => [component.componentGlobalId, 0]));
  const loaderBundle = createGltfLoader(dependencies);
  const totals = {
    glbCountProcessed: 0,
    sourceTriangleCount: 0,
    degenerateTriangleCount: 0,
    usableTriangleCount: 0,
    matchedTriangleCount: 0,
    unmatchedTriangleCount: 0,
    ambiguousTriangleCount: 0,
    renderableObjectCount: 0,
  };
  try {
    for (let index = 0; index < input.entries.length; index += 1) {
      const entry = input.entries[index];
      const componentIds = input.componentIdsByGlb.get(entry.globalId) || [];
      if (componentIds.length === 0) continue;
      const filePath = path.resolve(assetsDir, entry.path);
      let gltf = null;
      try {
        gltf = await loadGltf(loaderBundle.loader, filePath);
        const componentRecords = componentIds.map((componentId) => input.componentById.get(componentId));
        const sceneScale = Math.max(...componentRecords.map((component) => component.bounds.radius), 1);
        const geometry = visitRenderableTriangleBatches(
          dependencies.THREE,
          gltf,
          componentIds,
          (batch) => {
            if (batch.length === 0) return;
            const matched = matchTrianglesToComponents(batch, componentRecords, { sceneScale });
            totals.matchedTriangleCount += matched.stats.matchedTriangleCount;
            totals.unmatchedTriangleCount += matched.stats.unmatchedTriangleCount;
            totals.ambiguousTriangleCount += matched.stats.ambiguousTriangleCount;
            for (const [unitId, assigned] of matched.assignments) {
              if (assigned.length === 0) continue;
              triangleStore.appendTriangles(unitId, assigned);
              assignedTriangleCounts.set(unitId, assignedTriangleCounts.get(unitId) + assigned.length);
            }
          },
        );
        totals.glbCountProcessed += 1;
        totals.sourceTriangleCount += geometry.sourceTriangleCount;
        totals.degenerateTriangleCount += geometry.degenerateTriangleCount;
        totals.usableTriangleCount += geometry.usableTriangleCount;
        totals.renderableObjectCount += geometry.renderableObjectCount;
        assert(geometry.usableTriangleCount > 0, `GLB has no usable triangles: ${filePath}`);
      } finally {
        disposeLoadedGltf(gltf);
        gltf = null;
      }
      if (progressEvery > 0 && totals.glbCountProcessed % progressEvery === 0) {
        const memory = process.memoryUsage?.();
        const memoryText = memory
          ? ` heapUsedMiB=${(memory.heapUsed / 1048576).toFixed(1)} rssMiB=${(memory.rss / 1048576).toFixed(1)}`
          : '';
        console.log(
          `[v5-external-hit] ${totals.glbCountProcessed}/${input.entries.length} GLBs decoded`
          + ` compactTriangles=${triangleStore.triangleCount}${memoryText}`,
        );
      }
      if (gcEvery > 0 && totals.glbCountProcessed % gcEvery === 0) collectGarbageIfAvailable();
    }
  } finally {
    loaderBundle.dracoLoader?.dispose?.();
  }
  for (const [unitId, triangleCount] of assignedTriangleCounts) {
    assert(triangleCount > 0, `no real triangles were assigned to unit ${unitId}`);
  }
  return {
    triangleStore,
    totals: {
      ...totals,
      compactTriangleCount: triangleStore.triangleCount,
      retainedTriangleObjects: triangleStore.triangleObjectCount,
    },
  };
}

function normalizeOptions(options = {}) {
  const assetsDir = path.resolve(options.assetsDir || DEFAULT_ASSETS_DIR);
  const runtimeMetaPath = path.resolve(options.runtimeMetaPath || options.runtimeMetaFile || DEFAULT_RUNTIME_META);
  const glbIndexPath = path.resolve(options.glbIndexPath || options.glbIndexFile || DEFAULT_GLB_INDEX);
  const outputDir = path.resolve(options.outputDir || options.output || options.outputPath || DEFAULT_OUTPUT_DIR);
  const manifestPath = path.resolve(
    options.manifestPath
      || options.manifest
      || options.metaPath
      || path.join(outputDir, 'external_hit_probe_manifest.json'),
  );
  const sceneId = String(options.sceneId || '').trim();
  const shardIndex = options.shardIndex === undefined || options.shardIndex === null
    ? null
    : asNonNegativeInteger(options.shardIndex, 'shardIndex');
  const shardCount = options.shardCount === undefined || options.shardCount === null
    ? null
    : asPositiveInteger(options.shardCount, 'shardCount');
  if (shardIndex !== null) assert(shardCount !== null, 'shardCount is required with shardIndex');
  return {
    ...options,
    assetsDir,
    runtimeMetaPath,
    glbIndexPath,
    outputDir,
    manifestPath,
    sceneId,
    shardIndex,
    shardCount,
    resume: options.resume !== false,
    progressEvery: options.progressEvery === undefined ? 50 : asNonNegativeInteger(options.progressEvery, 'progressEvery'),
    gcEvery: options.gcEvery === undefined ? 0 : asNonNegativeInteger(options.gcEvery, 'gcEvery'),
    triangleChunkCapacity: options.triangleChunkCapacity === undefined
      ? 262144
      : asPositiveInteger(options.triangleChunkCapacity, 'triangleChunkCapacity'),
    bvhTargetLeafSize: options.bvhTargetLeafSize === undefined
      ? 10
      : asPositiveInteger(options.bvhTargetLeafSize, 'bvhTargetLeafSize'),
  };
}


function surfaceDegenerateUnitIds(options) {
  const direct = options.surfaceStartsByUnit || options.surfaceStarts || options.surfacePoints;
  if (direct) return [];
  const manifestPath = path.resolve(
    options.surfaceManifestPath
      || options.surfaceManifest
      || options.surfaceMetaPath
      || DEFAULT_SURFACE_MANIFEST,
  );
  const manifest = options.surfaceManifestObject || readJson(manifestPath);
  const values = manifest.degenerateUnitIds || [];
  assert(Array.isArray(values), 'surface degenerateUnitIds must be an array');
  return values.map((value, index) => asUint32(value, `surface degenerateUnitIds[${index}]`));
}


export function excludeDegenerateProbeUnits(input, degenerateUnitIds) {
  const excluded = new Set(degenerateUnitIds.map((value) => asUint32(value, 'degenerate unit ID')));
  if (excluded.size === 0) return input;
  for (const unitId of excluded) {
    assert(input.componentById.has(unitId), `surface degenerate unit ${unitId} is absent from runtime metadata`);
  }
  const components = input.components.filter((unit) => !excluded.has(unit.componentGlobalId));
  assert(components.length > 0, 'all runtime units are degenerate');
  const componentById = new Map(components.map((unit) => [unit.componentGlobalId, unit]));
  const componentIdsByGlb = new Map();
  for (const [glbId, unitIds] of input.componentIdsByGlb) {
    const retained = unitIds.filter((unitId) => !excluded.has(unitId));
    if (retained.length > 0) componentIdsByGlb.set(glbId, retained);
  }
  return { ...input, components, componentById, componentIdsByGlb };
}

/** Generate one complete or partitioned binary external-hit probe asset. */
export async function generateExternalHitProbes(options = {}) {
  const args = normalizeOptions(options);
  const runtimeMeta = args.runtimeMeta || readJson(args.runtimeMetaPath);
  const glbIndex = args.glbIndex || readJson(args.glbIndexPath);
  const degenerateUnitIds = surfaceDegenerateUnitIds(args);
  const input = normalizeUnitRecords(runtimeMeta, glbIndex, degenerateUnitIds);
  const probeInput = excludeDegenerateProbeUnits(input, degenerateUnitIds);
  const sceneId = args.sceneId || String(runtimeMeta.sceneId || runtimeMeta.scene || path.basename(args.assetsDir));
  assert(sceneId.length > 0, 'sceneId is required');
  const selectedUnits = selectedShardUnits(probeInput.components, args.shardIndex, args.shardCount);
  assert(selectedUnits.length > 0, 'selected shard contains no units');
  const selectedUnitIds = selectedUnits.map((unit) => unit.componentGlobalId);
  const manifest = makeProbeManifest({
    sceneId,
    unitIds: selectedUnitIds,
    totalUnitCount: input.components.length,
    outputDir: args.outputDir,
    shardIndex: args.shardIndex,
    shardCount: args.shardCount,
  });
  fs.mkdirSync(args.outputDir, { recursive: true });
  if (args.resume && fs.existsSync(args.manifestPath)) {
    const existing = readJson(args.manifestPath);
    if (manifestIsComplete(args.outputDir, existing)
      && existing.sceneId === manifest.sceneId
      && existing.numUnits === manifest.numUnits
      && sameStringArray((existing.unitIds || []).map(String), manifest.unitIds.map(String))) {
      return { outputDir: args.outputDir, manifestPath: args.manifestPath, manifest: existing, resumed: true };
    }
  }

  const dependencies = args.dependencies
    ? {
      ...(args.trianglesByUnit ? {} : await importDefaultDependencies()),
      ...args.dependencies,
    }
    : await importDefaultDependencies();
  const startsByUnit = normalizeSurfaceStarts(args, probeInput.components);
  const loadedTriangles = args.trianglesByUnit
    ? {
      triangleStore: compactDirectTriangles(args.trianglesByUnit, probeInput.components),
      totals: { source: 'direct', retainedTriangleObjects: 0 },
    }
    : await loadTrianglesFromGlbs({
      dependencies,
      input: probeInput,
      assetsDir: args.assetsDir,
      progressEvery: args.progressEvery,
      gcEvery: args.gcEvery,
      triangleChunkCapacity: args.triangleChunkCapacity,
    });
  const bvhIndex = buildTriangleBvh(
    dependencies.THREE,
    loadedTriangles.triangleStore,
    args.MeshBVH || MeshBVH,
    { targetLeafSize: args.bvhTargetLeafSize },
  );
  const geometryStats = {
    ...loadedTriangles.totals,
    ...bvhIndex.memoryStats,
  };
  const progressPath = path.join(args.outputDir, '.external_hit_probe_progress.json');
  const completed = args.resume ? readProgress(progressPath, selectedUnitIds, manifest) : new Set();
  const resumedFromProgress = completed.size > 0;
  const descriptors = openColumnOutputs(args.outputDir, manifest, args.resume);
  try {
    for (let localUnitIndex = 0; localUnitIndex < selectedUnits.length; localUnitIndex += 1) {
      const unit = selectedUnits[localUnitIndex];
      if (completed.has(unit.componentGlobalId)) continue;
      const rows = traceUnitRays({
        THREE: dependencies.THREE,
        bvhIndex,
        unitId: unit.componentGlobalId,
        starts: startsByUnit.get(unit.componentGlobalId),
        radius: unit.bounds.radius,
      });
      writeUnitColumns(descriptors, localUnitIndex, unit.componentGlobalId, rows);
      completed.add(unit.componentGlobalId);
      writeProgress(progressPath, selectedUnitIds, completed, manifest);
      if (args.progressEvery > 0 && (localUnitIndex + 1) % args.progressEvery === 0) {
        console.log(`[v5-external-hit] ${localUnitIndex + 1}/${selectedUnits.length} units traced`);
      }
    }
  } finally {
    closeColumnOutputs(descriptors);
    bvhIndex.geometry.dispose();
  }
  assert(completed.size === selectedUnits.length, 'external-hit probe generation did not complete all selected units');
  writeJsonAtomically(args.manifestPath, manifest);
  fs.rmSync(progressPath, { force: true });
  return {
    outputDir: args.outputDir,
    manifestPath: args.manifestPath,
    manifest,
    resumed: resumedFromProgress,
    geometryStats,
  };
}

function readManifestForDirectory(directory) {
  const candidates = [
    path.join(directory, 'external_hit_probe_manifest.json'),
    path.join(directory, 'manifest.json'),
  ];
  const manifestPath = candidates.find((candidate) => fs.existsSync(candidate));
  assert(manifestPath, `probe manifest is missing from ${directory}`);
  const manifest = readJson(manifestPath);
  validateProbeManifest(manifest);
  return { manifest, manifestPath };
}

function readShardUnitIds(directory, manifest) {
  const filePath = path.resolve(directory, manifest.files.unitIds);
  const bytes = fs.readFileSync(filePath);
  return decodeColumn('unitIds', bytes, manifest.rowCount);
}

/** Merge complete shard directories into one deterministic global columnar asset. */
export function mergeExternalHitProbeShards({ shardDirs, outputDir, manifestPath, resume = true }) {
  assert(Array.isArray(shardDirs) && shardDirs.length > 0, 'shardDirs must be a non-empty array');
  const shards = shardDirs.map((directory) => ({
    directory: path.resolve(directory),
    ...readManifestForDirectory(path.resolve(directory)),
  }));
  const first = shards[0].manifest;
  for (const shard of shards) {
    const manifest = shard.manifest;
    assert(manifest.schema === first.schema && manifest.sceneId === first.sceneId, 'shards disagree on scene');
    assert(manifest.distanceRatios.join(',') === first.distanceRatios.join(','), 'shards disagree on distance grid');
    assert(manifest.storage === first.storage, 'shards disagree on storage');
    assert(manifest.rowCount === manifest.numUnits * RAYS_PER_UNIT, 'shard rowCount is invalid');
  }
  const unitBlocks = new Map();
  const unitIdsInOrder = [];
  for (const shard of shards) {
    const ids = readShardUnitIds(shard.directory, shard.manifest);
    for (let localUnitIndex = 0; localUnitIndex < shard.manifest.numUnits; localUnitIndex += 1) {
      const rowStart = localUnitIndex * RAYS_PER_UNIT;
      const unitId = ids[rowStart];
      for (let row = rowStart; row < rowStart + RAYS_PER_UNIT; row += 1) {
        assert(ids[row] === unitId, `shard unit column is not contiguous for unit ${unitId}`);
      }
      assert(!unitBlocks.has(unitId), `duplicate unit ${unitId} across shards`);
      unitBlocks.set(unitId, { shard, localUnitIndex });
      unitIdsInOrder.push(unitId);
    }
  }
  unitIdsInOrder.sort((left, right) => left - right);
  assert(unitIdsInOrder.length > 0, 'shards contain no units');
  if (first.sceneNumUnits !== undefined) {
    assert(unitIdsInOrder.length === first.sceneNumUnits, 'shards do not cover every scene unit');
  }
  const targetDir = path.resolve(outputDir);
  const targetManifestPath = path.resolve(manifestPath || path.join(targetDir, 'external_hit_probe_manifest.json'));
  const targetManifest = makeProbeManifest({
    sceneId: first.sceneId,
    unitIds: unitIdsInOrder,
    totalUnitCount: first.sceneNumUnits || unitIdsInOrder.length,
    outputDir: targetDir,
    shardIndex: null,
    shardCount: null,
  });
  if (resume && fs.existsSync(targetManifestPath)) {
    const existing = readJson(targetManifestPath);
    if (manifestIsComplete(targetDir, existing)
      && sameStringArray((existing.unitIds || []).map(String), targetManifest.unitIds.map(String))) {
      return { outputDir: targetDir, manifestPath: targetManifestPath, manifest: existing, resumed: true };
    }
  }
  fs.mkdirSync(targetDir, { recursive: true });
  const descriptors = openColumnOutputs(targetDir, targetManifest, false);
  try {
    for (let targetUnitIndex = 0; targetUnitIndex < unitIdsInOrder.length; targetUnitIndex += 1) {
      const unitId = unitIdsInOrder[targetUnitIndex];
      const source = unitBlocks.get(unitId);
      for (const name of Object.keys(COLUMN_FILES)) {
        const sourcePath = path.resolve(source.shard.directory, source.shard.manifest.files[name]);
        const bytesPerRow = name === 'directions' ? 12 : (name === 'unitIds' || name === 'hitDistances' || name === 'maxDistances' ? 4 : 1);
        const byteOffset = source.localUnitIndex * RAYS_PER_UNIT * bytesPerRow;
        const byteLength = RAYS_PER_UNIT * bytesPerRow;
        const sourceFd = fs.openSync(sourcePath, 'r');
        const buffer = Buffer.alloc(byteLength);
        try {
          fs.readSync(sourceFd, buffer, 0, byteLength, byteOffset);
        } finally {
          fs.closeSync(sourceFd);
        }
        const targetOffset = targetUnitIndex * RAYS_PER_UNIT * bytesPerRow;
        fs.writeSync(descriptors[name].fd, buffer, 0, buffer.length, targetOffset);
      }
    }
  } finally {
    closeColumnOutputs(descriptors);
  }
  writeJsonAtomically(targetManifestPath, targetManifest);
  return { outputDir: targetDir, manifestPath: targetManifestPath, manifest: targetManifest, resumed: false };
}

/** Read all six columns using the manifest's fixed little-endian contract. */
export function readColumnarProbeAsset(outputDir, manifest = readManifestForDirectory(outputDir).manifest) {
  validateProbeManifest(manifest);
  const values = {};
  for (const [name, relative] of Object.entries(manifest.files)) {
    values[name] = decodeColumn(
      name,
      fs.readFileSync(path.resolve(outputDir, relative)),
      manifest.rowCount,
    );
  }
  return { manifest, values };
}

export function parseArgs(argv = process.argv) {
  const args = {
    assetsDir: DEFAULT_ASSETS_DIR,
    runtimeMetaPath: DEFAULT_RUNTIME_META,
    glbIndexPath: DEFAULT_GLB_INDEX,
    surfaceManifestPath: DEFAULT_SURFACE_MANIFEST,
    outputDir: DEFAULT_OUTPUT_DIR,
    manifestPath: null,
    sceneId: '',
    shardIndex: null,
    shardCount: null,
    resume: true,
    progressEvery: 50,
    gcEvery: 0,
    triangleChunkCapacity: 262144,
    bvhTargetLeafSize: 10,
    help: false,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === '--help' || key === '-h') {
      args.help = true;
      continue;
    }
    if (key === '--resume') {
      args.resume = true;
      continue;
    }
    if (key === '--no-resume') {
      args.resume = false;
      continue;
    }
    assert(key.startsWith('--'), `unknown argument ${key}`);
    const value = argv[index + 1];
    assert(value !== undefined && !value.startsWith('--'), `${key} requires a value`);
    index += 1;
    if (key === '--assets-dir') args.assetsDir = path.resolve(value);
    else if (key === '--runtime-meta') args.runtimeMetaPath = path.resolve(value);
    else if (key === '--glb-index') args.glbIndexPath = path.resolve(value);
    else if (key === '--surface-manifest' || key === '--surface-meta') args.surfaceManifestPath = path.resolve(value);
    else if (key === '--surface-points') args.surfacePointsPath = path.resolve(value);
    else if (key === '--output-dir' || key === '--output') args.outputDir = path.resolve(value);
    else if (key === '--manifest') args.manifestPath = path.resolve(value);
    else if (key === '--scene-id') args.sceneId = value;
    else if (key === '--shard-index') args.shardIndex = asNonNegativeInteger(value, 'shard-index');
    else if (key === '--shard-count') args.shardCount = asPositiveInteger(value, 'shard-count');
    else if (key === '--progress-every') args.progressEvery = asNonNegativeInteger(value, 'progress-every');
    else if (key === '--gc-every') args.gcEvery = asNonNegativeInteger(value, 'gc-every');
    else if (key === '--triangle-chunk-capacity') args.triangleChunkCapacity = asPositiveInteger(value, 'triangle-chunk-capacity');
    else if (key === '--bvh-target-leaf-size') args.bvhTargetLeafSize = asPositiveInteger(value, 'bvh-target-leaf-size');
    else throw new Error(`unknown argument ${key}`);
  }
  return args;
}

function printHelp() {
  console.log(`Usage: node generate_external_hit_probes.mjs [options]

Options:
  --assets-dir DIR          Scene asset root containing indexed GLBs.
  --runtime-meta FILE       runtimeVisibilityMeta.json path.
  --glb-index FILE          glbIndex.json path.
  --surface-manifest FILE   Existing 256-point local-surface manifest.
  --surface-points FILE     Override the points binary named by the manifest.
  --output-dir DIR          Columnar output directory.
  --manifest FILE           Output manifest path.
  --scene-id ID             Source scene identifier.
  --shard-index N           Zero-based contiguous shard index.
  --shard-count N           Number of shards.
  --resume / --no-resume    Resume from the progress sidecar (default on).
  --progress-every N        Progress interval in GLBs/units; zero disables it.
  --gc-every N              Call exposed V8 GC every N decoded GLBs (default 0).
  --triangle-chunk-capacity N  Triangles per compact typed-array chunk (default 262144).
  --bvh-target-leaf-size N  Target triangles per BVH leaf (default 10).
`);
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  try {
    const args = parseArgs(process.argv);
    if (args.help) {
      printHelp();
    } else {
      const result = await generateExternalHitProbes(args);
      console.log(JSON.stringify({
        schema: result.manifest.schema,
        outputDir: result.outputDir,
        manifest: result.manifestPath,
        numUnits: result.manifest.numUnits,
        rowCount: result.manifest.rowCount,
        resumed: result.resumed,
        geometryStats: result.geometryStats,
      }, null, 2));
    }
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  }
}
