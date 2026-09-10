#!/usr/bin/env node
/*
 * Export the geometry-shell HZB startup asset.
 *
 * The exporter deliberately consumes the scene package's glbIndex.json and
 * runtimeVisibilityMeta.json instead of guessing component IDs from file
 * names.  It decodes source POSITION/INDEX/EXT_mesh_gpu_instancing data and
 * writes only complete, certain-opaque primitives to the shell streams.
 */

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { MeshoptDecoder } from 'meshoptimizer/decoder';
import { MeshoptEncoder } from 'meshoptimizer/encoder';

export const SHELL_SCHEMA = 'geometry-shell-hzb-v2';
export const SHELL_VARIANTS = new Set(['lossless', 'equal-asset']);
export const OFFLINE_REPORT_SCHEMA = 'geometry-shell-hzb-offline-report-v1';
const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const JSON_CHUNK_TYPE = 0x4e4f534a;
const BIN_CHUNK_TYPE = 0x004e4942;
const TRIANGLES_MODE = 4;
const COMPONENT_TYPES = {
  5120: { bytes: 1, signed: true },
  5121: { bytes: 1, signed: false },
  5122: { bytes: 2, signed: true },
  5123: { bytes: 2, signed: false },
  5125: { bytes: 4, signed: false },
  5126: { bytes: 4, signed: true, float: true },
};
const TYPE_COMPONENTS = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};
const TRANSPARENCY_EXTENSIONS = new Set([
  'KHR_materials_transmission',
  'KHR_materials_volume',
  'KHR_materials_diffuse_transmission',
  'KHR_materials_dispersion',
]);
const KNOWN_OPAQUE_EXTENSIONS = new Set([
  'KHR_materials_anisotropy',
  'KHR_materials_clearcoat',
  'KHR_materials_emissive_strength',
  'KHR_materials_iridescence',
  'KHR_materials_ior',
  'KHR_materials_specular',
  'KHR_materials_unlit',
]);

function finite(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${name} must be finite.`);
  return number;
}

function positiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 0) throw new Error(`${name} must be a non-negative integer.`);
  return number;
}

function pad4(value) {
  return (value + 3) & ~3;
}

function identityMatrix() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function multiplyMatrix(a, b) {
  const out = new Array(16).fill(0);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      for (let k = 0; k < 4; k += 1) {
        out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
      }
    }
  }
  return out;
}

function matrixFromNode(node) {
  if (Array.isArray(node.matrix) && node.matrix.length === 16) {
    return node.matrix.map((value, index) => finite(value, `node.matrix[${index}]`));
  }
  const translation = node.translation || [0, 0, 0];
  const rotation = node.rotation || [0, 0, 0, 1];
  const scale = node.scale || [1, 1, 1];
  const [x, y, z, w] = rotation.map((value, index) => finite(value, `node.rotation[${index}]`));
  const [sx, sy, sz] = scale.map((value, index) => finite(value, `node.scale[${index}]`));
  const [tx, ty, tz] = translation.map((value, index) => finite(value, `node.translation[${index}]`));
  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;
  return [
    (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
    (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
    (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

function matrixFromTrs(translation, rotation, scale) {
  return matrixFromNode({ translation, rotation, scale });
}

function transformPoint(matrix, point) {
  return [
    matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
    matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
    matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
  ];
}

function parseGlb(filePath) {
  const bytes = fs.readFileSync(filePath);
  if (bytes.length < 20 || bytes.readUInt32LE(0) !== GLB_MAGIC) {
    throw new Error(`${filePath}: invalid GLB magic.`);
  }
  if (bytes.readUInt32LE(4) !== GLB_VERSION || bytes.readUInt32LE(8) !== bytes.length) {
    throw new Error(`${filePath}: unsupported GLB version or length.`);
  }
  let offset = 12;
  let json = null;
  let binary = null;
  while (offset + 8 <= bytes.length) {
    const length = bytes.readUInt32LE(offset);
    const type = bytes.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = start + length;
    if (end > bytes.length) throw new Error(`${filePath}: truncated GLB chunk.`);
    if (type === JSON_CHUNK_TYPE) json = JSON.parse(bytes.toString('utf8', start, end).trim());
    else if (type === BIN_CHUNK_TYPE) binary = bytes.subarray(start, end);
    offset = end;
  }
  if (!json) throw new Error(`${filePath}: JSON chunk is missing.`);
  return { filePath, directory: path.dirname(filePath), json, binary };
}

function dataUriBytes(uri) {
  const match = /^data:.*?;base64,(.*)$/s.exec(uri);
  if (!match) throw new Error('only base64 data URIs are supported for GLB buffers.');
  return Buffer.from(match[1], 'base64');
}

function bufferBytes(parsed, bufferIndex) {
  const index = Number(bufferIndex || 0);
  if (index === 0 && parsed.binary) return parsed.binary;
  const descriptor = parsed.json.buffers?.[index];
  if (!descriptor) throw new Error(`${parsed.filePath}: buffer ${index} is missing.`);
  if (!descriptor.uri) {
    throw new Error(`${parsed.filePath}: buffer ${index} has no GLB BIN chunk or URI.`);
  }
  if (descriptor.uri.startsWith('data:')) return dataUriBytes(descriptor.uri);
  const external = path.resolve(parsed.directory, decodeURIComponent(descriptor.uri));
  if (!external.startsWith(`${parsed.directory}${path.sep}`)) {
    throw new Error(`${parsed.filePath}: external buffer escapes its directory.`);
  }
  return fs.readFileSync(external);
}

function accessorLayout(parsed, accessorIndex) {
  const accessor = parsed.json.accessors?.[Number(accessorIndex)];
  if (!accessor) throw new Error(`${parsed.filePath}: accessor ${accessorIndex} is missing.`);
  const view = parsed.json.bufferViews?.[Number(accessor.bufferView)];
  if (!view) throw new Error(`${parsed.filePath}: bufferView for accessor ${accessorIndex} is missing.`);
  if (accessor.sparse) throw new Error(`${parsed.filePath}: sparse accessors are not supported.`);
  const component = COMPONENT_TYPES[accessor.componentType];
  const components = TYPE_COMPONENTS[accessor.type];
  if (!component || !components) throw new Error(`${parsed.filePath}: unsupported accessor ${accessorIndex} format.`);
  const elementBytes = component.bytes * components;
  return {
    accessor,
    view,
    count: positiveInteger(accessor.count, `accessor ${accessorIndex} count`),
    components,
    component,
    stride: Number(view.byteStride || elementBytes),
    elementBytes,
  };
}

async function accessorBytes(parsed, accessorIndex) {
  const layout = accessorLayout(parsed, accessorIndex);
  const compression = layout.view.extensions?.EXT_meshopt_compression;
  if (compression) {
    await MeshoptDecoder.ready;
    const compressedBuffer = bufferBytes(parsed, compression.buffer ?? 0);
    const start = Number(compression.byteOffset || 0);
    const end = start + Number(compression.byteLength);
    if (start < 0 || end > compressedBuffer.length) {
      throw new Error(`${parsed.filePath}: meshopt data for accessor ${accessorIndex} is out of range.`);
    }
    const stride = Number(compression.byteStride);
    const target = new Uint8Array(layout.count * stride);
    MeshoptDecoder.decodeGltfBuffer(
      target,
      layout.count,
      stride,
      new Uint8Array(compressedBuffer.subarray(start, end)),
      compression.mode,
      compression.filter,
    );
    return { ...layout, bytes: target, byteOffset: Number(layout.accessor.byteOffset || 0), stride };
  }
  const source = bufferBytes(parsed, layout.view.buffer ?? 0);
  const byteOffset = Number(layout.view.byteOffset || 0) + Number(layout.accessor.byteOffset || 0);
  const byteLength = layout.count * layout.stride;
  if (byteOffset < 0 || byteOffset + byteLength > source.length) {
    throw new Error(`${parsed.filePath}: accessor ${accessorIndex} is out of range.`);
  }
  return { ...layout, bytes: source, byteOffset, stride: layout.stride };
}

function readComponent(view, offset, componentType) {
  switch (componentType) {
    case 5120: return view.getInt8(offset);
    case 5121: return view.getUint8(offset);
    case 5122: return view.getInt16(offset, true);
    case 5123: return view.getUint16(offset, true);
    case 5125: return view.getUint32(offset, true);
    case 5126: return view.getFloat32(offset, true);
    default: throw new Error(`unsupported component type ${componentType}`);
  }
}

function normalizedComponent(value, componentType) {
  if (componentType === 5120) return Math.max(-1, value / 127);
  if (componentType === 5121) return value / 255;
  if (componentType === 5122) return Math.max(-1, value / 32767);
  if (componentType === 5123) return value / 65535;
  return value;
}

async function readAccessor(parsed, accessorIndex) {
  const layout = await accessorBytes(parsed, accessorIndex);
  const view = new DataView(layout.bytes.buffer, layout.bytes.byteOffset, layout.bytes.byteLength);
  const values = new Float32Array(layout.count * layout.components);
  const componentBytes = layout.component.bytes;
  for (let index = 0; index < layout.count; index += 1) {
    const elementOffset = layout.byteOffset + index * layout.stride;
    for (let component = 0; component < layout.components; component += 1) {
      const raw = readComponent(view, elementOffset + component * componentBytes, layout.accessor.componentType);
      values[index * layout.components + component] = layout.accessor.normalized
        ? normalizedComponent(raw, layout.accessor.componentType)
        : raw;
    }
  }
  return { ...layout, values };
}

async function readIndices(parsed, accessorIndex) {
  const result = await readAccessor(parsed, accessorIndex);
  if (result.components !== 1 || result.accessor.type !== 'SCALAR') {
    throw new Error(`${parsed.filePath}: index accessor must be SCALAR.`);
  }
  const indices = Uint32Array.from(result.values, (value) => {
    if (!Number.isInteger(value) || value < 0) throw new Error(`${parsed.filePath}: index value is invalid.`);
    return value;
  });
  if (indices.length % 3 !== 0) throw new Error(`${parsed.filePath}: index count is not divisible by three.`);
  return indices;
}

function meshNodes(parsed) {
  const sceneIndex = Number(parsed.json.scene ?? 0);
  const scene = parsed.json.scenes?.[sceneIndex];
  if (!scene) throw new Error(`${parsed.filePath}: default scene is missing.`);
  const result = [];
  const visiting = new Set();
  const visit = (nodeIndex, parentMatrix) => {
    const index = Number(nodeIndex);
    if (visiting.has(index)) throw new Error(`${parsed.filePath}: node graph contains a cycle.`);
    const node = parsed.json.nodes?.[index];
    if (!node) throw new Error(`${parsed.filePath}: node ${index} is missing.`);
    visiting.add(index);
    const worldMatrix = multiplyMatrix(parentMatrix, matrixFromNode(node));
    if (node.mesh != null) result.push({ node, nodeIndex: index, meshIndex: Number(node.mesh), worldMatrix });
    for (const child of node.children || []) visit(child, worldMatrix);
    visiting.delete(index);
  };
  for (const root of scene.nodes || []) visit(root, identityMatrix());
  return result;
}

async function nodeInstances(parsed, meshNode) {
  const extension = meshNode.node.extensions?.EXT_mesh_gpu_instancing;
  if (!extension) return new Float32Array(meshNode.worldMatrix);
  const attributes = extension.attributes || {};
  const firstAccessor = Object.values(attributes)[0];
  if (firstAccessor == null) throw new Error(`${parsed.filePath}: EXT_mesh_gpu_instancing has no attributes.`);
  const first = await readAccessor(parsed, firstAccessor);
  const count = first.count;
  const getAttribute = async (name, defaults, components) => {
    if (attributes[name] == null) return { count, values: Float32Array.from({ length: count * components }, (_, index) => defaults[index % components]) };
    const value = await readAccessor(parsed, attributes[name]);
    if (value.count !== count || value.components !== components) {
      throw new Error(`${parsed.filePath}: instancing attribute ${name} has an inconsistent shape.`);
    }
    return value;
  };
  const translations = await getAttribute('TRANSLATION', [0, 0, 0], 3);
  const rotations = await getAttribute('ROTATION', [0, 0, 0, 1], 4);
  const scales = await getAttribute('SCALE', [1, 1, 1], 3);
  const matrices = new Float32Array(count * 16);
  for (let index = 0; index < count; index += 1) {
    const translation = translations.values.slice(index * 3, index * 3 + 3);
    const rotation = rotations.values.slice(index * 4, index * 4 + 4);
    const scale = scales.values.slice(index * 3, index * 3 + 3);
    const matrix = multiplyMatrix(meshNode.worldMatrix, matrixFromTrs(translation, rotation, scale));
    matrices.set(matrix, index * 16);
  }
  return matrices;
}

export function classifyMaterial(material) {
  if (!material) return { occluder: false, reason: 'missing-material' };
  const alphaMode = String(material.alphaMode || 'OPAQUE').toUpperCase();
  if (alphaMode !== 'OPAQUE') return { occluder: false, reason: `alpha-mode-${alphaMode.toLowerCase()}` };
  const alpha = material.pbrMetallicRoughness?.baseColorFactor?.[3];
  if (alpha != null && (!Number.isFinite(Number(alpha)) || Number(alpha) < 1 - 1e-6)) {
    return { occluder: false, reason: 'non-opaque-base-color-alpha' };
  }
  const extensions = Object.keys(material.extensions || {});
  const transparency = extensions.find((name) => TRANSPARENCY_EXTENSIONS.has(name));
  if (transparency) return { occluder: false, reason: `transparency-extension-${transparency}` };
  const unknown = extensions.find((name) => !KNOWN_OPAQUE_EXTENSIONS.has(name));
  if (unknown) return { occluder: false, reason: `unknown-material-extension-${unknown}` };
  return { occluder: true, reason: null };
}

async function readPrimitive(parsed, meshNode, primitive, primitiveIndex) {
  if (Number(primitive.mode ?? TRIANGLES_MODE) !== TRIANGLES_MODE) {
    return { supported: false, reason: 'non-triangle-primitive', primitiveIndex };
  }
  if (primitive.attributes?.POSITION == null || primitive.indices == null) {
    return { supported: false, reason: 'missing-position-or-index', primitiveIndex };
  }
  const position = await readAccessor(parsed, primitive.attributes.POSITION);
  if (position.accessor.type !== 'VEC3') return { supported: false, reason: 'position-is-not-vec3', primitiveIndex };
  const indices = await readIndices(parsed, primitive.indices);
  const vertexCount = position.count;
  for (const index of indices) {
    if (index >= vertexCount) throw new Error(`${parsed.filePath}: primitive ${primitiveIndex} has an out-of-range index.`);
  }
  const material = classifyMaterial(parsed.json.materials?.[primitive.material]);
  const bounds = {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
  };
  for (let index = 0; index < position.values.length; index += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      bounds.min[axis] = Math.min(bounds.min[axis], position.values[index + axis]);
      bounds.max[axis] = Math.max(bounds.max[axis], position.values[index + axis]);
    }
  }
  return {
    supported: true,
    primitiveIndex,
    positions: position.values,
    indices,
    vertexCount,
    triangleCount: indices.length / 3,
    material,
    bounds,
    node: meshNode,
  };
}

async function inspectSourceEntry(entry, assetsDir, expectedInstanceCount) {
  const sourcePath = path.resolve(assetsDir, entry.path);
  if (!sourcePath.startsWith(`${path.resolve(assetsDir)}${path.sep}`) || !fs.existsSync(sourcePath)) {
    throw new Error(`GLB entry ${entry.globalId} points to a missing or escaping path: ${entry.path}`);
  }
  const parsed = parseGlb(sourcePath);
  const nodes = meshNodes(parsed);
  const instancesByNode = [];
  let instanceCount = 0;
  for (const node of nodes) {
    const matrices = await nodeInstances(parsed, node);
    const nodeInstanceCount = matrices.length / 16;
    instancesByNode.push({ node, matrices });
    instanceCount += nodeInstanceCount;
  }
  const declaredBuffers = parsed.json.buffers || [];
  const emptyPlaceholder = Boolean(
    expectedInstanceCount
      && instanceCount === 0
      && nodes.length === 0
      && (!parsed.json.meshes || parsed.json.meshes.length === 0)
      && declaredBuffers.every((buffer) => Number(buffer.byteLength || 0) === 0)
      && (!parsed.binary || parsed.binary.length === 0),
  );
  if (expectedInstanceCount != null && expectedInstanceCount !== instanceCount && !emptyPlaceholder) {
    throw new Error(
      `${entry.path}: source instance count ${instanceCount} does not match runtime metadata ${expectedInstanceCount}.`,
    );
  }
  const primitives = [];
  for (const { node, matrices } of instancesByNode) {
    const mesh = parsed.json.meshes?.[node.meshIndex];
    if (!mesh) throw new Error(`${entry.path}: mesh ${node.meshIndex} is missing.`);
    for (const primitive of mesh.primitives || []) {
      const inspectedPrimitive = await readPrimitive(parsed, node, primitive, primitives.length);
      inspectedPrimitive.instanceMatrices = matrices;
      primitives.push(inspectedPrimitive);
    }
  }
  return {
    entry,
    sourcePath,
    sourceBytes: fs.statSync(sourcePath).size,
    instancesByNode,
    instanceCount,
    expectedInstanceCount,
    emptyPlaceholder,
    primitives,
  };
}

function boundsMinMax(bounds) {
  if (bounds?.min && bounds?.max) return [...bounds.min, ...bounds.max].map(Number);
  if (bounds?.center && bounds?.size) {
    const center = bounds.center.map(Number);
    const size = bounds.size.map(Number);
    return [
      center[0] - size[0] * 0.5, center[1] - size[1] * 0.5, center[2] - size[2] * 0.5,
      center[0] + size[0] * 0.5, center[1] + size[1] * 0.5, center[2] + size[2] * 0.5,
    ];
  }
  throw new Error('runtime component bounds must use min/max or center/size.');
}

function buildRuntimeArrays(runtimeMeta, glbIndex) {
  const instanceCount = positiveInteger(runtimeMeta.instanceCount ?? runtimeMeta.componentCount, 'runtime instanceCount');
  const glbCount = positiveInteger(runtimeMeta.globalGlbCount, 'runtime globalGlbCount');
  if (glbCount !== Number(glbIndex.total) || glbIndex.entries.length !== glbCount) {
    throw new Error('glbIndex and runtimeVisibilityMeta GLB counts disagree.');
  }
  const records = runtimeMeta.componentRecords || [];
  if (records.length !== instanceCount) throw new Error('runtime component record count disagrees with instanceCount.');
  const aabbs = new Float32Array(instanceCount * 6);
  const instanceToGlb = new Uint32Array(instanceCount);
  const recordById = new Array(instanceCount);
  for (const record of records) {
    const id = positiveInteger(record.componentGlobalId, 'componentGlobalId');
    if (id >= instanceCount || recordById[id]) throw new Error('runtime component IDs are not unique and dense.');
    const bounds = boundsMinMax(record.bounds);
    if (bounds.some((value) => !Number.isFinite(value))) throw new Error(`component ${id} has non-finite bounds.`);
    aabbs.set(bounds, id * 6);
    const globalGlbId = positiveInteger(record.globalGlbId, 'globalGlbId');
    if (globalGlbId >= glbCount) throw new Error(`component ${id} has an out-of-range globalGlbId.`);
    instanceToGlb[id] = globalGlbId;
    recordById[id] = record;
  }
  if (recordById.some((record) => !record)) throw new Error('runtime component IDs are not dense.');
  return { instanceCount, glbCount, aabbs, instanceToGlb, recordById };
}

function typedBytes(values) {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

class BinaryWriter {
  constructor(filePath) {
    this.filePath = filePath;
    this.fd = fs.openSync(filePath, 'w');
    this.offset = 0;
  }

  write(bytes) {
    const source = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const offset = this.offset;
    let written = 0;
    while (written < source.length) written += fs.writeSync(this.fd, source, written, source.length - written);
    this.offset += source.length;
    return { offset, byteLength: source.length };
  }

  close() {
    fs.closeSync(this.fd);
  }
}

function encodeStream(values, count, stride, mode) {
  const encoded = MeshoptEncoder.encodeGltfBuffer(typedBytes(values), count, stride, mode);
  return new Uint8Array(encoded.buffer, encoded.byteOffset, encoded.byteLength);
}

function writeRaw(filePath, values) {
  fs.writeFileSync(filePath, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
}

function cameraBasis(forward) {
  const length = Math.hypot(...forward);
  const f = forward.map((value) => value / Math.max(length, 1e-8));
  const upSeed = Math.abs(f[1]) > 0.98 ? [0, 0, 1] : [0, 1, 0];
  const right = [
    f[1] * upSeed[2] - f[2] * upSeed[1],
    f[2] * upSeed[0] - f[0] * upSeed[2],
    f[0] * upSeed[1] - f[1] * upSeed[0],
  ];
  const rightLength = Math.hypot(...right);
  for (let axis = 0; axis < 3; axis += 1) right[axis] /= Math.max(rightLength, 1e-8);
  const up = [
    right[1] * f[2] - right[2] * f[1],
    right[2] * f[0] - right[0] * f[2],
    right[0] * f[1] - right[1] * f[0],
  ];
  return { forward: f, right, up };
}

function projectBoundsArea(bounds, matrix, pose) {
  const position = (pose.camera_pos || pose.position || pose.cameraWorld || []).map(Number);
  const forward = (pose.camera_forward || pose.forward || []).map(Number);
  if (position.length !== 3 || forward.length !== 3) return 0;
  const basis = cameraBasis(forward);
  const fovY = Number(pose.fov_y ?? pose.fovYDeg ?? pose.render_fov_y ?? 66);
  const aspect = Number(pose.aspect || 1);
  const tanY = Math.tan((fovY * Math.PI) / 360);
  const tanX = tanY * aspect;
  const points = [];
  for (const x of [bounds.min[0], bounds.max[0]]) {
    for (const y of [bounds.min[1], bounds.max[1]]) {
      for (const z of [bounds.min[2], bounds.max[2]]) {
        const world = transformPoint(matrix, [x, y, z]);
        const relative = [world[0] - position[0], world[1] - position[1], world[2] - position[2]];
        const depth = relative[0] * basis.forward[0] + relative[1] * basis.forward[1] + relative[2] * basis.forward[2];
        const viewX = relative[0] * basis.right[0] + relative[1] * basis.right[1] + relative[2] * basis.right[2];
        const viewY = relative[0] * basis.up[0] + relative[1] * basis.up[1] + relative[2] * basis.up[2];
        if (depth <= 0) return 1;
        points.push([viewX / (depth * tanX), viewY / (depth * tanY)]);
      }
    }
  }
  const rect = [
    Math.min(...points.map((point) => point[0])),
    Math.min(...points.map((point) => point[1])),
    Math.max(...points.map((point) => point[0])),
    Math.max(...points.map((point) => point[1])),
  ];
  const width = Math.max(0, Math.min(1, rect[2]) - Math.max(-1, rect[0]));
  const height = Math.max(0, Math.min(1, rect[3]) - Math.max(-1, rect[1]));
  return width * height;
}

async function readImportancePoses(filePath, requestedCount) {
  const rows = [];
  const input = fs.createReadStream(filePath, 'utf8');
  const lines = readline.createInterface({ input, crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    if (row.split && row.split !== 'train') continue;
    if (row.subpose_id != null && Number(row.subpose_id) !== 0) continue;
    rows.push(row);
    if (rows.length >= requestedCount) break;
  }
  lines.close();
  if (rows.length !== requestedCount) {
    throw new Error(`importance pose plan contains ${rows.length} train center views; expected ${requestedCount}.`);
  }
  return rows;
}

function candidateImportance(candidate, poses) {
  let score = 0;
  for (const matrix of candidate.matrices) score += poses.reduce(
    (sum, pose) => sum + projectBoundsArea(candidate.bounds, matrix, pose),
    0,
  );
  return score;
}

function directoryBytes(root) {
  let total = 0;
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile()) total += fs.statSync(file).size;
    }
  };
  visit(root);
  return total;
}

function isPathInside(child, parent) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
}

function offlineReportPath(args) {
  const report = args.offlineReport
    ? path.resolve(args.offlineReport)
    : path.resolve(`${args.outputDir}.offline.json`);
  if (isPathInside(report, args.outputDir)) {
    throw new Error(`offline report must be outside the runtime asset directory: ${report}`);
  }
  return report;
}

function runtimeAssetBytes(outputDir, files) {
  return files.reduce((sum, file) => sum + fs.statSync(path.join(outputDir, file)).size, 0);
}

function inferSceneName(sceneWeb, runtimeMeta, assetsDir, args) {
  const configured = sceneWeb.config?.sceneName
    || sceneWeb.config?.name
    || runtimeMeta.sceneName;
  if (configured) return String(configured);
  const explicitRoot = args.assetsDir || args.sceneRoot || args.dataRoot || assetsDir;
  const rootName = path.basename(explicitRoot);
  if (rootName && !['assets', 'out'].includes(rootName)) return rootName;
  const parentName = path.basename(path.dirname(assetsDir));
  return parentName || path.basename(assetsDir);
}

function buildShellMeta({
  args,
  assetsDir,
  sceneWeb,
  runtimeMeta,
  glbIndex,
  entries,
  selectedEntries,
  runtime,
  counters,
  streams,
  prototypes,
  binaryBytes,
  selection,
}) {
  const compressedGeometryBytes = binaryBytes[streams.positions.file]
    + binaryBytes[streams.indices.file] + binaryBytes[streams.transforms.file];
  const aabbFile = 'instance_aabb_fp32.bin';
  const mappingFile = 'instance_to_glb_uint32.bin';
  return {
    schema: SHELL_SCHEMA,
    variant: args.variant,
    sceneName: inferSceneName(sceneWeb, runtimeMeta, assetsDir, args),
    source: {
      layout: 'scene package with sceneWeb.json, glbIndex.json and runtimeVisibilityMeta.json',
      assetsDirectoryName: path.basename(assetsDir),
      lod: glbIndex.lod || 'LOD0',
      sceneGlbCount: entries.length,
      scannedGlbCount: selectedEntries.length,
      partialSource: selectedEntries.length !== entries.length,
    },
    geometryPolicy: {
      positions: 'source POSITION values decoded and retained',
      indices: 'source indexed triangles retained without reordering or simplification',
      transforms: 'final node and EXT_mesh_gpu_instancing transforms retained as float32 matrices',
      removedAttributes: ['NORMAL', 'TANGENT', 'TEXCOORD_0', 'TEXCOORD_1', 'COLOR_0', 'material', 'texture'],
      occluderRule: 'only certain OPAQUE primitives; transparent, alpha-cutout, unsupported or uncertain materials are not occluders',
      backfaceCulling: false,
      rasterization: 'opaque geometry contributes positive linear depth; untouched pixels retain camera far depth',
    },
    queryContract: {
      depthEncoding: 'positive_linear_view_depth_meters',
      sampleCount: 1,
      mipDimensions: 'explicit source/target dimensions in uniform buffers; no textureDimensions query',
      hzbReduction: 'max_2x2',
      proof: 'hzbMax + depthBiasM < candidateNear',
      candidateTest: 'all-candidate-conservative-aabb-hzb',
      uncertainty: 'all candidates use conservative projected AABB/HZB tests; non-finite, near-plane crossing, camera-inside-AABB and outside-screen projections are retained',
      point60: 'one current 60-degree camera query',
      region66: 'offline union of independently queried subposes under the 66-degree candidate region contract',
    },
    instanceCount: runtime.instanceCount,
    globalGlbCount: runtime.glbCount,
    prototypeCount: prototypes.length,
    prototypes,
    streams,
    files: {
      instanceAabbs: { file: aabbFile, encoding: 'float32-little-endian', byteLength: binaryBytes[aabbFile] },
      instanceToGlb: { file: mappingFile, encoding: 'uint32-little-endian', byteLength: binaryBytes[mappingFile] },
    },
    stats: {
      ...counters,
      compressedPositionBytes: binaryBytes[streams.positions.file],
      compressedIndexBytes: binaryBytes[streams.indices.file],
      compressedTransformBytes: binaryBytes[streams.transforms.file],
      compressedGeometryBytes,
      runtimePayloadBytes: binaryBytes[aabbFile] + binaryBytes[mappingFile],
      binaryPayloadBytes: binaryBytes[aabbFile] + binaryBytes[mappingFile] + compressedGeometryBytes,
      prototypeCount: prototypes.length,
      prototypeTriangles: prototypes.reduce((sum, prototype) => sum + prototype.triangleCount, 0),
      rasterizedTriangleInstanceCount: prototypes.reduce((sum, prototype) => sum + prototype.triangleCount * prototype.instanceCount, 0),
      excludedByReason: Object.fromEntries(Object.entries(counters.excludedByReason).sort(([a], [b]) => a.localeCompare(b))),
    },
    selection,
    generatedBy: 'geometry_shell_hzb_exporter.mjs',
  };
}

function parseArgs(argv) {
  const args = {
    assetsDir: '',
    sceneRoot: '',
    dataRoot: '',
    outputDir: '',
    variant: 'lossless',
    budgetBytes: 0,
    budgetFromDir: '',
    importancePosePlan: '',
    importancePoseCount: 128,
    offlineReport: '',
    maxGlbs: 0,
    progressEvery: 100,
    overwrite: false,
  };
  const valueOptions = new Set([
    'assets-dir', 'scene-root', 'data-root', 'output-dir', 'variant', 'budget-bytes',
    'budget-from-dir', 'importance-pose-plan', 'importance-pose-count', 'offline-report',
    'max-glbs', 'progress-every',
  ]);
  for (let index = 2; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--overwrite') {
      args.overwrite = true;
      continue;
    }
    const equal = token.indexOf('=');
    const key = token.slice(0, equal >= 0 ? equal : undefined).replace(/^--/, '');
    if (!valueOptions.has(key)) throw new Error(`Unknown argument: ${token}`);
    const value = equal >= 0 ? token.slice(equal + 1) : argv[++index];
    if (value == null) throw new Error(`Missing value for --${key}`);
    if (key === 'assets-dir') args.assetsDir = path.resolve(value);
    else if (key === 'scene-root') args.sceneRoot = path.resolve(value);
    else if (key === 'data-root') args.dataRoot = path.resolve(value);
    else if (key === 'output-dir') args.outputDir = path.resolve(value);
    else if (key === 'variant') args.variant = String(value);
    else if (key === 'budget-bytes') args.budgetBytes = Number(value);
    else if (key === 'budget-from-dir') args.budgetFromDir = path.resolve(value);
    else if (key === 'importance-pose-plan') args.importancePosePlan = path.resolve(value);
    else if (key === 'importance-pose-count') args.importancePoseCount = Number(value);
    else if (key === 'offline-report') args.offlineReport = path.resolve(value);
    else if (key === 'max-glbs') args.maxGlbs = Number(value);
    else if (key === 'progress-every') args.progressEvery = Number(value);
  }
  const roots = [args.assetsDir, args.sceneRoot, args.dataRoot].filter(Boolean);
  if (roots.length !== 1) throw new Error('Pass exactly one explicit --assets-dir, --scene-root, or --data-root.');
  if (!args.outputDir) throw new Error('--output-dir is required.');
  offlineReportPath(args);
  if (!SHELL_VARIANTS.has(args.variant)) throw new Error(`Unsupported --variant ${args.variant}.`);
  if (args.variant === 'equal-asset') {
    if (!(args.budgetBytes > 0) && !args.budgetFromDir) throw new Error('equal-asset requires --budget-bytes or --budget-from-dir.');
    if (!args.importancePosePlan) throw new Error('equal-asset requires --importance-pose-plan.');
  }
  if (!Number.isInteger(args.importancePoseCount) || args.importancePoseCount <= 0) {
    throw new Error('--importance-pose-count must be positive.');
  }
  if (!Number.isInteger(args.maxGlbs) || args.maxGlbs < 0) throw new Error('--max-glbs must be non-negative.');
  if (!Number.isInteger(args.progressEvery) || args.progressEvery < 0) throw new Error('--progress-every must be non-negative.');
  return args;
}

function resolveAssetsDir(args) {
  const root = args.assetsDir || args.sceneRoot || args.dataRoot;
  const assets = path.basename(root) === 'assets' ? root : path.join(root, 'assets');
  const resolved = fs.existsSync(path.join(root, 'sceneWeb.json')) ? root : assets;
  if (!fs.existsSync(resolved)) throw new Error(`scene/data root does not contain an assets directory: ${root}`);
  for (const name of ['sceneWeb.json', 'glbIndex.json', 'runtimeVisibilityMeta.json']) {
    if (!fs.existsSync(path.join(resolved, name))) throw new Error(`missing ${name} under ${resolved}`);
  }
  return path.resolve(resolved);
}

function prepareOutput(outputDir, overwrite) {
  if (fs.existsSync(outputDir)) {
    if (!overwrite) throw new Error(`output exists; pass --overwrite: ${outputDir}`);
    fs.rmSync(outputDir, { recursive: true, force: true });
  }
  fs.mkdirSync(outputDir, { recursive: true });
}

function createStreamDescriptors() {
  return {
    positions: { file: 'positions.meshopt.bin', encoding: 'meshopt', mode: 'ATTRIBUTES', stride: 12, segments: [] },
    indices: { file: 'indices.meshopt.bin', encoding: 'meshopt', mode: 'TRIANGLES', stride: 4, segments: [] },
    transforms: { file: 'transforms.meshopt.bin', encoding: 'meshopt', mode: 'ATTRIBUTES', stride: 64, segments: [] },
  };
}

function segmentDescriptor(segment, encoded) {
  return {
    offset: encoded.offset,
    byteLength: encoded.byteLength,
    decodedByteLength: segment.count * segment.stride,
    count: segment.count,
    stride: segment.stride,
    mode: segment.mode,
  };
}

function writeCandidateSegments(candidate, stageDir) {
  const base = path.join(stageDir, `prototype_${candidate.prototypeOrdinal}`);
  fs.mkdirSync(base, { recursive: true });
  fs.writeFileSync(path.join(base, 'positions.bin'), Buffer.from(candidate.positionEncoded));
  fs.writeFileSync(path.join(base, 'indices.bin'), Buffer.from(candidate.indexEncoded));
  fs.writeFileSync(path.join(base, 'transforms.bin'), Buffer.from(candidate.transformEncoded));
  return base;
}

function readSegment(filePath) {
  return new Uint8Array(fs.readFileSync(filePath));
}

function layoutCandidate(candidate, streams, offsets, ranges) {
  const positionSegment = {
    count: candidate.vertexCount,
    stride: 12,
    mode: 'ATTRIBUTES',
  };
  const indexSegment = {
    count: candidate.indexCount,
    stride: 4,
    mode: 'TRIANGLES',
  };
  const transformSegment = {
    count: candidate.matrices.length / 16,
    stride: 64,
    mode: 'ATTRIBUTES',
  };
  const positionRange = ranges.position;
  const indexRange = ranges.index;
  const transformRange = ranges.transform;
  const positionDescriptor = segmentDescriptor(positionSegment, positionRange);
  const indexDescriptor = segmentDescriptor(indexSegment, indexRange);
  const transformDescriptor = segmentDescriptor(transformSegment, transformRange);
  streams.positions.segments.push(positionDescriptor);
  streams.indices.segments.push(indexDescriptor);
  streams.transforms.segments.push(transformDescriptor);
  const prototype = {
    prototypeId: offsets.prototypeCount,
    sourceGlobalGlbId: candidate.sourceGlobalGlbId,
    sourcePrimitiveIndex: candidate.sourcePrimitiveIndex,
    vertexOffset: offsets.vertexOffset,
    vertexCount: candidate.vertexCount,
    indexOffset: offsets.indexOffset,
    indexCount: candidate.indexCount,
    triangleCount: candidate.triangleCount,
    instanceOffset: offsets.instanceOffset,
    instanceCount: candidate.matrices.length / 16,
    positionSegment: streams.positions.segments.length - 1,
    indexSegment: streams.indices.segments.length - 1,
    transformSegment: streams.transforms.segments.length - 1,
  };
  offsets.prototypeCount += 1;
  offsets.vertexOffset += candidate.vertexCount;
  offsets.indexOffset += candidate.indexCount;
  offsets.instanceOffset += candidate.matrices.length / 16;
  offsets.positionByteOffset += positionRange.byteLength;
  offsets.indexByteOffset += indexRange.byteLength;
  offsets.transformByteOffset += transformRange.byteLength;
  return {
    prototype,
    encodedBytes: positionRange.byteLength + indexRange.byteLength
      + transformRange.byteLength,
  };
}

function candidateEncodedBytes(candidate) {
  return {
    position: candidate.positionEncoded || readSegment(path.join(candidate.stageDir, 'positions.bin')),
    index: candidate.indexEncoded || readSegment(path.join(candidate.stageDir, 'indices.bin')),
    transform: candidate.transformEncoded || readSegment(path.join(candidate.stageDir, 'transforms.bin')),
  };
}

function appendCandidate(candidate, writers, streams, offsets) {
  const encoded = candidateEncodedBytes(candidate);
  const emitted = layoutCandidate(candidate, streams, offsets, {
    position: writers.positions.write(encoded.position),
    index: writers.indices.write(encoded.index),
    transform: writers.transforms.write(encoded.transform),
  });
  return emitted;
}

function projectCandidates(candidates) {
  const streams = createStreamDescriptors();
  const offsets = {
    prototypeCount: 0,
    vertexOffset: 0,
    indexOffset: 0,
    instanceOffset: 0,
    positionByteOffset: 0,
    indexByteOffset: 0,
    transformByteOffset: 0,
  };
  const prototypes = [];
  for (const candidate of candidates) {
    const encoded = candidateEncodedBytes(candidate);
    const emitted = layoutCandidate(candidate, streams, offsets, {
      position: { offset: offsets.positionByteOffset, byteLength: encoded.position.byteLength },
      index: { offset: offsets.indexByteOffset, byteLength: encoded.index.byteLength },
      transform: { offset: offsets.transformByteOffset, byteLength: encoded.transform.byteLength },
    });
    prototypes.push(emitted.prototype);
  }
  return { streams, prototypes, offsets };
}

function projectedBinaryBytes(runtime, streams) {
  const streamBytes = (stream) => stream.segments.reduce((sum, segment) => sum + segment.byteLength, 0);
  return {
    [streams.positions.file]: streamBytes(streams.positions),
    [streams.indices.file]: streamBytes(streams.indices),
    [streams.transforms.file]: streamBytes(streams.transforms),
    'instance_aabb_fp32.bin': runtime.aabbs.byteLength,
    'instance_to_glb_uint32.bin': runtime.instanceToGlb.byteLength,
  };
}

function jsonFileBytes(value) {
  return Buffer.byteLength(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function writeOfflineReport(args, meta, counters, primitiveAudits) {
  const reportPath = offlineReportPath(args);
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  const report = {
    schema: OFFLINE_REPORT_SCHEMA,
    runtimeAsset: false,
    purpose: 'primitive-level export audit; excluded from the runtime download asset',
    runtimeAssetDirectory: path.basename(args.outputDir),
    sceneName: meta.sceneName,
    variant: meta.variant,
    selection: meta.selection,
    aggregate: {
      sourceGlbCount: meta.source.sceneGlbCount,
      scannedGlbCount: meta.source.scannedGlbCount,
      sourcePrimitiveCount: counters.sourcePrimitiveCount,
      supportedPrimitiveCount: counters.supportedPrimitiveCount,
      opaquePrimitiveCount: counters.opaquePrimitiveCount,
      excludedPrimitiveCount: counters.excludedPrimitiveCount,
      excludedByReason: Object.fromEntries(
        Object.entries(counters.excludedByReason).sort(([a], [b]) => a.localeCompare(b)),
      ),
    },
    primitiveAudits,
  };
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  return reportPath;
}

async function buildExport(args) {
  const assetsDir = resolveAssetsDir(args);
  const sceneWeb = JSON.parse(fs.readFileSync(path.join(assetsDir, 'sceneWeb.json'), 'utf8'));
  const glbIndex = JSON.parse(fs.readFileSync(path.join(assetsDir, 'glbIndex.json'), 'utf8'));
  const runtimeMeta = JSON.parse(fs.readFileSync(path.join(assetsDir, 'runtimeVisibilityMeta.json'), 'utf8'));
  const runtime = buildRuntimeArrays(runtimeMeta, glbIndex);
  const entries = glbIndex.entries.map((entry, index) => {
    if (Number(entry.globalId) !== index) throw new Error('glbIndex entries must be dense by globalId.');
    return entry;
  });
  const selectedEntries = args.maxGlbs > 0 ? entries.slice(0, args.maxGlbs) : entries;
  const globalRecords = runtimeMeta.globalGlbRecords || [];
  if (globalRecords.length !== runtime.glbCount) throw new Error('runtime globalGlbRecords count disagrees with globalGlbCount.');
  await MeshoptDecoder.ready;
  await MeshoptEncoder.ready;
  if (!MeshoptEncoder.supported) throw new Error('meshoptimizer encoder is unavailable.');
  prepareOutput(args.outputDir, args.overwrite);
  const stageDir = args.variant === 'equal-asset' ? path.join(args.outputDir, '.staging') : null;
  if (stageDir) fs.mkdirSync(stageDir, { recursive: true });
  const streams = createStreamDescriptors();
  const writers = {
    positions: new BinaryWriter(path.join(args.outputDir, streams.positions.file)),
    indices: new BinaryWriter(path.join(args.outputDir, streams.indices.file)),
    transforms: new BinaryWriter(path.join(args.outputDir, streams.transforms.file)),
  };
  const offsets = {
    prototypeCount: 0,
    vertexOffset: 0,
    indexOffset: 0,
    instanceOffset: 0,
    positionByteOffset: 0,
    indexByteOffset: 0,
    transformByteOffset: 0,
  };
  const prototypes = [];
  const candidates = [];
  const primitiveAudits = [];
  const counters = {
    sourceGlbCount: entries.length,
    scannedGlbCount: selectedEntries.length,
    sourceGlbBytes: 0,
    sourcePrimitiveCount: 0,
    supportedPrimitiveCount: 0,
    opaquePrimitiveCount: 0,
    excludedPrimitiveCount: 0,
    excludedByReason: {},
    sourcePrototypeTriangles: 0,
    sourceInstanceTriangleCount: 0,
    emptyPlaceholderGlbCount: 0,
    emptyPlaceholderInstanceCount: 0,
  };

  for (let entryIndex = 0; entryIndex < selectedEntries.length; entryIndex += 1) {
    const entry = selectedEntries[entryIndex];
    const globalRecord = globalRecords[Number(entry.globalId)];
    const expectedInstanceCount = globalRecord?.componentGlobalIds?.length ?? 0;
    const inspected = await inspectSourceEntry(entry, assetsDir, expectedInstanceCount);
    counters.sourceGlbBytes += inspected.sourceBytes;
    if (inspected.emptyPlaceholder) {
      counters.emptyPlaceholderGlbCount += 1;
      counters.emptyPlaceholderInstanceCount += inspected.expectedInstanceCount;
      counters.excludedByReason['empty-source-placeholder'] =
        (counters.excludedByReason['empty-source-placeholder'] || 0) + 1;
    }
    for (const primitive of inspected.primitives) {
      counters.sourcePrimitiveCount += 1;
      const primitiveInstanceCount = primitive.instanceMatrices.length / 16;
      if (!primitive.supported) {
        counters.excludedPrimitiveCount += 1;
        counters.excludedByReason[primitive.reason] = (counters.excludedByReason[primitive.reason] || 0) + 1;
        primitiveAudits.push({
          sourceGlobalGlbId: entry.globalId,
          sourcePrimitiveIndex: primitive.primitiveIndex,
          sourcePath: entry.path,
          occluder: false,
          reason: primitive.reason,
        });
        continue;
      }
      counters.supportedPrimitiveCount += 1;
      counters.sourcePrototypeTriangles += primitive.triangleCount;
      counters.sourceInstanceTriangleCount += primitive.triangleCount * primitiveInstanceCount;
      const primitiveAudit = {
        sourceGlobalGlbId: entry.globalId,
        sourcePrimitiveIndex: primitive.primitiveIndex,
        sourcePath: entry.path,
        vertexCount: primitive.vertexCount,
        triangleCount: primitive.triangleCount,
        instanceCount: primitiveInstanceCount,
        materialDecision: primitive.material,
        occluder: primitive.material.occluder,
        reason: primitive.material.reason,
        selected: false,
      };
      primitiveAudits.push(primitiveAudit);
      if (!primitive.material.occluder) {
        counters.excludedPrimitiveCount += 1;
        counters.excludedByReason[primitive.material.reason] = (counters.excludedByReason[primitive.material.reason] || 0) + 1;
        continue;
      }
      counters.opaquePrimitiveCount += 1;
      const matrices = primitive.instanceMatrices;
      const positionEncoded = encodeStream(primitive.positions, primitive.vertexCount, 12, 'ATTRIBUTES');
      const indexEncoded = encodeStream(primitive.indices, primitive.indices.length, 4, 'TRIANGLES');
      const transformEncoded = encodeStream(matrices, matrices.length / 16, 64, 'ATTRIBUTES');
      const candidate = {
        prototypeOrdinal: candidates.length,
        sourceGlobalGlbId: entry.globalId,
        sourcePrimitiveIndex: primitive.primitiveIndex,
        sourcePath: entry.path,
        vertexCount: primitive.vertexCount,
        indexCount: primitive.indices.length,
        triangleCount: primitive.triangleCount,
        matrices,
        bounds: primitive.bounds,
        positionEncoded,
        indexEncoded,
        transformEncoded,
        compressedBytes: positionEncoded.byteLength + indexEncoded.byteLength
          + transformEncoded.byteLength,
        stageDir: null,
        audit: primitiveAudit,
      };
      if (stageDir) candidate.stageDir = writeCandidateSegments(candidate, stageDir);
      else {
        const emitted = appendCandidate(candidate, writers, streams, offsets);
        prototypes.push(emitted.prototype);
        primitiveAudit.selected = true;
      }
      candidates.push(candidate);
    }
    if (args.progressEvery > 0 && (entryIndex + 1) % args.progressEvery === 0) {
      console.log(`[geometry-shell-hzb] scanned ${entryIndex + 1}/${selectedEntries.length} GLBs; opaque primitives=${counters.opaquePrimitiveCount}`);
    }
  }

  let importance = null;
  let budgetBytes = null;
  let selection = null;
  if (args.variant === 'equal-asset') {
    importance = await readImportancePoses(args.importancePosePlan, args.importancePoseCount);
    budgetBytes = args.budgetBytes > 0 ? args.budgetBytes : directoryBytes(args.budgetFromDir);
    const fixedPayloadBytes = runtime.aabbs.byteLength + runtime.instanceToGlb.byteLength;
    const available = budgetBytes - fixedPayloadBytes;
    if (available < 0) throw new Error(`equal-asset budget ${budgetBytes} is smaller than fixed runtime payload ${fixedPayloadBytes}.`);
    for (const candidate of candidates) candidate.importanceScore = candidateImportance(candidate, importance);
    const ranked = [...candidates].sort((a, b) => {
      const densityA = a.importanceScore / Math.max(1, a.compressedBytes);
      const densityB = b.importanceScore / Math.max(1, b.compressedBytes);
      return densityB - densityA || a.sourceGlobalGlbId - b.sourceGlobalGlbId || a.sourcePrimitiveIndex - b.sourcePrimitiveIndex;
    });
    let used = 0;
    const selectedRanked = [];
    for (const candidate of ranked) {
      if (used + candidate.compressedBytes > available) continue;
      selectedRanked.push(candidate);
      used += candidate.compressedBytes;
    }
    const settleProjectedSelection = () => {
      const selectedSet = new Set(selectedRanked);
      const selectedCandidates = candidates.filter((candidate) => selectedSet.has(candidate));
      const layout = projectCandidates(selectedCandidates);
      const projectedBytes = projectedBinaryBytes(runtime, layout.streams);
      const projectedSelection = {
        mode: 'complete-primitive-deletion-only',
        budgetBytes,
        fixedRuntimePayloadBytes: fixedPayloadBytes,
        selectedCompressedGeometryBytes: used,
        selectedPrimitiveCount: selectedCandidates.length,
        candidatePrimitiveCount: candidates.length,
        importancePoseCount: importance.length,
        importancePosePlan: path.basename(args.importancePosePlan),
        metadataReserveBytes: 0,
        shellMetaBytes: 0,
        totalAssetBytes: 0,
      };
      let projectedMeta = null;
      for (let iteration = 0; iteration < 8; iteration += 1) {
        projectedMeta = buildShellMeta({
          args,
          assetsDir,
          sceneWeb,
          runtimeMeta,
          glbIndex,
          entries,
          selectedEntries,
          runtime,
          counters,
          streams: layout.streams,
          prototypes: layout.prototypes,
          binaryBytes: projectedBytes,
          selection: projectedSelection,
        });
        const shellMetaBytes = jsonFileBytes(projectedMeta);
        const totalAssetBytes = projectedMeta.stats.binaryPayloadBytes + shellMetaBytes;
        if (projectedSelection.shellMetaBytes === shellMetaBytes
            && projectedSelection.totalAssetBytes === totalAssetBytes) {
          return { selectedSet, selectedCandidates, selection: projectedSelection };
        }
        projectedSelection.metadataReserveBytes = shellMetaBytes;
        projectedSelection.shellMetaBytes = shellMetaBytes;
        projectedSelection.totalAssetBytes = totalAssetBytes;
      }
      throw new Error('equal-asset projected metadata byte accounting did not converge.');
    };
    let projected = settleProjectedSelection();
    while (projected.selection.totalAssetBytes > budgetBytes && selectedRanked.length > 0) {
      const excess = projected.selection.totalAssetBytes - budgetBytes;
      let removedBytes = 0;
      while (selectedRanked.length > 0 && removedBytes < excess) {
        const removed = selectedRanked.pop();
        removedBytes += removed.compressedBytes;
        used -= removed.compressedBytes;
      }
      projected = settleProjectedSelection();
    }
    if (projected.selection.totalAssetBytes > budgetBytes) {
      throw new Error(`equal-asset budget ${budgetBytes} cannot fit fixed runtime payload and shell metadata.`);
    }
    for (const candidate of projected.selectedCandidates) {
      const emitted = appendCandidate(candidate, writers, streams, offsets);
      prototypes.push(emitted.prototype);
      candidate.audit.selected = true;
    }
    selection = projected.selection;
  }
  writers.positions.close();
  writers.indices.close();
  writers.transforms.close();
  if (stageDir) fs.rmSync(stageDir, { recursive: true, force: true });

  const aabbFile = 'instance_aabb_fp32.bin';
  const mappingFile = 'instance_to_glb_uint32.bin';
  writeRaw(path.join(args.outputDir, aabbFile), runtime.aabbs);
  writeRaw(path.join(args.outputDir, mappingFile), runtime.instanceToGlb);
  const binaryFiles = [
    streams.positions.file,
    streams.indices.file,
    streams.transforms.file,
    aabbFile,
    mappingFile,
  ];
  const binaryBytes = Object.fromEntries(binaryFiles.map((file) => [file, fs.statSync(path.join(args.outputDir, file)).size]));
  const buildMeta = () => buildShellMeta({
    args,
    assetsDir,
    sceneWeb,
    runtimeMeta,
    glbIndex,
    entries,
    selectedEntries,
    runtime,
    counters,
    streams,
    prototypes,
    binaryBytes,
    selection,
  });
  let meta = buildMeta();
  if (selection) {
    let converged = false;
    for (let iteration = 0; iteration < 8; iteration += 1) {
      const shellMetaBytes = jsonFileBytes(meta);
      const totalAssetBytes = meta.stats.binaryPayloadBytes + shellMetaBytes;
      if (selection.shellMetaBytes === shellMetaBytes && selection.totalAssetBytes === totalAssetBytes) {
        converged = true;
        break;
      }
      selection.shellMetaBytes = shellMetaBytes;
      selection.totalAssetBytes = totalAssetBytes;
      meta = buildMeta();
    }
    if (!converged) throw new Error('equal-asset metadata byte accounting did not converge.');
    const finalShellMetaBytes = jsonFileBytes(meta);
    const finalTotalAssetBytes = meta.stats.binaryPayloadBytes + finalShellMetaBytes;
    if (selection.shellMetaBytes !== finalShellMetaBytes || selection.totalAssetBytes !== finalTotalAssetBytes) {
      throw new Error('equal-asset metadata byte accounting changed after convergence.');
    }
    if (selection.totalAssetBytes > selection.budgetBytes) {
      throw new Error(`equal-asset output ${selection.totalAssetBytes} exceeds budget ${selection.budgetBytes}.`);
    }
  }
  const shellMetaFile = 'shell_meta.json';
  const shellMetaPath = path.join(args.outputDir, shellMetaFile);
  fs.writeFileSync(shellMetaPath, `${JSON.stringify(meta, null, 2)}\n`, 'utf8');
  const runtimeFiles = [...binaryFiles, shellMetaFile];
  const actualRuntimeAssetBytes = runtimeAssetBytes(args.outputDir, runtimeFiles);
  if (selection) {
    if (actualRuntimeAssetBytes !== selection.totalAssetBytes) {
      throw new Error(
        `equal-asset runtime files total ${actualRuntimeAssetBytes} disagrees with accounted total ${selection.totalAssetBytes}.`,
      );
    }
    if (actualRuntimeAssetBytes > selection.budgetBytes) {
      throw new Error(`equal-asset runtime files total ${actualRuntimeAssetBytes} exceeds budget ${selection.budgetBytes}.`);
    }
  }
  const reportPath = writeOfflineReport(args, meta, counters, primitiveAudits);
  return {
    outputDir: args.outputDir,
    meta,
    binaryBytes,
    runtimeFiles,
    runtimeAssetBytes: actualRuntimeAssetBytes,
    offlineReportPath: reportPath,
  };
}

export {
  buildExport,
  buildRuntimeArrays,
  inspectSourceEntry,
  parseArgs,
  readAccessor,
  readIndices,
  resolveAssetsDir,
};

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  const args = parseArgs(process.argv);
  buildExport(args)
    .then(({ outputDir, meta, binaryBytes, runtimeAssetBytes, offlineReportPath }) => {
      console.log(JSON.stringify({
        schema: meta.schema,
        variant: meta.variant,
        outputDir,
        sceneName: meta.sceneName,
        sourceGlbCount: meta.source.sceneGlbCount,
        scannedGlbCount: meta.source.scannedGlbCount,
        instanceCount: meta.instanceCount,
        globalGlbCount: meta.globalGlbCount,
        prototypeCount: meta.prototypeCount,
        prototypeTriangles: meta.stats.prototypeTriangles,
        rasterizedTriangleInstanceCount: meta.stats.rasterizedTriangleInstanceCount,
        compressedGeometryBytes: meta.stats.compressedGeometryBytes,
        binaryPayloadBytes: meta.stats.binaryPayloadBytes,
        runtimeAssetBytes,
        offlineReportPath,
        binaryBytes,
        excludedByReason: meta.stats.excludedByReason,
      }, null, 2));
    })
    .catch((error) => {
      console.error(error?.stack || error);
      process.exitCode = 1;
    });
}
