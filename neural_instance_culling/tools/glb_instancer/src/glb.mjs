import fs from 'node:fs';
import path from 'node:path';

const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const JSON_CHUNK_TYPE = 0x4e4f534a;
const BIN_CHUNK_TYPE = 0x004e4942;

const COMPONENT_UNSIGNED_BYTE = 5121;
const COMPONENT_UNSIGNED_SHORT = 5123;
const COMPONENT_UNSIGNED_INT = 5125;
const COMPONENT_FLOAT = 5126;

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
  if (Array.isArray(node.matrix) && node.matrix.length === 16) return node.matrix.map(Number);
  const t = node.translation || [0, 0, 0];
  const r = node.rotation || [0, 0, 0, 1];
  const s = node.scale || [1, 1, 1];
  const [x, y, z, w] = r.map(Number);
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
    (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
    (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
    (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
    Number(t[0]), Number(t[1]), Number(t[2]), 1,
  ];
}

function transformPositions(positions, matrix) {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    out[i] = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    out[i + 1] = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    out[i + 2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
  }
  return out;
}

function parseChunks(bytes, filePath) {
  if (bytes.length < 28 || bytes.readUInt32LE(0) !== GLB_MAGIC) throw new Error(`${filePath}: invalid GLB magic`);
  if (bytes.readUInt32LE(4) !== GLB_VERSION) throw new Error(`${filePath}: only glTF 2.0 is supported`);
  if (bytes.readUInt32LE(8) !== bytes.length) throw new Error(`${filePath}: inconsistent GLB length`);
  let json = null;
  let bin = null;
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const length = bytes.readUInt32LE(offset);
    const type = bytes.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = start + length;
    if (end > bytes.length) throw new Error(`${filePath}: truncated GLB chunk`);
    if (type === JSON_CHUNK_TYPE) json = JSON.parse(bytes.toString('utf8', start, end).trim());
    if (type === BIN_CHUNK_TYPE) bin = bytes.subarray(start, end);
    offset = end;
  }
  if (!json || !bin) throw new Error(`${filePath}: JSON and BIN chunks are required`);
  return { json, bin };
}

function accessorInfo(json, accessorIndex, expectedType, allowedComponents, normalized = null) {
  const accessor = json.accessors?.[accessorIndex];
  if (!accessor) throw new Error(`missing accessor ${accessorIndex}`);
  const view = json.bufferViews?.[accessor.bufferView];
  if (!view || Number(view.buffer || 0) !== 0 || view.byteStride || accessor.sparse) {
    throw new Error(`unsupported accessor layout ${accessorIndex}`);
  }
  if (accessor.type !== expectedType || !allowedComponents.includes(accessor.componentType)) {
    throw new Error(`unexpected accessor format ${accessorIndex}`);
  }
  if (normalized != null && Boolean(accessor.normalized) !== normalized) {
    throw new Error(`unexpected normalized flag ${accessorIndex}`);
  }
  return {
    accessor,
    byteOffset: Number(view.byteOffset || 0) + Number(accessor.byteOffset || 0),
  };
}

function readIndices(bin, info) {
  const count = Number(info.accessor.count);
  const offset = bin.byteOffset + info.byteOffset;
  let source;
  if (info.accessor.componentType === COMPONENT_UNSIGNED_BYTE) source = new Uint8Array(bin.buffer, offset, count);
  else if (info.accessor.componentType === COMPONENT_UNSIGNED_SHORT) source = new Uint16Array(bin.buffer, offset, count);
  else source = new Uint32Array(bin.buffer, offset, count);
  return Uint32Array.from(source);
}

function findMeshNode(json) {
  const scene = json.scenes?.[json.scene ?? 0];
  if (!scene) throw new Error('default scene is missing');
  const found = [];
  const visit = (nodeIndex, parentMatrix) => {
    const node = json.nodes?.[nodeIndex];
    if (!node) throw new Error(`missing node ${nodeIndex}`);
    const worldMatrix = multiplyMatrix(parentMatrix, matrixFromNode(node));
    if (node.mesh != null) found.push({ node, nodeIndex, meshIndex: Number(node.mesh), worldMatrix });
    for (const child of node.children || []) visit(Number(child), worldMatrix);
  };
  for (const root of scene.nodes || []) visit(Number(root), identityMatrix());
  if (found.length !== 1) throw new Error(`current version expects exactly one mesh node, found ${found.length}`);
  if (found[0].node.extensions?.EXT_mesh_gpu_instancing) throw new Error('instanced input GLBs are not accepted as monolithic input');
  return found[0];
}

export function readMonolithicGlb(filePath) {
  const bytes = fs.readFileSync(filePath);
  const { json, bin } = parseChunks(bytes, filePath);
  const meshNode = findMeshNode(json);
  const mesh = json.meshes?.[meshNode.meshIndex];
  if (!mesh || (mesh.primitives || []).length !== 1) throw new Error('current version expects one mesh primitive');
  const primitive = mesh.primitives[0];
  if (Number(primitive.mode ?? 4) !== 4 || primitive.indices == null || primitive.attributes?.POSITION == null) {
    throw new Error('current version expects indexed triangle geometry');
  }
  const positionInfo = accessorInfo(json, primitive.attributes.POSITION, 'VEC3', [COMPONENT_FLOAT], false);
  const vertexCount = Number(positionInfo.accessor.count);
  const sourcePositions = new Float32Array(
    bin.buffer,
    bin.byteOffset + positionInfo.byteOffset,
    vertexCount * 3,
  );
  const positions = transformPositions(sourcePositions, meshNode.worldMatrix);
  const indexInfo = accessorInfo(
    json,
    primitive.indices,
    'SCALAR',
    [COMPONENT_UNSIGNED_BYTE, COMPONENT_UNSIGNED_SHORT, COMPONENT_UNSIGNED_INT],
    false,
  );
  const indices = readIndices(bin, indexInfo);
  if (indices.length % 3 !== 0) throw new Error('index count must be divisible by three');

  let colors;
  if (primitive.attributes.COLOR_0 != null) {
    const colorInfo = accessorInfo(json, primitive.attributes.COLOR_0, 'VEC4', [COMPONENT_UNSIGNED_BYTE], true);
    if (Number(colorInfo.accessor.count) !== vertexCount) throw new Error('POSITION and COLOR_0 counts differ');
    colors = new Uint8Array(vertexCount * 4);
    colors.set(new Uint8Array(bin.buffer, bin.byteOffset + colorInfo.byteOffset, vertexCount * 4));
  } else {
    colors = new Uint8Array(vertexCount * 4);
    colors.fill(255);
  }

  return {
    filePath,
    bytes: bytes.length,
    positions,
    colors,
    indices,
    vertexCount,
    triangleCount: indices.length / 3,
  };
}

function writeGlb(filePath, json, binary) {
  const jsonRaw = Buffer.from(JSON.stringify(json), 'utf8');
  const jsonPadded = Buffer.alloc(pad4(jsonRaw.length), 0x20);
  jsonRaw.copy(jsonPadded);
  const binaryPaddedLength = pad4(binary.length);
  const totalLength = 12 + 8 + jsonPadded.length + 8 + binaryPaddedLength;
  const header = Buffer.alloc(12);
  header.writeUInt32LE(GLB_MAGIC, 0);
  header.writeUInt32LE(GLB_VERSION, 4);
  header.writeUInt32LE(totalLength, 8);
  const jsonHeader = Buffer.alloc(8);
  jsonHeader.writeUInt32LE(jsonPadded.length, 0);
  jsonHeader.writeUInt32LE(JSON_CHUNK_TYPE, 4);
  const binHeader = Buffer.alloc(8);
  binHeader.writeUInt32LE(binaryPaddedLength, 0);
  binHeader.writeUInt32LE(BIN_CHUNK_TYPE, 4);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const fd = fs.openSync(filePath, 'w');
  try {
    fs.writeSync(fd, header);
    fs.writeSync(fd, jsonHeader);
    fs.writeSync(fd, jsonPadded);
    fs.writeSync(fd, binHeader);
    fs.writeSync(fd, binary);
    if (binaryPaddedLength > binary.length) fs.writeSync(fd, Buffer.alloc(binaryPaddedLength - binary.length));
  } finally {
    fs.closeSync(fd);
  }
}

export function writePrototypeGlb(filePath, prototype) {
  const { positions, colors, indices, instances, name } = prototype;
  const vertexCount = positions.length / 3;
  const indexBytes = indices.length * 4;
  const positionOffset = pad4(indexBytes);
  const positionBytes = positions.length * 4;
  const colorOffset = pad4(positionOffset + positionBytes);
  const colorBytes = colors.length;
  const repeated = instances.length > 1;
  const translationOffset = pad4(colorOffset + colorBytes);
  const translationBytes = repeated ? instances.length * 3 * 4 : 0;
  const rotationOffset = pad4(translationOffset + translationBytes);
  const rotationBytes = repeated ? instances.length * 4 * 4 : 0;
  const scaleOffset = pad4(rotationOffset + rotationBytes);
  const scaleBytes = repeated ? instances.length * 3 * 4 : 0;
  const binaryLength = pad4(scaleOffset + scaleBytes);
  const binary = Buffer.alloc(binaryLength);
  new Uint32Array(binary.buffer, binary.byteOffset, indices.length).set(indices);
  new Float32Array(binary.buffer, binary.byteOffset + positionOffset, positions.length).set(positions);
  binary.set(colors, colorOffset);

  if (repeated) {
    const translations = new Float32Array(binary.buffer, binary.byteOffset + translationOffset, instances.length * 3);
    const rotations = new Float32Array(binary.buffer, binary.byteOffset + rotationOffset, instances.length * 4);
    const scales = new Float32Array(binary.buffer, binary.byteOffset + scaleOffset, instances.length * 3);
    for (let i = 0; i < instances.length; i += 1) {
      translations.set(instances[i].translation, i * 3);
      rotations.set(instances[i].rotation, i * 4);
      scales.set([1, 1, 1], i * 3);
    }
  }

  const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (let i = 0; i < positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      bounds.min[axis] = Math.min(bounds.min[axis], positions[i + axis]);
      bounds.max[axis] = Math.max(bounds.max[axis], positions[i + axis]);
    }
  }
  const accessors = [
    { bufferView: 0, componentType: COMPONENT_UNSIGNED_INT, count: indices.length, type: 'SCALAR', min: [0], max: [vertexCount - 1] },
    { bufferView: 1, componentType: COMPONENT_FLOAT, count: vertexCount, type: 'VEC3', min: bounds.min, max: bounds.max },
    { bufferView: 2, componentType: COMPONENT_UNSIGNED_BYTE, normalized: true, count: vertexCount, type: 'VEC4' },
  ];
  const bufferViews = [
    { buffer: 0, byteOffset: 0, byteLength: indexBytes, target: 34963 },
    { buffer: 0, byteOffset: positionOffset, byteLength: positionBytes, target: 34962 },
    { buffer: 0, byteOffset: colorOffset, byteLength: colorBytes, target: 34962 },
  ];
  const geometryNode = { name: 'geometry_0', mesh: 0 };
  if (repeated) {
    const translationAccessor = accessors.length;
    accessors.push({ bufferView: bufferViews.length, componentType: COMPONENT_FLOAT, count: instances.length, type: 'VEC3' });
    bufferViews.push({ buffer: 0, byteOffset: translationOffset, byteLength: translationBytes });
    const rotationAccessor = accessors.length;
    accessors.push({ bufferView: bufferViews.length, componentType: COMPONENT_FLOAT, count: instances.length, type: 'VEC4' });
    bufferViews.push({ buffer: 0, byteOffset: rotationOffset, byteLength: rotationBytes });
    const scaleAccessor = accessors.length;
    accessors.push({ bufferView: bufferViews.length, componentType: COMPONENT_FLOAT, count: instances.length, type: 'VEC3' });
    bufferViews.push({ buffer: 0, byteOffset: scaleOffset, byteLength: scaleBytes });
    geometryNode.extensions = {
      EXT_mesh_gpu_instancing: {
        attributes: {
          TRANSLATION: translationAccessor,
          ROTATION: rotationAccessor,
          SCALE: scaleAccessor,
        },
      },
    };
  }
  const json = {
    asset: { version: '2.0', generator: 'slm-glb-instancer 0.1.0' },
    scene: 0,
    scenes: [{ name, nodes: [0] }],
    nodes: [{ name: 'world', children: [1] }, geometryNode],
    meshes: [{
      name: 'geometry_0',
      extras: {},
      primitives: [{ attributes: { POSITION: 1, COLOR_0: 2 }, indices: 0, material: 0, mode: 4 }],
    }],
    materials: [{
      name: 'VertexColorMaterial',
      pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1 },
    }],
    accessors,
    bufferViews,
    buffers: [{ byteLength: binary.length }],
  };
  if (repeated) {
    json.extensionsUsed = ['EXT_mesh_gpu_instancing'];
    json.extensionsRequired = ['EXT_mesh_gpu_instancing'];
  }
  writeGlb(filePath, json, binary);
}

export function inspectGlbStructure(filePath) {
  const bytes = fs.readFileSync(filePath);
  const { json } = parseChunks(bytes, filePath);
  return {
    bytes: bytes.length,
    nodeCount: json.nodes?.length || 0,
    meshCount: json.meshes?.length || 0,
    primitiveCount: (json.meshes || []).reduce((sum, mesh) => sum + (mesh.primitives || []).length, 0),
    materialCount: json.materials?.length || 0,
    extensionsUsed: json.extensionsUsed || [],
  };
}

