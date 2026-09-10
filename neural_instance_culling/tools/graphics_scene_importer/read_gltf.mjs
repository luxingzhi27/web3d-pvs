import fs from 'node:fs';
import path from 'node:path';

import { MeshoptDecoder } from 'meshoptimizer/decoder';

const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const JSON_CHUNK_TYPE = 0x4e4f534a;
const BIN_CHUNK_TYPE = 0x004e4942;
const TRIANGLES_MODE = 4;

const COMPONENT_TYPES = {
  5120: { bytes: 1, read: 'getInt8' },
  5121: { bytes: 1, read: 'getUint8' },
  5122: { bytes: 2, read: 'getInt16' },
  5123: { bytes: 2, read: 'getUint16' },
  5125: { bytes: 4, read: 'getUint32' },
  5126: { bytes: 4, read: 'getFloat32' },
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

const KNOWN_ALPHA_MODES = new Set(['OPAQUE', 'MASK', 'BLEND']);
const TRANSPARENCY_EXTENSIONS = new Set([
  'KHR_materials_transmission',
  'KHR_materials_volume',
  'KHR_materials_diffuse_transmission',
  'KHR_materials_dispersion',
]);
const SUPPORTED_IMAGE_MIMES = new Set(['image/png', 'image/jpeg', 'image/jpg', 'image/webp']);

function fail(filePath, message) {
  throw new Error(`${filePath}: ${message}`);
}

function finite(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${label} must be finite`);
  return number;
}

function nonNegativeInteger(value, label) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 0) throw new Error(`${label} must be a non-negative integer`);
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
      for (let k = 0; k < 4; k += 1) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
    }
  }
  return out;
}

function nodeMatrix(node, label) {
  if (Array.isArray(node.matrix)) {
    if (node.matrix.length !== 16) throw new Error(`${label}.matrix must have 16 values`);
    return node.matrix.map((value, index) => finite(value, `${label}.matrix[${index}]`));
  }
  const translation = Array.isArray(node.translation) ? node.translation : [0, 0, 0];
  const rotation = Array.isArray(node.rotation) ? node.rotation : [0, 0, 0, 1];
  const scale = Array.isArray(node.scale) ? node.scale : [1, 1, 1];
  if (translation.length !== 3 || rotation.length !== 4 || scale.length !== 3) {
    throw new Error(`${label} TRS has an invalid length`);
  }
  const [tx, ty, tz] = translation.map((value, index) => finite(value, `${label}.translation[${index}]`));
  const [x, y, z, w] = rotation.map((value, index) => finite(value, `${label}.rotation[${index}]`));
  const [sx, sy, sz] = scale.map((value, index) => finite(value, `${label}.scale[${index}]`));
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

function transformPositions(values, matrix) {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 3) {
    const x = values[index];
    const y = values[index + 1];
    const z = values[index + 2];
    output[index] = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    output[index + 1] = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    output[index + 2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
  }
  return output;
}

function normalMatrix(matrix, label) {
  const a00 = matrix[0];
  const a01 = matrix[4];
  const a02 = matrix[8];
  const a10 = matrix[1];
  const a11 = matrix[5];
  const a12 = matrix[9];
  const a20 = matrix[2];
  const a21 = matrix[6];
  const a22 = matrix[10];
  const determinant = a00 * (a11 * a22 - a12 * a21)
    - a01 * (a10 * a22 - a12 * a20)
    + a02 * (a10 * a21 - a11 * a20);
  if (Math.abs(determinant) < 1e-12) throw new Error(`${label} has a singular world transform`);
  const inverse = [
    (a11 * a22 - a12 * a21) / determinant,
    (a02 * a21 - a01 * a22) / determinant,
    (a01 * a12 - a02 * a11) / determinant,
    (a12 * a20 - a10 * a22) / determinant,
    (a00 * a22 - a02 * a20) / determinant,
    (a02 * a10 - a00 * a12) / determinant,
    (a10 * a21 - a11 * a20) / determinant,
    (a01 * a20 - a00 * a21) / determinant,
    (a00 * a11 - a01 * a10) / determinant,
  ];
  return inverse;
}

function transformDirections(values, matrix, label, useNormalMatrix = true) {
  const output = new Float32Array(values.length);
  const n = useNormalMatrix ? normalMatrix(matrix, label) : matrix;
  for (let index = 0; index < values.length; index += 3) {
    const x = values[index];
    const y = values[index + 1];
    const z = values[index + 2];
    let tx;
    let ty;
    let tz;
    if (useNormalMatrix) {
      // The inverse-transpose is stored as a row-major 3x3 matrix.
      tx = n[0] * x + n[3] * y + n[6] * z;
      ty = n[1] * x + n[4] * y + n[7] * z;
      tz = n[2] * x + n[5] * y + n[8] * z;
    } else {
      tx = n[0] * x + n[4] * y + n[8] * z;
      ty = n[1] * x + n[5] * y + n[9] * z;
      tz = n[2] * x + n[6] * y + n[10] * z;
    }
    const length = Math.hypot(tx, ty, tz);
    if (length > 1e-12) {
      tx /= length;
      ty /= length;
      tz /= length;
    }
    output[index] = tx;
    output[index + 1] = ty;
    output[index + 2] = tz;
  }
  return output;
}

function boundsForPositions(values) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < values.length; index += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], values[index + axis]);
      max[axis] = Math.max(max[axis], values[index + axis]);
    }
  }
  if (!Number.isFinite(min[0])) return null;
  return {
    min,
    max,
    center: min.map((value, axis) => (value + max[axis]) * 0.5),
    size: min.map((value, axis) => Math.max(0, max[axis] - value)),
  };
}

function parseGlb(filePath, bytes) {
  if (bytes.length < 20 || bytes.readUInt32LE(0) !== GLB_MAGIC) fail(filePath, 'invalid GLB magic');
  if (bytes.readUInt32LE(4) !== GLB_VERSION) fail(filePath, 'only glTF 2.0 GLB files are supported');
  if (bytes.readUInt32LE(8) !== bytes.length) fail(filePath, 'GLB length does not match the file');
  let json = null;
  let binary = null;
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const length = bytes.readUInt32LE(offset);
    const type = bytes.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = start + length;
    if (end > bytes.length || end < start) fail(filePath, 'GLB chunk exceeds the file');
    if (type === JSON_CHUNK_TYPE) {
      try {
        json = JSON.parse(bytes.toString('utf8', start, end).trim());
      } catch (error) {
        throw new Error(`${filePath}: GLB JSON chunk is invalid: ${error.message}`);
      }
    } else if (type === BIN_CHUNK_TYPE) {
      binary = bytes.subarray(start, end);
    }
    offset = end;
  }
  if (!json || typeof json !== 'object') fail(filePath, 'GLB JSON chunk is missing');
  return { json, binary };
}

function decodeDataUri(uri, filePath) {
  const match = /^data:([^,]*?),(.*)$/s.exec(uri);
  if (!match) fail(filePath, 'unsupported data URI');
  const metadata = match[1].toLowerCase();
  if (metadata.includes(';base64')) return Buffer.from(match[2], 'base64');
  return Buffer.from(decodeURIComponent(match[2]), 'utf8');
}

function resolveExternalUri(baseDir, uri, filePath) {
  if (/^[a-z][a-z0-9+.-]*:/i.test(uri)) fail(filePath, `network or custom URI is not allowed: ${uri}`);
  const resolved = path.resolve(baseDir, decodeURIComponent(uri.split('#', 1)[0]));
  if (resolved !== baseDir && !resolved.startsWith(`${baseDir}${path.sep}`)) {
    fail(filePath, `external resource escapes the source directory: ${uri}`);
  }
  return resolved;
}

function loadBufferBytes(filePath, baseDir, descriptor, index, binary) {
  if (index === 0 && binary && !descriptor.uri) return binary;
  if (!descriptor.uri) fail(filePath, `buffer ${index} has no URI or GLB BIN chunk`);
  if (descriptor.uri.startsWith('data:')) return decodeDataUri(descriptor.uri, filePath);
  const externalPath = resolveExternalUri(baseDir, descriptor.uri, filePath);
  try {
    return fs.readFileSync(externalPath);
  } catch (error) {
    throw new Error(`${filePath}: cannot read buffer ${index} ${externalPath}: ${error.message}`);
  }
}

function parseSource(filePath) {
  const resolved = path.resolve(filePath);
  const fileBytes = fs.readFileSync(resolved);
  const extension = path.extname(resolved).toLowerCase();
  let json;
  let binary = null;
  let format;
  if (extension === '.glb') {
    ({ json, binary } = parseGlb(resolved, fileBytes));
    format = 'glb';
  } else if (extension === '.gltf') {
    try {
      json = JSON.parse(fileBytes.toString('utf8'));
    } catch (error) {
      throw new Error(`${resolved}: glTF JSON is invalid: ${error.message}`);
    }
    format = 'gltf';
  } else {
    fail(resolved, 'input must have a .gltf or .glb extension');
  }
  if (!json || json.asset?.version !== '2.0') fail(resolved, 'only glTF asset version 2.0 is supported');
  if (!Array.isArray(json.buffers)) fail(resolved, 'buffers must be an array');
  const buffers = json.buffers.map((descriptor, index) => {
    if (!descriptor || typeof descriptor !== 'object') fail(resolved, `buffer ${index} is not an object`);
    const value = loadBufferBytes(resolved, path.dirname(resolved), descriptor, index, binary);
    const declared = nonNegativeInteger(descriptor.byteLength, `buffer ${index}.byteLength`);
    if (value.length < declared) fail(resolved, `buffer ${index} is shorter than byteLength`);
    return value;
  });
  return {
    filePath: resolved,
    directory: path.dirname(resolved),
    format,
    json,
    buffers,
    sourceBytes: fileBytes.length + buffers.reduce((sum, value, index) => {
      if (format === 'glb' && index === 0 && binary) return sum;
      return sum + value.length;
    }, 0),
  };
}

function bufferViewBytes(parsed, viewIndex) {
  const view = parsed.json.bufferViews?.[viewIndex];
  if (!view || typeof view !== 'object') fail(parsed.filePath, `bufferView ${viewIndex} is missing`);
  const bufferIndex = nonNegativeInteger(view.buffer ?? 0, `bufferView ${viewIndex}.buffer`);
  const source = parsed.buffers[bufferIndex];
  if (!source) fail(parsed.filePath, `bufferView ${viewIndex} references missing buffer ${bufferIndex}`);
  const extension = view.extensions?.EXT_meshopt_compression || view.extensions?.KHR_meshopt_compression;
  if (extension) {
    const compressedBufferIndex = nonNegativeInteger(extension.buffer ?? 0, `bufferView ${viewIndex} meshopt buffer`);
    const compressedSource = parsed.buffers[compressedBufferIndex];
    const compressedOffset = nonNegativeInteger(extension.byteOffset ?? 0, `bufferView ${viewIndex} meshopt byteOffset`);
    const compressedLength = nonNegativeInteger(extension.byteLength, `bufferView ${viewIndex} meshopt byteLength`);
    const count = nonNegativeInteger(extension.count, `bufferView ${viewIndex} meshopt count`);
    const stride = nonNegativeInteger(extension.byteStride, `bufferView ${viewIndex} meshopt byteStride`);
    if (!compressedSource || compressedOffset + compressedLength > compressedSource.length || stride === 0) {
      fail(parsed.filePath, `bufferView ${viewIndex} meshopt data is out of range`);
    }
    if (!['ATTRIBUTES', 'TRIANGLES', 'INDICES'].includes(extension.mode)) {
      fail(parsed.filePath, `bufferView ${viewIndex} has an unsupported meshopt mode`);
    }
    if (!MeshoptDecoder.supported) fail(parsed.filePath, 'meshopt decoder is unavailable');
    return MeshoptDecoder.ready.then(() => {
      const target = new Uint8Array(count * stride);
      MeshoptDecoder.decodeGltfBuffer(
        target,
        count,
        stride,
        new Uint8Array(compressedSource.buffer, compressedSource.byteOffset + compressedOffset, compressedLength),
        extension.mode,
        extension.filter,
      );
      return { bytes: Buffer.from(target), stride, byteLength: target.length };
    });
  }
  const offset = nonNegativeInteger(view.byteOffset ?? 0, `bufferView ${viewIndex}.byteOffset`);
  const length = nonNegativeInteger(view.byteLength, `bufferView ${viewIndex}.byteLength`);
  if (offset + length > source.length) fail(parsed.filePath, `bufferView ${viewIndex} is out of range`);
  return Promise.resolve({ bytes: source.subarray(offset, offset + length), stride: view.byteStride || null, byteLength: length });
}

function mimeFromExtension(uri) {
  const extension = path.extname(String(uri).split(/[?#]/, 1)[0]).toLowerCase();
  if (extension === '.png') return 'image/png';
  if (extension === '.jpg' || extension === '.jpeg') return 'image/jpeg';
  if (extension === '.webp') return 'image/webp';
  if (extension === '.ktx2') return 'image/ktx2';
  return null;
}

function mimeFromBytes(bytes) {
  if (bytes.length >= 8 && bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) return 'image/png';
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'image/jpeg';
  if (bytes.length >= 12 && bytes.toString('ascii', 0, 4) === 'RIFF' && bytes.toString('ascii', 8, 12) === 'WEBP') return 'image/webp';
  return null;
}

function imageExtension(mimeType) {
  if (mimeType === 'image/png') return 'png';
  if (mimeType === 'image/jpeg' || mimeType === 'image/jpg') return 'jpg';
  if (mimeType === 'image/webp') return 'webp';
  return 'bin';
}

async function loadImageResources(parsed) {
  const images = Array.isArray(parsed.json.images) ? parsed.json.images : [];
  const resources = [];
  for (let imageIndex = 0; imageIndex < images.length; imageIndex += 1) {
    const descriptor = images[imageIndex];
    let bytes = null;
    let failure = null;
    try {
      if (descriptor?.bufferView != null) {
        const view = await bufferViewBytes(
          parsed,
          nonNegativeInteger(descriptor.bufferView, `image ${imageIndex}.bufferView`),
        );
        bytes = Buffer.from(view.bytes);
      } else if (typeof descriptor?.uri === 'string') {
        if (descriptor.uri.startsWith('data:')) bytes = decodeDataUri(descriptor.uri, parsed.filePath);
        else bytes = fs.readFileSync(resolveExternalUri(parsed.directory, descriptor.uri, parsed.filePath));
      } else {
        failure = 'image has neither URI nor bufferView';
      }
    } catch (error) {
      failure = error.message;
    }
    const declaredMime = typeof descriptor?.mimeType === 'string' ? descriptor.mimeType.toLowerCase() : null;
    const dataUriMime = typeof descriptor?.uri === 'string'
      ? (/^data:([^;,]+)/i.exec(descriptor.uri)?.[1] || null)?.toLowerCase()
      : null;
    const mimeType = declaredMime || dataUriMime || mimeFromExtension(descriptor?.uri) || (bytes && mimeFromBytes(bytes));
    if (!failure && (!bytes || bytes.length === 0)) failure = 'image data is empty';
    if (!failure && (!mimeType || !SUPPORTED_IMAGE_MIMES.has(mimeType))) {
      failure = `image MIME type is not supported by the GLTFLoader path: ${mimeType || 'unknown'}`;
    }
    resources.push({
      sourceImageIndex: imageIndex,
      name: descriptor?.name || `image_${imageIndex}`,
      mimeType: mimeType || null,
      extension: imageExtension(mimeType),
      available: !failure,
      reason: failure,
      bytes: failure ? null : bytes,
    });
  }
  return resources;
}

function readComponent(dataView, offset, componentType) {
  const component = COMPONENT_TYPES[componentType];
  if (!component) throw new Error(`unsupported component type ${componentType}`);
  return dataView[component.read](offset, component.bytes === 1 ? undefined : true);
}

function normalizedValue(value, componentType) {
  if (componentType === 5120) return Math.max(-1, value / 127);
  if (componentType === 5121) return value / 255;
  if (componentType === 5122) return Math.max(-1, value / 32767);
  if (componentType === 5123) return value / 65535;
  return value;
}

async function readAccessor(parsed, accessorIndex) {
  const accessor = parsed.json.accessors?.[accessorIndex];
  if (!accessor || typeof accessor !== 'object') fail(parsed.filePath, `accessor ${accessorIndex} is missing`);
  const component = COMPONENT_TYPES[accessor.componentType];
  const components = TYPE_COMPONENTS[accessor.type];
  if (!component || !components) fail(parsed.filePath, `accessor ${accessorIndex} has an unsupported format`);
  const count = nonNegativeInteger(accessor.count, `accessor ${accessorIndex}.count`);
  if (accessor.sparse) fail(parsed.filePath, `sparse accessor ${accessorIndex} is not supported in G1`);
  const elementBytes = component.bytes * components;
  let bytes = Buffer.alloc(count * elementBytes);
  let stride = elementBytes;
  let accessorOffset = 0;
  if (accessor.bufferView != null) {
    const viewResult = await bufferViewBytes(parsed, nonNegativeInteger(accessor.bufferView, `accessor ${accessorIndex}.bufferView`));
    bytes = viewResult.bytes;
    stride = viewResult.stride || elementBytes;
    accessorOffset = nonNegativeInteger(accessor.byteOffset ?? 0, `accessor ${accessorIndex}.byteOffset`);
  }
  if (stride < elementBytes) fail(parsed.filePath, `accessor ${accessorIndex} has a short byteStride`);
  const lastByte = count === 0 ? accessorOffset : accessorOffset + (count - 1) * stride + elementBytes;
  if (lastByte > bytes.length) fail(parsed.filePath, `accessor ${accessorIndex} is out of range`);
  const dataView = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(count * components);
  for (let item = 0; item < count; item += 1) {
    const start = accessorOffset + item * stride;
    for (let componentIndex = 0; componentIndex < components; componentIndex += 1) {
      const raw = readComponent(dataView, start + componentIndex * component.bytes, accessor.componentType);
      values[item * components + componentIndex] = accessor.normalized
        ? normalizedValue(raw, accessor.componentType)
        : raw;
    }
  }
  return {
    accessor,
    values,
    count,
    components,
    componentType: accessor.componentType,
    normalized: Boolean(accessor.normalized),
  };
}

async function readIndices(parsed, accessorIndex, vertexCount) {
  const result = await readAccessor(parsed, accessorIndex);
  if (result.components !== 1 || result.accessor.type !== 'SCALAR' || result.normalized) {
    fail(parsed.filePath, `index accessor ${accessorIndex} must be an unnormalized SCALAR`);
  }
  const indices = Uint32Array.from(result.values, (value) => {
    if (!Number.isInteger(value) || value < 0 || value >= vertexCount) {
      fail(parsed.filePath, `index accessor ${accessorIndex} contains an invalid index`);
    }
    return value;
  });
  if (indices.length % 3 !== 0) fail(parsed.filePath, 'triangle index count must be divisible by three');
  return indices;
}

function textureInfo(json, materialIndex, imageResources) {
  const material = materialIndex == null ? null : json.materials?.[materialIndex];
  const textureDef = material?.pbrMetallicRoughness?.baseColorTexture;
  if (!textureDef) return null;
  const textureIndex = Number(textureDef.index);
  const texture = Number.isInteger(textureIndex) && textureIndex >= 0 ? json.textures?.[textureIndex] : null;
  const sourceIndex = texture?.source ?? texture?.extensions?.KHR_texture_basisu?.source;
  const imageIndex = Number(sourceIndex);
  const image = Number.isInteger(imageIndex) && imageIndex >= 0 ? imageResources?.[imageIndex] : null;
  const samplerIndex = texture?.sampler == null ? null : Number(texture.sampler);
  const samplerSource = Number.isInteger(samplerIndex) && samplerIndex >= 0
    ? (json.samplers?.[samplerIndex] || {})
    : {};
  const sampler = {};
  for (const name of ['magFilter', 'minFilter', 'wrapS', 'wrapT']) {
    if (samplerSource[name] != null) sampler[name] = Number(samplerSource[name]);
  }
  const transformSource = textureDef.extensions?.KHR_texture_transform;
  const textureTransform = transformSource && typeof transformSource === 'object'
    ? {
      ...(Array.isArray(transformSource.offset) ? { offset: transformSource.offset.map(Number) } : {}),
      ...(Array.isArray(transformSource.scale) ? { scale: transformSource.scale.map(Number) } : {}),
      ...(transformSource.rotation != null ? { rotation: Number(transformSource.rotation) } : {}),
      ...(transformSource.texCoord != null ? { texCoord: Number(transformSource.texCoord) } : {}),
    }
    : null;
  const texCoord = Number(textureDef.texCoord ?? 0);
  const coordinateAvailable = texCoord === 0 || texCoord === 1;
  return {
    textureIndex: Number.isInteger(textureIndex) && textureIndex >= 0 ? textureIndex : null,
    imageIndex: Number.isInteger(imageIndex) && imageIndex >= 0 ? imageIndex : null,
    texCoord,
    sampler,
    textureTransform,
    available: Boolean(image?.available && coordinateAvailable),
    reason: !coordinateAvailable
      ? `baseColor texture uses unsupported TEXCOORD_${texCoord}`
      : image?.available ? null : image?.reason || 'baseColor texture image is unavailable',
    mimeType: image?.mimeType || null,
    extension: image?.extension || 'bin',
    name: texture?.name || image?.name || null,
  };
}

function alphaInfo(json, materialIndex, imageResources) {
  const baseColorTexture = textureInfo(json, materialIndex, imageResources);
  const base = {
    materialIndex: materialIndex == null ? null : materialIndex,
    baseColorTexture,
    texturePreserved: !baseColorTexture || baseColorTexture.available,
  };
  if (materialIndex == null) {
    return {
      ...base,
      alphaMode: 'UNKNOWN',
      alphaCutoff: null,
      occluder: false,
      staticPvsEligible: true,
      alwaysResident: false,
      reason: 'missing-material',
    };
  }
  const material = json.materials?.[materialIndex];
  if (!material || typeof material !== 'object') {
    return {
      ...base,
      alphaMode: 'UNKNOWN',
      alphaCutoff: null,
      occluder: false,
      staticPvsEligible: true,
      alwaysResident: false,
      reason: 'missing-material-record',
    };
  }
  const alphaMode = String(material.alphaMode || 'OPAQUE').toUpperCase();
  if (!KNOWN_ALPHA_MODES.has(alphaMode)) {
    return {
      ...base,
      alphaMode: 'UNKNOWN',
      alphaCutoff: null,
      occluder: false,
      staticPvsEligible: true,
      alwaysResident: false,
      reason: 'unknown-alpha-mode',
    };
  }
  const transmissionExtension = Object.keys(material.extensions || {}).find((name) => TRANSPARENCY_EXTENSIONS.has(name));
  const maskTextureUnavailable = alphaMode === 'MASK' && baseColorTexture && !baseColorTexture.available;
  return {
    ...base,
    alphaMode,
    alphaCutoff: alphaMode === 'MASK' ? Number(material.alphaCutoff ?? 0.5) : null,
    occluder: alphaMode === 'OPAQUE',
    staticPvsEligible: alphaMode !== 'BLEND' && !maskTextureUnavailable,
    alwaysResident: alphaMode === 'BLEND',
    reason: maskTextureUnavailable
      ? 'mask-baseColor-alpha-texture-unavailable'
      : alphaMode === 'OPAQUE' ? 'opaque' : alphaMode === 'MASK' ? 'mask-not-occluder' : 'blend-always-resident',
    transparencyExtension: transmissionExtension || null,
  };
}

function materialForOutput(json, materialIndex, info) {
  const source = materialIndex == null ? null : json.materials?.[materialIndex];
  const pbr = source?.pbrMetallicRoughness || {};
  return {
    name: source?.name || `material_${materialIndex ?? 'default'}`,
    alphaMode: info.alphaMode === 'UNKNOWN' ? 'OPAQUE' : info.alphaMode,
    ...(info.alphaMode === 'MASK' ? { alphaCutoff: info.alphaCutoff ?? 0.5 } : {}),
    doubleSided: Boolean(source?.doubleSided),
    pbrMetallicRoughness: {
      baseColorFactor: Array.isArray(pbr.baseColorFactor) && pbr.baseColorFactor.length === 4
        ? pbr.baseColorFactor.map(Number)
        : [1, 1, 1, 1],
      metallicFactor: Number(pbr.metallicFactor ?? 1),
      roughnessFactor: Number(pbr.roughnessFactor ?? 1),
    },
  };
}

function primitiveReason(node, mesh, primitive, nodeIndex, primitiveIndex, animationNodes) {
  if (node.skin != null) return 'skinned-node';
  if (animationNodes.has(nodeIndex)) return 'animated-node';
  if (node.weights != null || mesh.weights != null || primitive.targets) return 'morph-target-or-weights';
  if (Number(primitive.mode ?? TRIANGLES_MODE) !== TRIANGLES_MODE) return 'non-triangle-primitive';
  if (primitive.attributes?.POSITION == null) return 'missing-position';
  return null;
}

function nodeSegment(node, nodeIndex) {
  const name = typeof node.name === 'string' && node.name ? node.name : `node_${nodeIndex}`;
  return `${name}[${nodeIndex}]`;
}

function sceneRoots(json) {
  const sceneIndex = nonNegativeInteger(json.scene ?? 0, 'scene index');
  const scene = json.scenes?.[sceneIndex];
  if (!scene || !Array.isArray(scene.nodes)) throw new Error('default scene is missing or has no node list');
  return { sceneIndex, roots: scene.nodes.map((value, index) => nonNegativeInteger(value, `scene.nodes[${index}]`)) };
}

function animationNodeSet(json) {
  const result = new Set();
  for (const animation of json.animations || []) {
    for (const channel of animation.channels || []) {
      const node = channel.target?.node;
      if (node != null) result.add(nonNegativeInteger(node, 'animation target node'));
    }
  }
  return result;
}

function cloneAttribute(values, components, type) {
  return { values, components, type };
}

async function readPrimitive(
  parsed,
  node,
  world,
  nodeIndex,
  nodePath,
  mesh,
  meshIndex,
  primitive,
  primitiveIndex,
  animationNodes,
  imageResources,
) {
  const material = alphaInfo(parsed.json, primitive.material, imageResources);
  const reason = primitiveReason(node, mesh, primitive, nodeIndex, primitiveIndex, animationNodes);
  const sourcePositionAccessor = primitive.attributes?.POSITION;
  const positionResult = sourcePositionAccessor == null
    ? null
    : await readAccessor(parsed, nonNegativeInteger(sourcePositionAccessor, 'POSITION accessor'));
  const sourceVertexCount = positionResult?.count || 0;
  const sourceTriangleCount = primitive.indices == null
    ? Math.floor(sourceVertexCount / 3)
    : Number(parsed.json.accessors?.[primitive.indices]?.count || 0) / 3;
  const sourceInfo = {
    sourceNodeIndex: nodeIndex,
    sourceNodePath: nodePath,
    sourceMeshIndex: meshIndex,
    sourcePrimitiveIndex: primitiveIndex,
    sourceMaterialIndex: primitive.material ?? null,
    sourceVertexCount,
    sourceTriangleCount,
    material,
  };
  if (material.alphaMode === 'BLEND') {
    return {
      sourceInfo,
      excluded: {
        ...sourceInfo,
        reason: 'blend-always-resident',
        alwaysResident: true,
        staticPvsEligible: false,
      },
    };
  }
  if (material.alphaMode === 'MASK' && material.baseColorTexture && !material.baseColorTexture.available) {
    return {
      sourceInfo,
      excluded: {
        ...sourceInfo,
        reason: 'mask-baseColor-alpha-texture-unavailable',
        alwaysResident: false,
        staticPvsEligible: false,
      },
    };
  }
  if (reason) return { sourceInfo, excluded: { ...sourceInfo, reason } };
  if (!positionResult || positionResult.components !== 3 || positionResult.accessor.type !== 'VEC3') {
    return { sourceInfo, excluded: { ...sourceInfo, reason: 'invalid-position-accessor' } };
  }
  const positions = transformPositions(positionResult.values, world);
  const indices = primitive.indices == null
    ? Uint32Array.from({ length: sourceVertexCount }, (_value, index) => index)
    : await readIndices(parsed, nonNegativeInteger(primitive.indices, 'index accessor'), sourceVertexCount);
  const attributes = {};
  for (const name of ['NORMAL', 'TEXCOORD_0', 'TEXCOORD_1', 'COLOR_0', 'TANGENT']) {
    const accessorIndex = primitive.attributes?.[name];
    if (accessorIndex == null) continue;
    const attribute = await readAccessor(parsed, nonNegativeInteger(accessorIndex, `${name} accessor`));
    if (name === 'NORMAL' && attribute.components === 3) {
      attributes[name] = cloneAttribute(transformDirections(attribute.values, world, `node ${nodeIndex} normal`), 3, 'VEC3');
    } else if (name === 'TANGENT' && attribute.components === 4) {
      const xyz = new Float32Array(attribute.count * 3);
      for (let index = 0; index < attribute.count; index += 1) xyz.set(attribute.values.slice(index * 4, index * 4 + 3), index * 3);
      const transformed = transformDirections(xyz, world, `node ${nodeIndex} tangent`, false);
      const tangent = new Float32Array(attribute.count * 4);
      for (let index = 0; index < attribute.count; index += 1) {
        tangent.set(transformed.slice(index * 3, index * 3 + 3), index * 4);
        tangent[index * 4 + 3] = attribute.values[index * 4 + 3];
      }
      attributes[name] = cloneAttribute(tangent, 4, 'VEC4');
    } else if (
      ((name === 'TEXCOORD_0' || name === 'TEXCOORD_1') && attribute.components === 2)
      || (name === 'COLOR_0' && (attribute.components === 3 || attribute.components === 4))
    ) {
      attributes[name] = cloneAttribute(attribute.values, attribute.components, attribute.accessor.type);
    }
  }
  return {
    renderable: {
      ...sourceInfo,
      nodeName: node.name || `node_${nodeIndex}`,
      worldMatrix: world,
      positions,
      indices,
      attributes,
      vertexCount: positions.length / 3,
      triangleCount: indices.length / 3,
      materialOutput: materialForOutput(parsed.json, primitive.material, material),
      materialTexture: material.baseColorTexture?.available ? material.baseColorTexture : null,
      bounds: boundsForPositions(positions),
    },
  };
}

export async function readGltfScene(filePath) {
  const parsed = parseSource(filePath);
  const { sceneIndex, roots } = sceneRoots(parsed.json);
  const animations = Array.isArray(parsed.json.animations) ? parsed.json.animations.length : 0;
  const skins = Array.isArray(parsed.json.skins) ? parsed.json.skins.length : 0;
  const animationNodes = animationNodeSet(parsed.json);
  const imageResources = await loadImageResources(parsed);
  const renderables = [];
  const excluded = [];
  const nodes = [];
  const visited = new Set();

  async function visit(nodeIndex, parentMatrix, parentPath, stack) {
    if (stack.has(nodeIndex)) throw new Error(`${parsed.filePath}: node graph contains a cycle at node ${nodeIndex}`);
    const node = parsed.json.nodes?.[nodeIndex];
    if (!node || typeof node !== 'object') fail(parsed.filePath, `node ${nodeIndex} is missing`);
    const world = multiplyMatrix(parentMatrix, nodeMatrix(node, `node ${nodeIndex}`));
    const currentPath = parentPath ? `${parentPath}/${nodeSegment(node, nodeIndex)}` : nodeSegment(node, nodeIndex);
    nodes.push({
      nodeIndex,
      nodePath: currentPath,
      name: node.name || `node_${nodeIndex}`,
      meshIndex: node.mesh ?? null,
      dynamic: Boolean(node.skin != null || animationNodes.has(nodeIndex) || node.weights != null),
    });
    const meshIndex = node.mesh == null ? null : nonNegativeInteger(node.mesh, `node ${nodeIndex}.mesh`);
    if (meshIndex != null) {
      const mesh = parsed.json.meshes?.[meshIndex];
      if (!mesh || !Array.isArray(mesh.primitives)) fail(parsed.filePath, `mesh ${meshIndex} is missing or invalid`);
      for (let primitiveIndex = 0; primitiveIndex < mesh.primitives.length; primitiveIndex += 1) {
        const primitive = mesh.primitives[primitiveIndex];
        const result = await readPrimitive(
          parsed,
          { ...node, _world: world },
          world,
          nodeIndex,
          currentPath,
          mesh,
          meshIndex,
          primitive,
          primitiveIndex,
          animationNodes,
          imageResources,
        );
        if (result.renderable) {
          result.renderable.worldMatrix = world;
          renderables.push(result.renderable);
        } else if (result.excluded) {
          excluded.push(result.excluded);
        }
      }
    }
    const nextStack = new Set(stack);
    nextStack.add(nodeIndex);
    for (const child of node.children || []) {
      await visit(nonNegativeInteger(child, `node ${nodeIndex}.children`), world, currentPath, nextStack);
    }
  }

  for (const root of roots) await visit(root, identityMatrix(), '', new Set());
  const sourceBounds = boundsForPositions(Float32Array.from(renderables.flatMap((item) => Array.from(item.positions))));
  const primitiveCount = (parsed.json.meshes || []).reduce(
    (sum, mesh) => sum + (Array.isArray(mesh.primitives) ? mesh.primitives.length : 0),
    0,
  );
  return {
    source: {
      filePath: parsed.filePath,
      format: parsed.format,
      sourceBytes: parsed.sourceBytes,
      sceneIndex,
      nodeCount: Array.isArray(parsed.json.nodes) ? parsed.json.nodes.length : 0,
      meshCount: Array.isArray(parsed.json.meshes) ? parsed.json.meshes.length : 0,
      primitiveCount,
      animationCount: animations,
      skinCount: skins,
      extensionsUsed: Array.isArray(parsed.json.extensionsUsed) ? [...parsed.json.extensionsUsed].sort() : [],
      coordinateUnit: parsed.json.asset?.extras?.unit || parsed.json.extras?.unit || 'unspecified',
      imageCount: imageResources.length,
      availableImageCount: imageResources.filter((image) => image.available).length,
      unavailableImageCount: imageResources.filter((image) => !image.available).length,
    },
    json: parsed.json,
    nodes,
    renderables,
    excluded,
    sceneBounds: sourceBounds,
    imageResources,
  };
}

export const readGltf = readGltfScene;
export { boundsForPositions, identityMatrix, multiplyMatrix };
