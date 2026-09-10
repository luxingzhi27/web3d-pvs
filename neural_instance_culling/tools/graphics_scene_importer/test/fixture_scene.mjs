import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

function pad4(value) {
  return (value + 3) & ~3;
}

function appendTyped(parts, state, values) {
  const offset = pad4(state.length);
  if (offset > state.length) parts.push(Buffer.alloc(offset - state.length));
  const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
  parts.push(bytes);
  state.length = offset + bytes.length;
  return { byteOffset: offset, byteLength: bytes.length };
}

function gridGeometry(columns = 20, rows = 20) {
  const positions = [];
  for (let row = 0; row <= rows; row += 1) {
    for (let column = 0; column <= columns; column += 1) positions.push(column, row, 0);
  }
  const indices = [];
  const rowWidth = columns + 1;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const topLeft = row * rowWidth + column;
      const topRight = topLeft + 1;
      const bottomLeft = topLeft + rowWidth;
      const bottomRight = bottomLeft + 1;
      indices.push(topLeft, bottomLeft, topRight, topRight, bottomLeft, bottomRight);
    }
  }
  return { positions: Float32Array.from(positions), indices: Uint16Array.from(indices) };
}

function triangleGeometry(z) {
  return {
    positions: new Float32Array([-1, 0, z, 1, 0, z, 0, 1, z]),
    indices: new Uint16Array([0, 1, 2]),
  };
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const typeBytes = Buffer.from(type, 'ascii');
  const payload = Buffer.concat([typeBytes, data]);
  const chunk = Buffer.alloc(12 + data.length);
  chunk.writeUInt32BE(data.length, 0);
  payload.copy(chunk, 4);
  chunk.writeUInt32BE(crc32(payload), 8 + data.length);
  return chunk;
}

function alphaPng() {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(1, 0);
  header.writeUInt32BE(1, 4);
  header[8] = 8;
  header[9] = 6;
  const pixels = zlib.deflateSync(Buffer.from([0, 255, 255, 255, 128]));
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk('IHDR', header),
    pngChunk('IDAT', pixels),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

function buildFixtureJson(binaryLength, views, imageView) {
  return {
    asset: {
      version: '2.0',
      generator: 'slm-graphics-scene-importer-g1-fixture',
      extras: { unit: 'meter' },
    },
    scene: 0,
    scenes: [{ name: 'g1_fixture', nodes: [0] }],
    nodes: [
      { name: 'FixtureRoot', translation: [10, 20, 30], children: [1, 2, 3, 4, 5] },
      { name: 'OpaqueGrid', mesh: 0, translation: [1, 2, 3] },
      { name: 'MaskTriangle', mesh: 1, translation: [-2, 0, 0] },
      { name: 'BlendTriangle', mesh: 2, translation: [2, 0, 0] },
      { name: 'OpaqueGridCopy', mesh: 0, translation: [30, 0, 0] },
      { name: 'MaskTriangleCopy', mesh: 1, translation: [4, 0, 0] },
    ],
    meshes: [
      {
        name: 'large_grid',
        primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0, mode: 4 }],
      },
      {
        name: 'mask_triangle',
        primitives: [{ attributes: { POSITION: 2 }, indices: 3, material: 1, mode: 4 }],
      },
      {
        name: 'blend_triangle',
        primitives: [{ attributes: { POSITION: 4 }, indices: 5, material: 2, mode: 4 }],
      },
    ],
    materials: [
      {
        name: 'Opaque',
        pbrMetallicRoughness: { baseColorFactor: [0.7, 0.7, 0.7, 1] },
      },
      {
        name: 'Masked',
        alphaMode: 'MASK',
        alphaCutoff: 0.25,
        pbrMetallicRoughness: {
          baseColorFactor: [0.2, 0.8, 0.2, 1],
          baseColorTexture: { index: 0, texCoord: 0 },
        },
      },
      {
        name: 'Blended',
        alphaMode: 'BLEND',
        pbrMetallicRoughness: { baseColorFactor: [0.2, 0.2, 1, 0.5] },
      },
    ],
    samplers: [{ magFilter: 9729, minFilter: 9729, wrapS: 10497, wrapT: 10497 }],
    textures: [{ name: 'mask_alpha_texture', sampler: 0, source: 0 }],
    images: [{ name: 'mask_alpha', bufferView: imageView, mimeType: 'image/png' }],
    buffers: [{ byteLength: binaryLength, uri: 'fixture.bin' }],
    bufferViews: views,
    accessors: [
      { bufferView: 0, componentType: 5126, count: 441, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: 2400, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 3, componentType: 5123, count: 3, type: 'SCALAR' },
      { bufferView: 4, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 5, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
  };
}

function writeGlb(filePath, json, binary) {
  const rawJson = Buffer.from(JSON.stringify({ ...json, buffers: [{ byteLength: binary.length }] }), 'utf8');
  const jsonChunk = Buffer.alloc(pad4(rawJson.length), 0x20);
  rawJson.copy(jsonChunk);
  const binaryChunkLength = pad4(binary.length);
  const totalLength = 12 + 8 + jsonChunk.length + 8 + binaryChunkLength;
  const output = Buffer.alloc(totalLength);
  output.writeUInt32LE(0x46546c67, 0);
  output.writeUInt32LE(2, 4);
  output.writeUInt32LE(totalLength, 8);
  output.writeUInt32LE(jsonChunk.length, 12);
  output.writeUInt32LE(0x4e4f534a, 16);
  jsonChunk.copy(output, 20);
  const binaryHeader = 20 + jsonChunk.length;
  output.writeUInt32LE(binaryChunkLength, binaryHeader);
  output.writeUInt32LE(0x004e4942, binaryHeader + 4);
  binary.copy(output, binaryHeader + 8);
  fs.writeFileSync(filePath, output);
}

export function createFixtureScene(directory) {
  const root = path.resolve(directory);
  fs.mkdirSync(root, { recursive: true });
  const parts = [];
  const state = { length: 0 };
  const grid = gridGeometry();
  const mask = triangleGeometry(-1);
  const blend = triangleGeometry(-2);
  const image = alphaPng();
  const viewData = [grid.positions, grid.indices, mask.positions, mask.indices, blend.positions, blend.indices, image];
  const views = viewData.map((values, index) => {
    const range = appendTyped(parts, state, values);
    return {
      buffer: 0,
      byteOffset: range.byteOffset,
      byteLength: range.byteLength,
      target: index % 2 === 0 ? 34962 : 34963,
    };
  });
  const binary = Buffer.concat(parts);
  const json = buildFixtureJson(binary.length, views, 6);
  const glbPath = path.join(root, 'fixture.glb');
  const gltfPath = path.join(root, 'fixture.gltf');
  const binPath = path.join(root, 'fixture.bin');
  writeGlb(glbPath, json, binary);
  fs.writeFileSync(binPath, binary);
  fs.writeFileSync(gltfPath, `${JSON.stringify(json, null, 2)}\n`, 'utf8');
  return { glbPath, gltfPath, binPath, json, binaryBytes: binary.length };
}

function nonClosedShellGeometry() {
  const positions = [];
  const indices = [];
  const addWall = (corners) => {
    const base = positions.length / 3;
    positions.push(...corners);
    indices.push(base, base + 2, base + 1, base + 1, base + 2, base + 3);
  };
  addWall([2, 0, 2, 2, 4, 2, 2, 0, 14, 2, 4, 14]);
  addWall([14, 0, 14, 14, 4, 14, 14, 0, 2, 14, 4, 2]);
  addWall([14, 0, 2, 14, 4, 2, 2, 0, 2, 2, 4, 2]);
  addWall([2, 0, 14, 2, 4, 14, 14, 0, 14, 14, 4, 14]);
  return {
    positions: Float32Array.from(positions),
    indices: Uint16Array.from(indices),
  };
}

export function createNonClosedShellScene(directory) {
  const root = path.resolve(directory);
  fs.mkdirSync(root, { recursive: true });
  const geometry = nonClosedShellGeometry();
  const parts = [];
  const state = { length: 0 };
  const positionView = appendTyped(parts, state, geometry.positions);
  const indexView = appendTyped(parts, state, geometry.indices);
  const binary = Buffer.concat(parts);
  const json = {
    asset: {
      version: '2.0',
      generator: 'slm-graphics-scene-importer-g1-shell-fixture',
      extras: { unit: 'meter' },
    },
    scene: 0,
    scenes: [{ name: 'non_closed_shell', nodes: [0] }],
    nodes: [{ name: 'NonClosedShell', mesh: 0 }],
    meshes: [{
      name: 'non_closed_shell',
      primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0, mode: 4 }],
    }],
    materials: [{
      name: 'OpaqueShell',
      pbrMetallicRoughness: { baseColorFactor: [0.6, 0.6, 0.6, 1] },
    }],
    buffers: [{ byteLength: binary.length, uri: 'shell.bin' }],
    bufferViews: [
      { buffer: 0, byteOffset: positionView.byteOffset, byteLength: positionView.byteLength, target: 34962 },
      { buffer: 0, byteOffset: indexView.byteOffset, byteLength: indexView.byteLength, target: 34963 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: geometry.positions.length / 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: geometry.indices.length, type: 'SCALAR' },
    ],
  };
  const glbPath = path.join(root, 'non_closed_shell.glb');
  const gltfPath = path.join(root, 'non_closed_shell.gltf');
  const binPath = path.join(root, 'shell.bin');
  writeGlb(glbPath, json, binary);
  fs.writeFileSync(binPath, binary);
  fs.writeFileSync(gltfPath, `${JSON.stringify(json, null, 2)}\n`, 'utf8');
  return { glbPath, gltfPath, binPath, json, binaryBytes: binary.length };
}
