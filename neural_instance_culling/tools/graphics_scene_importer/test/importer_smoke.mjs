#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { auditOutputAssets } from '../audit_scene.mjs';
import { PARTITION_SCHEMA, partitionScene } from '../partition_units.mjs';
import { readGltfScene } from '../read_gltf.mjs';
import { writeSlmScene } from '../write_slm_scene.mjs';
import { createFixtureScene } from './fixture_scene.mjs';

function unitRecordSignature(unit) {
  return {
    unitId: unit.unitId,
    sourceNodePath: unit.sourceNodePath,
    sourcePrimitiveIndex: unit.sourcePrimitiveIndex,
    sourceTriangleIndices: unit.sourceTriangleIndices,
    sourceComponentOrdinals: unit.sourceComponentOrdinals,
    partitionSchema: unit.partitionSchema,
    componentCount: unit.componentCount,
    sourceComponentCount: unit.sourceComponentCount,
    partitionReason: unit.partitionReason,
    oversizedComponentSplit: unit.oversizedComponentSplit,
    oversize: unit.oversize,
    oversizeReason: unit.oversizeReason,
    triangleCount: unit.triangleCount,
    vertexCount: unit.vertexCount,
    compressedGeometryBytes: unit.compression.encodedGeometryBytes,
  };
}

function unitSignature(partition) {
  return partition.units.map((unit) => unitRecordSignature(unit));
}

function largeGridRenderable(columns = 256, rows = 256) {
  const positions = new Float32Array((columns + 1) * (rows + 1) * 3);
  let position = 0;
  for (let row = 0; row <= rows; row += 1) {
    for (let column = 0; column <= columns; column += 1) {
      positions[position] = column;
      positions[position + 1] = row;
      positions[position + 2] = 0;
      position += 3;
    }
  }
  const indices = new Uint32Array(columns * rows * 6);
  const rowWidth = columns + 1;
  let index = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const topLeft = row * rowWidth + column;
      const topRight = topLeft + 1;
      const bottomLeft = topLeft + rowWidth;
      const bottomRight = bottomLeft + 1;
      indices[index] = topLeft;
      indices[index + 1] = bottomLeft;
      indices[index + 2] = topRight;
      indices[index + 3] = topRight;
      indices[index + 4] = bottomLeft;
      indices[index + 5] = bottomRight;
      index += 6;
    }
  }
  return {
    positions,
    indices,
    attributes: {},
    material: { alphaMode: 'OPAQUE' },
    sourceNodePath: 'large-synthetic-grid',
    sourcePrimitiveIndex: 0,
    sourceMeshIndex: 0,
  };
}

function memoryMiB(snapshot, field) {
  return Number((snapshot[field] / (1024 * 1024)).toFixed(2));
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-graphics-scene-importer-g1-'));
try {
  const fixture = createFixtureScene(path.join(root, 'fixture'));
  const glbScene = await readGltfScene(fixture.glbPath);
  const gltfScene = await readGltfScene(fixture.gltfPath);
  assert.equal(glbScene.source.format, 'glb');
  assert.equal(gltfScene.source.format, 'gltf');
  assert.equal(glbScene.source.nodeCount, 6);
  assert.equal(glbScene.renderables.length, 4);
  assert.equal(glbScene.excluded.length, 1);
  assert.equal(glbScene.excluded[0].reason, 'blend-always-resident');
  assert.equal(glbScene.excluded[0].alwaysResident, true);
  assert.deepEqual(glbScene.renderables[0].positions.slice(0, 3), new Float32Array([11, 22, 33]));
  const maskSource = glbScene.renderables.find((item) => item.material.alphaMode === 'MASK');
  assert.equal(maskSource.material.occluder, false);
  assert.equal(maskSource.material.baseColorTexture.available, true);
  assert.deepEqual(
    unitSignature(await partitionScene(glbScene, { targetBytes: 512 })),
    unitSignature(await partitionScene(gltfScene, { targetBytes: 512 })),
  );

  const partition = await partitionScene(glbScene, { targetBytes: 512 });
  assert.equal(partition.partitionSchema, PARTITION_SCHEMA);
  assert.equal(partition.partition.schema, PARTITION_SCHEMA);
  assert.equal(partition.partition.binCount, 16);
  assert.ok(partition.units.length > glbScene.renderables.length);
  assert.deepEqual(partition.units.map((unit) => unit.unitId), [...partition.units.keys()]);
  assert.equal(
    partition.units.reduce((sum, unit) => sum + unit.triangleCount, 0),
    glbScene.renderables.reduce((sum, item) => sum + item.triangleCount, 0),
  );
  assert.ok(partition.units.every((unit) => unit.compression.encodedGeometryBytes > 0));
  assert.ok(partition.units.every((unit) => unit.partitionSchema === PARTITION_SCHEMA));
  assert.ok(partition.units.every((unit) => unit.componentCount === 1));
  assert.ok(partition.units.every((unit) => unit.sourceComponentOrdinals.length === 1));
  assert.ok(partition.units.every((unit) => unit.oversize === false));
  assert.ok(partition.units.some((unit) => unit.partitionReason === 'oversized-connected-component-binned-sah'));
  assert.ok(partition.units.every((unit) => unit.compression.encodedGeometryBytes <= 512));
  assert.equal(partition.units.some((unit) => unit.material.alphaMode === 'BLEND'), false);

  const disconnectedRenderable = {
    positions: new Float32Array([
      0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
      10, 0, 0, 11, 0, 0, 11, 1, 0, 10, 1, 0,
    ]),
    indices: new Uint16Array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]),
    attributes: {},
    material: { alphaMode: 'OPAQUE' },
    sourceNodePath: 'Disconnected',
    sourcePrimitiveIndex: 0,
    sourceMeshIndex: 0,
  };
  const disconnected = await partitionScene({ renderables: [disconnectedRenderable] }, { targetBytes: 512 });
  assert.equal(disconnected.componentCount, 2);
  assert.equal(disconnected.units.length, 1);
  assert.deepEqual(disconnected.units[0].sourceComponentOrdinals, [0, 1]);
  assert.equal(disconnected.units[0].componentCount, 2);
  assert.equal(disconnected.units[0].sourceComponentCount, 2);
  assert.equal(disconnected.units[0].partitionReason, 'connected-component-binned-sah-pack');
  assert.equal(disconnected.units[0].oversizedComponentSplit, false);
  assert.ok(disconnected.units[0].compression.encodedGeometryBytes <= 512);

  const separatePrimitives = await partitionScene({
    renderables: [
      disconnectedRenderable,
      { ...disconnectedRenderable, sourcePrimitiveIndex: 1 },
    ],
  }, { targetBytes: 512 });
  assert.equal(separatePrimitives.units.length, 2);
  assert.deepEqual(
    separatePrimitives.units.map((unit) => unit.sourcePrimitiveIndex),
    [0, 1],
  );

  const oversize = await partitionScene({
    renderables: [{
      ...disconnectedRenderable,
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: new Uint16Array([0, 1, 2]),
    }],
  }, { targetBytes: 1 });
  assert.equal(oversize.units.length, 1);
  assert.equal(oversize.units[0].triangleCount, 1);
  assert.equal(oversize.units[0].oversize, true);
  assert.equal(oversize.units[0].oversizeReason, 'single-triangle');
  assert.equal(oversize.units[0].oversizedComponentSplit, false);

  const largePrimitive = largeGridRenderable();
  const largeTriangleCount = largePrimitive.indices.length / 3;
  const largeBefore = process.memoryUsage();
  const largeStart = process.hrtime.bigint();
  const largeFirst = await partitionScene({ renderables: [largePrimitive] });
  const largeElapsedMs = Number(process.hrtime.bigint() - largeStart) / 1e6;
  const largeAfter = process.memoryUsage();
  assert.equal(largeTriangleCount, 131072);
  assert.equal(largeFirst.componentCount, 1);
  assert.equal(
    largeFirst.units.reduce((sum, unit) => sum + unit.triangleCount, 0),
    largeTriangleCount,
  );
  assert.ok(largeFirst.units.length > 1);
  assert.ok(largeFirst.units.every((unit) => unit.componentCount === 1));
  assert.ok(largeFirst.units.every((unit) => unit.oversize === false));
  assert.ok(largeFirst.units.every((unit) => unit.oversizedComponentSplit === true));
  assert.ok(largeFirst.units.every((unit) => unit.compression.encodedGeometryBytes <= 128 * 1024));
  const largeStreamSignature = [];
  const largeSecond = await partitionScene({ renderables: [largePrimitive] }, {
    retainUnits: false,
    onUnit: (unit) => largeStreamSignature.push(unitRecordSignature(unit)),
  });
  assert.equal(largeSecond.units.length, 0);
  assert.equal(largeSecond.componentCount, 1);
  assert.deepEqual(unitSignature(largeFirst), largeStreamSignature);
  console.log(JSON.stringify({
    largePrimitiveSmoke: {
      triangles: largeTriangleCount,
      units: largeFirst.units.length,
      elapsedMs: Number(largeElapsedMs.toFixed(2)),
      rssObservedBeforeMiB: memoryMiB(largeBefore, 'rss'),
      rssObservedAfterMiB: memoryMiB(largeAfter, 'rss'),
      heapObservedAfterMiB: memoryMiB(largeAfter, 'heapUsed'),
      arrayBuffersObservedAfterMiB: memoryMiB(largeAfter, 'arrayBuffers'),
      deterministic: true,
    },
  }, null, 2));

  const outputAssets = path.join(root, 'converted', 'assets');
  const result = await writeSlmScene(glbScene, outputAssets, {
    sceneName: 'g1_fixture',
    targetBytes: 512,
  });
  assert.equal(result.unitCount, partition.units.length);
  for (const file of ['sceneWeb.json', 'glbIndex.json', 'runtimeVisibilityMeta.json', 'conversionManifest.json', 'scene_source_audit.json', 'scene_audit.json']) {
    assert.ok(fs.existsSync(path.join(outputAssets, file)), file);
  }
  const index = JSON.parse(fs.readFileSync(path.join(outputAssets, 'glbIndex.json'), 'utf8'));
  const runtime = JSON.parse(fs.readFileSync(path.join(outputAssets, 'runtimeVisibilityMeta.json'), 'utf8'));
  const manifest = JSON.parse(fs.readFileSync(path.join(outputAssets, 'conversionManifest.json'), 'utf8'));
  const sceneWeb = JSON.parse(fs.readFileSync(path.join(outputAssets, 'sceneWeb.json'), 'utf8'));
  assert.equal(index.total, result.unitCount);
  assert.equal(runtime.instanceCount, result.unitCount);
  assert.equal(runtime.globalGlbCount, result.unitCount);
  assert.equal(manifest.schema, 'pvs-standard-graphics-scene-connected-sah-pack-v2');
  assert.equal(manifest.partitionSchema, PARTITION_SCHEMA);
  assert.equal(manifest.componentCount, partition.componentCount);
  assert.equal(manifest.partition.schema, PARTITION_SCHEMA);
  assert.equal(manifest.partition.oversizeUnitCount, 0);
  assert.ok(manifest.partition.oversizedComponentSplitUnitCount > 0);
  assert.equal(
    manifest.partition.oversizedComponentSplitUnitCount,
    manifest.units.filter((unit) => unit.oversizedComponentSplit).length,
  );
  assert.equal(sceneWeb.partitionSchema, PARTITION_SCHEMA);
  assert.equal(sceneWeb.componentCount, partition.componentCount);
  assert.equal(sceneWeb.config.partitionSchema, PARTITION_SCHEMA);
  assert.ok(sceneWeb.config.renderableUnitSemantics.includes('connected component'));
  assert.equal(manifest.oneUnitPerResource, true);
  assert.equal(manifest.targetUnitBytes, 512);
  assert.equal(manifest.units.length, result.unitCount);
  assert.equal(manifest.unitRecords.length, result.unitCount);
  assert.ok(manifest.units.every((unit) => unit.partitionSchema === PARTITION_SCHEMA));
  assert.ok(manifest.units.every((unit) => unit.componentCount >= 1));
  assert.ok(manifest.units.every((unit) => Object.hasOwn(unit, 'oversizedComponentSplit')));
  assert.ok(manifest.units.every((unit) => unit.sourceComponentOrdinals.length === 1));
  assert.ok(manifest.units.every((unit) => Object.hasOwn(unit, 'oversize')));
  assert.equal(manifest.units.filter((unit) => unit.alphaMode === 'BLEND').length, 0);
  assert.equal(manifest.sourceAudit.blendExcludedFromPvsCount, 1);
  assert.equal(manifest.sourceAudit.blendAlwaysResidentPrimitiveCount, 1);
  assert.equal(manifest.sourceAudit.blendAlwaysResidentPolicy.includes('always resident'), true);
  assert.equal(manifest.sharedImages.length, 1);
  assert.ok(fs.existsSync(path.join(outputAssets, manifest.sharedImages[0].path)));
  assert.ok(manifest.units.every((unit) => !Object.hasOwn(unit, 'hash') && !Object.hasOwn(unit, 'glbHash')));

  const firstOutput = path.join(outputAssets, index.entries[0].path);
  const outputScene = await readGltfScene(firstOutput);
  assert.equal(outputScene.renderables.length, 1);
  assert.ok(outputScene.renderables[0].triangleCount > 0);
  const maskRecords = manifest.units.filter((unit) => unit.alphaMode === 'MASK');
  assert.equal(maskRecords.length, 2);
  const maskRecord = maskRecords[0];
  assert.ok(maskRecord);
  const maskOutput = path.join(outputAssets, maskRecord.path);
  const maskJson = (() => {
    const bytes = fs.readFileSync(maskOutput);
    const jsonLength = bytes.readUInt32LE(12);
    return JSON.parse(bytes.toString('utf8', 20, 20 + jsonLength).trim());
  })();
  assert.equal(maskJson.images.length, 1);
  assert.equal(maskJson.textures[0].source, 0);
  assert.equal(maskJson.samplers[0].wrapS, 10497);
  assert.equal(maskJson.materials[0].pbrMetallicRoughness.baseColorTexture.index, 0);
  assert.equal(maskJson.images[0].uri, 'shared/images/image_0.png');
  const secondMaskOutput = path.join(outputAssets, maskRecords[1].path);
  const secondMaskBytes = fs.readFileSync(secondMaskOutput);
  const secondMaskJsonLength = secondMaskBytes.readUInt32LE(12);
  const secondMaskJson = JSON.parse(secondMaskBytes.toString('utf8', 20, 20 + secondMaskJsonLength).trim());
  assert.equal(secondMaskJson.images[0].uri, maskJson.images[0].uri);
  const maskOutputScene = await readGltfScene(maskOutput);
  assert.equal(maskOutputScene.renderables[0].material.alphaMode, 'MASK');
  assert.equal(maskOutputScene.renderables[0].material.baseColorTexture.available, true);

  const threeRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../../slm2viewer/node_modules/three');
  const [{ GLTFLoader }, { MeshoptDecoder }, { LoadingManager, Texture }] = await Promise.all([
    import(pathToFileURL(path.join(threeRoot, 'examples/jsm/loaders/GLTFLoader.js')).href),
    import(pathToFileURL(path.join(threeRoot, 'examples/jsm/libs/meshopt_decoder.module.js')).href),
    import(pathToFileURL(path.join(threeRoot, 'build/three.module.js')).href),
  ]);
  globalThis.self = globalThis;
  const manager = new LoadingManager();
  manager.addHandler(/\.png$/i, {
    load(uri, onLoad, _onProgress, onError) {
      try {
        const imageBytes = fs.readFileSync(fileURLToPath(uri));
        assert.deepEqual([...imageBytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
        onLoad(new Texture({ data: imageBytes, width: 1, height: 1 }));
      } catch (error) {
        onError(error);
      }
    },
  });
  const loader = new GLTFLoader(manager).setMeshoptDecoder(MeshoptDecoder);
  const maskBytes = fs.readFileSync(maskOutput);
  const loaded = await new Promise((resolve, reject) => loader.parse(
    maskBytes.buffer.slice(maskBytes.byteOffset, maskBytes.byteOffset + maskBytes.byteLength),
    `${pathToFileURL(path.dirname(maskOutput)).href}/`,
    resolve,
    reject,
  ));
  let loadedMaskMaterial = null;
  loaded.scene.traverse((object) => {
    if (object.material?.map) loadedMaskMaterial = object.material;
  });
  assert.ok(loadedMaskMaterial);

  const unitAudit = auditOutputAssets(outputAssets);
  assert.equal(unitAudit.unitCount, result.unitCount);
  assert.equal(unitAudit.oneUnitPerResource, true);
  assert.equal(unitAudit.materialAlphaModeCounts.BLEND, 0);
  assert.equal(unitAudit.blendExcludedFromPvsCount, 1);
  assert.equal(unitAudit.targetUnitBytes, 512);
  console.log(JSON.stringify({ status: 'ok', units: result.unitCount, outputAssets }, null, 2));
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
