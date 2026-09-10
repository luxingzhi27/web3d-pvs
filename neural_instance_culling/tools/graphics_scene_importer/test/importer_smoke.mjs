#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { auditOutputAssets } from '../audit_scene.mjs';
import { partitionScene } from '../partition_units.mjs';
import { readGltfScene } from '../read_gltf.mjs';
import { writeSlmScene } from '../write_slm_scene.mjs';
import { createFixtureScene } from './fixture_scene.mjs';

function unitSignature(partition) {
  return partition.units.map((unit) => ({
    unitId: unit.unitId,
    sourceNodePath: unit.sourceNodePath,
    sourcePrimitiveIndex: unit.sourcePrimitiveIndex,
    sourceTriangleIndices: unit.sourceTriangleIndices,
    sourceComponentOrdinals: unit.sourceComponentOrdinals,
    triangleCount: unit.triangleCount,
    vertexCount: unit.vertexCount,
    compressedGeometryBytes: unit.compression.encodedGeometryBytes,
  }));
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-graphics-scene-importer-g1-'));
try {
  const fixture = createFixtureScene(path.join(root, 'fixture'));
  const glbScene = await readGltfScene(fixture.glbPath);
  const gltfScene = await readGltfScene(fixture.gltfPath);
  assert.equal(glbScene.source.format, 'glb');
  assert.equal(gltfScene.source.format, 'gltf');
  assert.equal(glbScene.source.nodeCount, 5);
  assert.equal(glbScene.renderables.length, 3);
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
  assert.ok(partition.units.length > glbScene.renderables.length);
  assert.deepEqual(partition.units.map((unit) => unit.unitId), [...partition.units.keys()]);
  assert.equal(
    partition.units.reduce((sum, unit) => sum + unit.triangleCount, 0),
    glbScene.renderables.reduce((sum, item) => sum + item.triangleCount, 0),
  );
  assert.ok(partition.units.every((unit) => unit.compression.encodedGeometryBytes > 0));
  assert.equal(partition.units.some((unit) => unit.material.alphaMode === 'BLEND'), false);

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
  assert.equal(index.total, result.unitCount);
  assert.equal(runtime.instanceCount, result.unitCount);
  assert.equal(runtime.globalGlbCount, result.unitCount);
  assert.equal(manifest.oneUnitPerResource, true);
  assert.equal(manifest.targetUnitBytes, 512);
  assert.equal(manifest.units.length, result.unitCount);
  assert.equal(manifest.unitRecords.length, result.unitCount);
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
  const maskRecord = manifest.units.find((unit) => unit.alphaMode === 'MASK');
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
