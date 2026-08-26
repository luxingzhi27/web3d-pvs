#!/usr/bin/env node
import assert from 'node:assert/strict';
import {
  BoxGeometry,
  Group,
  InstancedMesh,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  PlaneGeometry,
} from 'three';
import { StaticSceneOptimizer } from '../src/StaticSceneOptimizer.js';

const rootScene = new Object3D();
const material = new MeshStandardMaterial({ color: 0x8090a0 });
let renderRequests = 0;
const loader = {
  rootScene,
  globalGlbHashToId: {},
  globalGlbToComponentIds: [],
  isMaterialConfigReady: true,
  requestRender() {
    renderRequests += 1;
  },
  modelCacheMgr: {
    objectsPool: {},
    applyRenderState(item) {
      const visible = Boolean(item.isVisible && item.isInScene);
      if (item.staticBatchHash) {
        loader.staticSceneOptimizer.setHashVisible(item.staticBatchHash, visible);
      } else {
        item.meshObject.visible = visible;
      }
    },
  },
};
loader.staticSceneOptimizer = new StaticSceneOptimizer(loader, {
  minBatchSize: 4,
  maxBatchSize: 8,
  idleFlushMs: 10,
});

for (let index = 0; index < 4; index += 1) {
  const hash = `0-${index}`;
  loader.globalGlbHashToId[hash] = index;
  loader.globalGlbToComponentIds[index] = [index];
  const scene = new Group();
  const nested = new Group();
  const mesh = new Mesh(new PlaneGeometry(1, 1), material);
  mesh.position.set(index * 2, 0, 0);
  nested.add(mesh);
  scene.add(nested);
  rootScene.add(scene);
  loader.modelCacheMgr.objectsPool[hash] = {
    meshObject: scene,
    isVisible: index !== 3,
    isInScene: true,
    staticBatchHash: null,
  };
  assert.equal(loader.staticSceneOptimizer.optimizeResident(hash, { scene, animations: [] }), true);
  assert.equal(loader.modelCacheMgr.objectsPool[hash].meshObject.isMesh, true);
  assert.equal(loader.modelCacheMgr.objectsPool[hash].meshObject.matrixAutoUpdate, false);
}

assert.equal(loader.staticSceneOptimizer.processPending(performance.now() + 20), true);
const stats = loader.staticSceneOptimizer.getStats();
assert.equal(stats.flattenedGlbs, 4);
assert.equal(stats.batchedGlbs, 4);
assert.equal(stats.batchCount, 1);
assert.equal(stats.drawCallReduction, 3);
assert.ok(renderRequests > 0);

for (let index = 0; index < 4; index += 1) {
  const hash = `0-${index}`;
  const item = loader.modelCacheMgr.objectsPool[hash];
  const binding = loader.staticSceneOptimizer.bindingsByHash.get(hash);
  assert.equal(loader.staticSceneOptimizer.bindingsByComponentId.get(index), binding);
  assert.equal(item.meshObject.isMesh, undefined);
  assert.equal(binding.batch.getVisibleAt(binding.instanceId), index !== 3);
}

loader.staticSceneOptimizer.setHashVisible('0-0', false);
const firstBinding = loader.staticSceneOptimizer.bindingsByHash.get('0-0');
assert.equal(firstBinding.batch.getVisibleAt(firstBinding.instanceId), false);
loader.staticSceneOptimizer.setHashHighlight('0-1', material.color);

for (let index = 0; index < 4; index += 1) {
  loader.staticSceneOptimizer.releaseHash(`0-${index}`);
}
assert.equal(loader.staticSceneOptimizer.getStats().batchCount, 0);
assert.equal(rootScene.children.some((child) => child.isBatchedMesh), false);

const instancedHash = '0-9';
loader.globalGlbHashToId[instancedHash] = 9;
loader.globalGlbToComponentIds[9] = [9, 10];
const instancedScene = new Group();
const instanced = new InstancedMesh(new BoxGeometry(), material, 2);
instancedScene.add(instanced);
rootScene.add(instancedScene);
loader.modelCacheMgr.objectsPool[instancedHash] = {
  meshObject: instancedScene,
  isVisible: true,
  isInScene: true,
};
loader.staticSceneOptimizer.optimizeResident(instancedHash, { scene: instancedScene, animations: [] });
assert.equal(loader.modelCacheMgr.objectsPool[instancedHash].meshObject, instanced);
assert.equal(loader.staticSceneOptimizer.bindingsByHash.has(instancedHash), false);

console.log('Static scene flattening and runtime BatchedMesh tests passed.');
