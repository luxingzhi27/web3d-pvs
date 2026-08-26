#!/usr/bin/env node
import assert from 'node:assert/strict';
import { RenderVisibilitySystem } from '../src/RenderVisibilitySystem.js';

function makeItem() {
  return {
    meshObject: {
      parent: null,
      visible: false,
      removeFromParent() {
        if (this.parent) this.parent = null;
      },
    },
    isVisible: false,
    isInScene: false,
    instancedBindingInvalid: false,
    weight: 0,
  };
}

const pool = { a: makeItem(), b: makeItem(), c: makeItem() };
const rootScene = {
  add(mesh) {
    mesh.parent = this;
  },
};
const loader = {
  useNeuralPVS: true,
  rootScene,
  decodeModelInfo(info) {
    return { hash: info.id, url: `${info.id}.glb` };
  },
  modelCacheMgr: {
    objectsPool: pool,
    getNowMs: () => 100,
    applyRenderState(item) {
      item.meshObject.visible = Boolean(item.isVisible && item.isInScene && !item.instancedBindingInvalid);
    },
  },
};

const system = new RenderVisibilitySystem(loader);
let stats = system.setWorkingSet([
  { id: 'a', weight: 0.8 },
  { id: 'b', weight: 0.7 },
], 'global-glb', 1);
assert.equal(stats.addedCount, 2);
assert.equal(stats.removedCount, 0);
assert.equal(stats.evaluatedCount, 2);
assert.equal(pool.a.meshObject.visible, true);
assert.equal(pool.b.meshObject.visible, true);

stats = system.setWorkingSet([
  { id: 'b', weight: 0.7 },
  { id: 'c', weight: 0.6 },
], 'global-glb', 1);
assert.equal(stats.addedCount, 1);
assert.equal(stats.removedCount, 1);
assert.equal(stats.evaluatedCount, 2);
assert.equal(pool.a.meshObject.visible, false);
assert.equal(pool.a.isInScene, true);
assert.equal(pool.b.meshObject.visible, true);
assert.equal(pool.c.meshObject.visible, true);

stats = system.applyDelta(
  [{ id: 'a', weight: 0.9 }],
  [{ id: 'c', weight: 0 }],
  'global-glb',
  1,
);
assert.equal(stats.deltaApplied, true);
assert.equal(stats.addedCount, 1);
assert.equal(stats.removedCount, 1);
assert.equal(stats.evaluatedCount, 2);
assert.deepEqual(Array.from(system.workingSetHashes).sort(), ['a', 'b']);
assert.equal(pool.a.meshObject.visible, true);
assert.equal(pool.b.meshObject.visible, true);
assert.equal(pool.c.meshObject.visible, false);
assert.equal(pool.c.isInScene, true);
assert.equal(stats.detachedCount, 0);

pool.b.isVisible = false;
pool.b.isInScene = false;
pool.b.meshObject.parent = null;
system.markResidentChanged('b');
assert.equal(pool.b.meshObject.visible, true);
assert.equal(pool.b.isInScene, true);
assert.equal(system.lastStats.evaluatedCount, 1);

console.log('RenderVisibilitySystem incremental working-set tests passed.');
