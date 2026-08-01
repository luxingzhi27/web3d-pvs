#!/usr/bin/env node
/* Minimal source-level runtime regression checks for the M9 frontend fixes. */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { CacheMgr } from '../slm2/CacheMgr.js';
import { RenderVisibilitySystem } from '../src/RenderVisibilitySystem.js';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_DIR = path.resolve(SCRIPT_DIR, '..');
const loaderSource = fs.readFileSync(path.join(VIEWER_DIR, 'slm2/SLM2Loader.js'), 'utf8');
const viewerSource = fs.readFileSync(path.join(VIEWER_DIR, 'src/viewer.js'), 'utf8');
const loaderClassStart = loaderSource.indexOf('export class SLM2Loader');
assert.notEqual(loaderClassStart, -1, 'SLM2Loader class must be present');

class FakeMatrix4 {
  constructor() {
    this.elements = new Array(16).fill(0);
    this.identity();
  }

  identity() {
    this.elements.fill(0);
    this.elements[0] = 1;
    this.elements[5] = 1;
    this.elements[10] = 1;
    this.elements[15] = 1;
    return this;
  }

  clone() {
    const copy = new FakeMatrix4();
    copy.elements = this.elements.slice();
    return copy;
  }
}

const testConsole = { warn() {} };
const context = vm.createContext({
  console: testConsole,
  INSTANCED_VISIBILITY_MATRIX: new FakeMatrix4(),
});
const classSource = loaderSource
  .slice(loaderClassStart)
  .replace(/^export class SLM2Loader/, 'class SLM2Loader');
vm.runInContext(`${classSource}\nglobalThis.TestSLM2Loader = SLM2Loader;`, context);
const SLM2Loader = context.TestSLM2Loader;

function createLoader() {
  const loader = Object.create(SLM2Loader.prototype);
  loader.instancedVisibilityBindingsByHash = {};
  loader.instancedVisibilityBindingByComponentId = {};
  loader.expectedInstancedVisibilityHashes = new Set();
  loader.loadedInstancedVisibilityStatesByHash = {};
  loader.lastRenderableComponentIdsForInstancing = [];
  loader.instancedRetainUntilByComponentId = new Map();
  loader.useNeuralPVS = false;
  return loader;
}

function createMesh(count) {
  return {
    isInstancedMesh: true,
    count,
    visible: true,
    instanceMatrix: { needsUpdate: false },
    getMatrixAt(index, target) {
      target.identity();
      target.elements[12] = index;
    },
    setMatrixAt() {},
    computeBoundingSphere() {},
  };
}

function createRoot(mesh) {
  return {
    visible: true,
    traverse(callback) {
      callback(mesh);
    },
  };
}

function attachFakePool(loader, hash, item) {
  loader.modelCacheMgr = {
    objectsPool: { [hash]: item },
    getNowMs: () => 0,
    applyRenderState(value) {
      value.meshObject.visible = Boolean(
        value.isVisible && value.isInScene && !value.instancedBindingInvalid,
      );
    },
  };
}

function testNormalUpdatePath() {
  const loader = createLoader();
  const events = [];
  const renderUpdateArgs = [];
  let sceneCullingCalls = 0;
  loader.isSceneInitialized = true;
  loader.updateLoading = () => events.push('loading');
  loader.processPendingSceneInsertions = () => events.push('insertions');
  loader.syncCamera = () => events.push('camera');
  loader._checkNeuralBackendUpgrade = () => events.push('backend');
  loader.modelCacheMgr = { update: () => events.push('cache') };
  loader.renderVisibilitySystem = {
    update: (...args) => {
      renderUpdateArgs.push(args);
      events.push('render-visibility');
    },
  };
  loader.processImageTask = () => events.push('images');
  loader.sceneCulling = () => { sceneCullingCalls += 1; };

  loader.update(16, 16);
  loader.update(16, 32);

  assert.equal(events.filter((event) => event === 'render-visibility').length, 2);
  assert.deepEqual(renderUpdateArgs, [[], []], 'normal update must not force a render refresh');
  assert.equal(sceneCullingCalls, 0, 'render update must not start neural prediction');
  assert.deepEqual(events.slice(0, 6), [
    'loading', 'insertions', 'camera', 'backend', 'cache', 'render-visibility',
  ]);
  assert.equal(events[6], 'images', 'render update must precede image work');
}

function testCacheMgrFailClosedFlag() {
  const cache = new CacheMgr({ sceneMgr: null });
  const root = { visible: true };
  const item = { meshObject: root, isVisible: true, isInScene: true, instancedBindingInvalid: true };
  cache.applyRenderState(item);
  assert.equal(root.visible, false, 'CacheMgr must hide an invalidly bound GLB root');
  item.instancedBindingInvalid = false;
  cache.applyRenderState(item);
  assert.equal(root.visible, true, 'valid bindings must retain normal render state behavior');
}

function testHiddenResidentIsDetached() {
  const item = {
    meshObject: { visible: true, isInScene: true },
    isVisible: true,
    isInScene: true,
  };
  let detached = 0;
  const system = Object.create(RenderVisibilitySystem.prototype);
  system.loader = {
    modelCacheMgr: {
      detachFromScene(value, hash) {
        assert.equal(value, item);
        assert.equal(hash, 'resident');
        detached += 1;
        value.isInScene = false;
        return true;
      },
      applyRenderState() {
        throw new Error('hidden resident should be detached before render-state fallback');
      },
    },
  };
  system.visibleHashes = new Set(['resident']);
  system.detachedCountThisUpdate = 0;
  system._setVisible(item, 'resident', false, 0, 0);
  assert.equal(detached, 1);
  assert.equal(system.detachedCountThisUpdate, 1);
  assert.equal(item.isVisible, false);
}

function testInvalidBindingFailsClosed() {
  const loader = createLoader();
  const hash = 'bad-binding';
  const mesh = createMesh(2);
  const root = createRoot(mesh);
  const item = { meshObject: root, isVisible: true, isInScene: true };
  attachFakePool(loader, hash, item);
  loader.instancedVisibilityBindingsByHash[hash] = [100];
  loader.instancedVisibilityBindingByComponentId[100] = { hash, instanceIndex: 0 };

  loader._applyInstancedVisibility([100]);
  loader._showAllResidentInstancedMeshes();

  const state = loader.loadedInstancedVisibilityStatesByHash[hash];
  assert.equal(state.disabled, true);
  assert.equal(state.bindingError, 'component-count-mismatch');
  assert.equal(item.instancedBindingInvalid, true);
  assert.equal(root.visible, false, 'invalid binding must hide the complete GLB root');
  assert.equal(mesh.visible, false, 'invalid binding must hide the InstancedMesh');
  assert.equal(mesh.count, 0, 'invalid binding must draw zero instances');
}

function testMissingExpectedBindingFailsClosed() {
  const loader = createLoader();
  const hash = 'missing-binding';
  const mesh = createMesh(2);
  const root = createRoot(mesh);
  const item = { meshObject: root, isVisible: true, isInScene: true };
  attachFakePool(loader, hash, item);
  loader.expectedInstancedVisibilityHashes.add(hash);

  const state = loader._ensureLoadedInstancedVisibilityState(hash);
  assert.equal(state.disabled, true);
  assert.equal(state.bindingError, 'missing-runtime-binding');
  assert.equal(item.instancedBindingInvalid, true);
  assert.equal(item.isVisible, false);
  assert.equal(root.visible, false, 'missing binding must hide the complete GLB root');
  assert.equal(mesh.visible, false, 'missing binding must hide the InstancedMesh');
  assert.equal(mesh.count, 0, 'missing binding must draw zero instances');
}

function testUpdateRunsBeforeRender() {
  const updateIndex = viewerSource.indexOf('this.slm2Loader.update(dt, this.prevTime);');
  const renderIndex = viewerSource.indexOf('this.render();');
  assert.notEqual(updateIndex, -1, 'viewer must update SLM2Loader each frame');
  assert.notEqual(renderIndex, -1, 'viewer must render each frame');
  assert.ok(updateIndex < renderIndex, 'SLM2Loader update must precede the frame render');
}

function testValidBindingStillFiltersInstances() {
  const loader = createLoader();
  const hash = 'valid-binding';
  const mesh = createMesh(2);
  const root = createRoot(mesh);
  const item = { meshObject: root, isVisible: true, isInScene: true };
  attachFakePool(loader, hash, item);
  loader.instancedVisibilityBindingsByHash[hash] = [200, 201];
  loader.instancedVisibilityBindingByComponentId[200] = { hash, instanceIndex: 0 };
  loader.instancedVisibilityBindingByComponentId[201] = { hash, instanceIndex: 1 };

  loader._applyInstancedVisibility([201]);

  const state = loader.loadedInstancedVisibilityStatesByHash[hash];
  assert.equal(state.disabled, false);
  assert.equal(item.instancedBindingInvalid, false);
  assert.equal(root.visible, true);
  assert.equal(mesh.visible, true);
  assert.equal(mesh.count, 1, 'valid binding must preserve instance-level filtering');
}

testNormalUpdatePath();
testCacheMgrFailClosedFlag();
testHiddenResidentIsDetached();
testInvalidBindingFailsClosed();
testMissingExpectedBindingFailsClosed();
testValidBindingStillFiltersInstances();
testUpdateRunsBeforeRender();
console.log('M9 frontend regression checks passed (7 cases).');
