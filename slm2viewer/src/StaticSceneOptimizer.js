import {
  BatchedMesh,
  Color,
  Matrix4,
  Object3D,
} from 'three';

const WHITE = new Color(1, 1, 1);
const DEFAULT_MIN_BATCH_SIZE = 4;
const DEFAULT_MAX_BATCH_SIZE = 64;
const DEFAULT_IDLE_FLUSH_MS = 1500;
const DEFAULT_MAX_VERTICES_PER_MESH = 250000;
const DEFAULT_MAX_VERTICES_PER_BATCH = 2000000;
const DEFAULT_MAX_INDICES_PER_BATCH = 4000000;

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function geometrySignature(geometry) {
  const attributes = Object.keys(geometry.attributes || {}).sort().map((name) => {
    const attribute = geometry.attributes[name];
    const arrayName = attribute?.array?.constructor?.name || 'unknown';
    return `${name}:${arrayName}:${attribute.itemSize}:${attribute.normalized ? 1 : 0}`;
  });
  const index = geometry.index;
  const indexSignature = index
    ? `${index.array?.constructor?.name || 'unknown'}:${index.itemSize}:${index.normalized ? 1 : 0}`
    : 'none';
  return `${indexSignature}|${attributes.join(',')}`;
}

function hasMorphAttributes(geometry) {
  return Object.values(geometry.morphAttributes || {}).some((attributes) => (
    Array.isArray(attributes) && attributes.length > 0
  ));
}

function isFullGeometryDrawRange(geometry) {
  const drawRange = geometry.drawRange || { start: 0, count: Infinity };
  const fullCount = geometry.index?.count || geometry.getAttribute('position')?.count || 0;
  return Number(drawRange.start || 0) === 0
    && (!Number.isFinite(drawRange.count) || Number(drawRange.count) >= fullCount);
}

function collectSingleStaticMesh(scene, gltf) {
  if (!scene || (Array.isArray(gltf?.animations) && gltf.animations.length > 0)) return null;
  const meshes = [];
  scene.traverse((node) => {
    if (node?.isMesh) meshes.push(node);
  });
  if (meshes.length !== 1 || meshes[0].isSkinnedMesh) return null;
  return meshes[0];
}

export class StaticSceneOptimizer {
  constructor(loader, options = {}) {
    this.loader = loader;
    this.minBatchSize = Number(options.minBatchSize || DEFAULT_MIN_BATCH_SIZE);
    this.maxBatchSize = Number(options.maxBatchSize || DEFAULT_MAX_BATCH_SIZE);
    this.idleFlushMs = Number(options.idleFlushMs || DEFAULT_IDLE_FLUSH_MS);
    this.maxVerticesPerMesh = Number(
      options.maxVerticesPerMesh || DEFAULT_MAX_VERTICES_PER_MESH,
    );
    this.maxVerticesPerBatch = Number(
      options.maxVerticesPerBatch || DEFAULT_MAX_VERTICES_PER_BATCH,
    );
    this.maxIndicesPerBatch = Number(
      options.maxIndicesPerBatch || DEFAULT_MAX_INDICES_PER_BATCH,
    );
    this.pendingByKey = new Map();
    this.pendingHashes = new Set();
    this.failedHashes = new Set();
    this.bindingsByHash = new Map();
    this.bindingsByComponentId = new Map();
    this.batches = new Set();
    this.stats = {
      flattenedGlbs: 0,
      batchedGlbs: 0,
      batchCount: 0,
      pendingGlbs: 0,
      rejectedGlbs: 0,
      releasedGlbs: 0,
    };
  }

  _requestRender(reason) {
    if (this.loader && typeof this.loader.requestRender === 'function') {
      this.loader.requestRender(reason);
    }
  }

  _componentCount(hash) {
    const glbId = this.loader?.globalGlbHashToId?.[hash];
    const componentIds = glbId == null ? null : this.loader?.globalGlbToComponentIds?.[glbId];
    return Array.isArray(componentIds) ? componentIds.length : 0;
  }

  _componentIds(hash) {
    const glbId = this.loader?.globalGlbHashToId?.[hash];
    const componentIds = glbId == null ? null : this.loader?.globalGlbToComponentIds?.[glbId];
    return Array.isArray(componentIds) ? componentIds : [];
  }

  _flattenSingleMesh(hash, item, gltf) {
    const scene = item?.meshObject;
    const mesh = collectSingleStaticMesh(scene, gltf);
    if (!mesh) return null;

    scene.updateMatrixWorld(true);
    const worldMatrix = mesh.matrixWorld.clone();
    const wasInScene = Boolean(item.isInScene);
    mesh.removeFromParent();
    if (scene.parent) scene.removeFromParent();

    mesh.matrix.copy(worldMatrix);
    mesh.matrixWorld.copy(worldMatrix);
    mesh.matrixAutoUpdate = false;
    mesh.matrixWorldAutoUpdate = false;
    mesh.userData.__slmStaticHash = hash;
    mesh.asset = gltf?.asset || scene.asset;
    if (wasInScene && this.loader?.rootScene) this.loader.rootScene.add(mesh);

    item.meshObject = mesh;
    item.isInScene = wasInScene;
    item.staticFlattened = true;
    this.loader?.modelCacheMgr?.applyRenderState(item);
    this.stats.flattenedGlbs += 1;
    return mesh;
  }

  _isBatchable(hash, mesh) {
    if (!mesh || !mesh.isMesh || mesh.isInstancedMesh || mesh.isSkinnedMesh || mesh.isBatchedMesh) {
      return false;
    }
    if (this._componentCount(hash) !== 1) return false;
    if (Array.isArray(mesh.material) || !mesh.material || mesh.material.transparent) return false;
    if (!mesh.geometry || hasMorphAttributes(mesh.geometry)) return false;
    if ((mesh.geometry.groups || []).length > 1 || !isFullGeometryDrawRange(mesh.geometry)) return false;
    const position = mesh.geometry.getAttribute('position');
    if (!position || position.count <= 0 || position.count > this.maxVerticesPerMesh) return false;
    return true;
  }

  _batchKey(hash, mesh) {
    const taskId = String(hash).split('-')[0];
    return `${taskId}|${mesh.material.uuid}|${geometrySignature(mesh.geometry)}`;
  }

  optimizeResident(hash, gltf = null) {
    if (!hash || this.bindingsByHash.has(hash) || this.failedHashes.has(hash)) return false;
    const item = this.loader?.modelCacheMgr?.objectsPool?.[hash];
    if (!item?.meshObject) return false;

    const mesh = item.staticFlattened
      ? item.meshObject
      : this._flattenSingleMesh(hash, item, gltf);
    if (!mesh || !this.loader?.isMaterialConfigReady || !this._isBatchable(hash, mesh)) {
      return Boolean(mesh);
    }
    if (this.pendingHashes.has(hash)) return true;

    const key = this._batchKey(hash, mesh);
    let group = this.pendingByKey.get(key);
    if (!group) {
      group = { key, records: [], lastAddedAt: 0 };
      this.pendingByKey.set(key, group);
    }
    group.records.push({ hash, mesh });
    group.lastAddedAt = nowMs();
    this.pendingHashes.add(hash);
    this.stats.pendingGlbs = this.pendingHashes.size;
    this._requestRender('static-batch-pending');
    return true;
  }

  optimizeAllResidents() {
    const pool = this.loader?.modelCacheMgr?.objectsPool || {};
    for (const hash of Object.keys(pool)) this.optimizeResident(hash);
  }

  _validRecords(group) {
    const valid = [];
    for (const record of group.records) {
      const item = this.loader?.modelCacheMgr?.objectsPool?.[record.hash];
      if (!item || item.meshObject !== record.mesh || !this._isBatchable(record.hash, record.mesh)) {
        this.pendingHashes.delete(record.hash);
        continue;
      }
      valid.push(record);
    }
    group.records = valid;
    return valid;
  }

  _selectBatchRecords(records) {
    const selected = [];
    let vertexCount = 0;
    let indexCount = 0;
    for (const record of records) {
      const geometry = record.mesh.geometry;
      const nextVertices = geometry.getAttribute('position').count;
      const nextIndices = geometry.index?.count || 0;
      if (selected.length > 0 && (
        selected.length >= this.maxBatchSize
        || vertexCount + nextVertices > this.maxVerticesPerBatch
        || indexCount + nextIndices > this.maxIndicesPerBatch
      )) break;
      selected.push(record);
      vertexCount += nextVertices;
      indexCount += nextIndices;
    }
    return { selected, vertexCount, indexCount };
  }

  _commitBatch(records, vertexCount, indexCount) {
    if (records.length < 2) return false;
    const material = records[0].mesh.material;
    const maxIndexCount = indexCount > 0 ? indexCount : Math.max(1, vertexCount * 2);
    const batch = new BatchedMesh(records.length, vertexCount, maxIndexCount, material);
    batch.name = `SLMStaticBatch_${this.stats.batchCount}`;
    batch.perObjectFrustumCulled = false;
    batch.frustumCulled = false;
    batch.sortObjects = false;
    batch.matrixAutoUpdate = false;
    batch.matrixWorldAutoUpdate = false;
    batch.updateMatrix();
    batch.updateMatrixWorld(true);

    const rootInverse = new Matrix4();
    if (this.loader?.rootScene) {
      this.loader.rootScene.updateMatrixWorld(true);
      rootInverse.copy(this.loader.rootScene.matrixWorld).invert();
    }
    const committed = [];
    try {
      for (const record of records) {
        const item = this.loader.modelCacheMgr.objectsPool[record.hash];
        const geometryId = batch.addGeometry(record.mesh.geometry);
        const instanceId = batch.addInstance(geometryId);
        const relativeMatrix = rootInverse.clone().multiply(record.mesh.matrixWorld);
        batch.setMatrixAt(instanceId, relativeMatrix);
        batch.setVisibleAt(instanceId, Boolean(item.isVisible && item.isInScene));
        committed.push({
          hash: record.hash,
          mesh: record.mesh,
          item,
          batch,
          geometryId,
          instanceId,
        });
      }
    } catch (error) {
      batch.dispose();
      console.warn('[StaticSceneOptimizer] Failed to build a static batch.', error);
      for (const record of records) this.failedHashes.add(record.hash);
      this.stats.rejectedGlbs += records.length;
      return false;
    }

    this.loader.rootScene.add(batch);
    const batchState = { mesh: batch, activeCount: committed.length };
    this.batches.add(batchState);
    for (const entry of committed) {
      entry.batchState = batchState;
      const handle = new Object3D();
      handle.name = `SLMStaticBatchHandle_${entry.hash}`;
      handle.userData.__slmStaticBatchHash = entry.hash;
      handle.asset = entry.mesh.asset;
      entry.mesh.removeFromParent();
      entry.mesh.geometry.dispose();
      entry.item.meshObject = handle;
      entry.item.staticBatchHash = entry.hash;
      entry.item.staticBatchState = batchState;
      entry.item.isInScene = true;
      entry.componentId = this._componentIds(entry.hash)[0];
      this.bindingsByHash.set(entry.hash, entry);
      if (Number.isInteger(Number(entry.componentId))) {
        this.bindingsByComponentId.set(Number(entry.componentId), entry);
      }
      this.loader.modelCacheMgr.applyRenderState(entry.item);
      this.stats.batchedGlbs += 1;
    }
    this.stats.batchCount = this.batches.size;
    this._requestRender('static-batch-committed');
    return true;
  }

  processPending(timestamp = nowMs(), flushRemainder = false) {
    for (const [key, group] of this.pendingByKey) {
      const records = this._validRecords(group);
      if (records.length === 0) {
        this.pendingByKey.delete(key);
        continue;
      }
      const minimumSize = flushRemainder ? 2 : this.minBatchSize;
      const ready = records.length >= this.maxBatchSize
        || (records.length >= minimumSize && timestamp - group.lastAddedAt >= this.idleFlushMs);
      if (!ready) continue;

      const selectedInfo = this._selectBatchRecords(records);
      if (selectedInfo.selected.length < 2) continue;
      const selectedHashes = new Set(selectedInfo.selected.map((record) => record.hash));
      group.records = records.filter((record) => !selectedHashes.has(record.hash));
      for (const record of selectedInfo.selected) this.pendingHashes.delete(record.hash);
      this._commitBatch(
        selectedInfo.selected,
        selectedInfo.vertexCount,
        selectedInfo.indexCount,
      );
      if (group.records.length === 0) this.pendingByKey.delete(key);
      this.stats.pendingGlbs = this.pendingHashes.size;
      return true;
    }
    this.stats.pendingGlbs = this.pendingHashes.size;
    return false;
  }

  hasReadyWork(timestamp = nowMs(), flushRemainder = false) {
    for (const group of this.pendingByKey.values()) {
      const count = group.records.length;
      const minimumSize = flushRemainder ? 2 : this.minBatchSize;
      if (count >= this.maxBatchSize) return true;
      if (count >= minimumSize && timestamp - group.lastAddedAt >= this.idleFlushMs) return true;
    }
    return false;
  }

  nextWakeDelay(timestamp = nowMs(), flushRemainder = false) {
    let delay = Infinity;
    for (const group of this.pendingByKey.values()) {
      const minimumSize = flushRemainder ? 2 : this.minBatchSize;
      if (group.records.length < minimumSize) continue;
      delay = Math.min(delay, Math.max(0, this.idleFlushMs - (timestamp - group.lastAddedAt)));
    }
    return Number.isFinite(delay) ? delay : null;
  }

  setHashVisible(hash, visible) {
    const entry = this.bindingsByHash.get(hash);
    if (!entry) return false;
    const nextVisible = Boolean(visible);
    if (entry.batch.getVisibleAt(entry.instanceId) === nextVisible) return true;
    entry.batch.setVisibleAt(entry.instanceId, nextVisible);
    this._requestRender('static-batch-visibility');
    return true;
  }

  setHashHighlight(hash, color = null) {
    const entry = this.bindingsByHash.get(hash);
    if (!entry) return false;
    entry.batch.setColorAt(entry.instanceId, color || WHITE);
    this._requestRender('static-batch-highlight');
    return true;
  }

  setComponentVisible(componentId, visible) {
    const entry = this.bindingsByComponentId.get(Number(componentId));
    return entry ? this.setHashVisible(entry.hash, visible) : false;
  }

  releaseHash(hash) {
    this.pendingHashes.delete(hash);
    this.failedHashes.delete(hash);
    for (const [key, group] of this.pendingByKey) {
      group.records = group.records.filter((record) => record.hash !== hash);
      if (group.records.length === 0) this.pendingByKey.delete(key);
    }
    const entry = this.bindingsByHash.get(hash);
    if (!entry) {
      this.stats.pendingGlbs = this.pendingHashes.size;
      return false;
    }
    entry.batch.deleteInstance(entry.instanceId);
    entry.batch.deleteGeometry(entry.geometryId);
    const batchState = entry.batchState;
    entry.item.staticBatchHash = null;
    entry.item.staticBatchState = null;
    this.bindingsByHash.delete(hash);
    if (Number.isInteger(Number(entry.componentId))) {
      this.bindingsByComponentId.delete(Number(entry.componentId));
    }
    entry.item && (entry.item.isInScene = false);
    if (batchState) {
      batchState.activeCount -= 1;
      if (batchState.activeCount <= 0) {
        batchState.mesh.removeFromParent();
        batchState.mesh.dispose();
        this.batches.delete(batchState);
      }
    }
    this.stats.releasedGlbs += 1;
    this.stats.batchedGlbs = this.bindingsByHash.size;
    this.stats.batchCount = this.batches.size;
    this.stats.pendingGlbs = this.pendingHashes.size;
    this._requestRender('static-batch-released');
    return true;
  }

  getStats() {
    return {
      ...this.stats,
      batchCount: this.batches.size,
      batchedGlbs: this.bindingsByHash.size,
      pendingGlbs: this.pendingHashes.size,
      drawCallReduction: Math.max(0, this.bindingsByHash.size - this.batches.size),
    };
  }
}
