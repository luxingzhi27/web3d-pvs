// Exact GLB render working-set reconciler.
// The WebGPU instance filter already owns the 60-degree frustum decision; this
// class only applies additions/removals to resident GLBs and never culls again.

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

export class RenderVisibilitySystem {
  constructor(loader) {
    this.loader = loader;
    this.currentIdMode = null;
    this.currentEpoch = 0;
    this.workingSetHashes = new Set();
    this.workingSetWeights = new Map();
    this.visibleHashes = new Set();
    this.dirtyHashes = new Set();
    this.lastStats = null;
  }

  clear() {
    this.currentIdMode = null;
    this.currentEpoch = 0;
    this.workingSetHashes.clear();
    this.workingSetWeights.clear();
    this.visibleHashes.clear();
    this.dirtyHashes.clear();
    this.lastStats = null;
  }

  _isNeuralGlobalMode() {
    return this.loader.useNeuralPVS
      && (this.currentIdMode === 'global-glb' || this.currentIdMode === 'global-glb-priority');
  }

  _applyHash(hash, shouldShow, weight, stats) {
    const pool = this.loader?.modelCacheMgr?.objectsPool || {};
    const item = pool[hash];
    if (!item || !item.meshObject) {
      this.visibleHashes.delete(hash);
      if (stats) stats.missingResidentCount += 1;
      return;
    }

    if (shouldShow && !item.instancedBindingInvalid) {
      if (!item.isInScene && this.loader.rootScene) {
        this.loader.rootScene.add(item.meshObject);
        item.isInScene = true;
        if (stats) stats.attachedCount += 1;
      }
      item.isVisible = true;
      item.weight = Number(weight || 0);
      item.lastVisibleAt = this.loader.modelCacheMgr.getNowMs();
      this.visibleHashes.add(hash);
      if (stats) stats.shownCount += 1;
    } else {
      item.isVisible = false;
      item.weight = 0;
      this.visibleHashes.delete(hash);
      if (stats) stats.hiddenCount += 1;
    }
    this.loader.modelCacheMgr.applyRenderState(item);
  }

  setWorkingSet(modelInfos, idMode, epoch = 0) {
    const startedAt = nowMs();
    const nextHashes = new Set();
    const nextWeights = new Map();
    for (const modelInfo of modelInfos || []) {
      const decoded = this.loader.decodeModelInfo(modelInfo);
      if (!decoded?.hash) continue;
      nextHashes.add(decoded.hash);
      nextWeights.set(decoded.hash, Number(modelInfo.weight || 0));
    }

    const removed = [];
    const added = [];
    const changedWeights = [];
    for (const hash of this.workingSetHashes) {
      if (!nextHashes.has(hash)) removed.push(hash);
    }
    for (const hash of nextHashes) {
      if (!this.workingSetHashes.has(hash)) added.push(hash);
      else if (this.workingSetWeights.get(hash) !== nextWeights.get(hash)) changedWeights.push(hash);
    }

    this.currentIdMode = idMode || null;
    this.currentEpoch = Number(epoch || 0);
    this.workingSetHashes = nextHashes;
    this.workingSetWeights = nextWeights;
    const stats = {
      epoch: this.currentEpoch,
      workingSetSize: nextHashes.size,
      activeEvaluationSize: removed.length + added.length + changedWeights.length,
      evaluatedCount: removed.length + added.length + changedWeights.length,
      addedCount: added.length,
      removedCount: removed.length,
      weightChangedCount: changedWeights.length,
      shownCount: 0,
      hiddenCount: 0,
      attachedCount: 0,
      detachedCount: 0,
      missingResidentCount: 0,
      visibleCount: 0,
      skippedByGate: false,
      duplicateCullRemoved: true,
      durationMs: 0,
    };

    for (const hash of removed) this._applyHash(hash, false, 0, stats);
    for (const hash of added) this._applyHash(hash, true, nextWeights.get(hash), stats);
    for (const hash of changedWeights) this._applyHash(hash, true, nextWeights.get(hash), stats);
    for (const hash of Array.from(this.dirtyHashes)) {
      this._applyHash(hash, nextHashes.has(hash), nextWeights.get(hash), stats);
      this.dirtyHashes.delete(hash);
    }

    stats.visibleCount = this.visibleHashes.size;
    stats.durationMs = nowMs() - startedAt;
    this.lastStats = stats;
    this.loader.requestRender?.('render-working-set');
    return stats;
  }

  applyDelta(addedModelInfos, removedModelInfos, idMode, epoch = this.currentEpoch) {
    const startedAt = nowMs();
    const added = [];
    const removed = [];
    for (const modelInfo of removedModelInfos || []) {
      const decoded = this.loader.decodeModelInfo(modelInfo);
      if (!decoded?.hash || !this.workingSetHashes.has(decoded.hash)) continue;
      this.workingSetHashes.delete(decoded.hash);
      this.workingSetWeights.delete(decoded.hash);
      removed.push(decoded.hash);
    }
    for (const modelInfo of addedModelInfos || []) {
      const decoded = this.loader.decodeModelInfo(modelInfo);
      if (!decoded?.hash || this.workingSetHashes.has(decoded.hash)) continue;
      this.workingSetHashes.add(decoded.hash);
      this.workingSetWeights.set(decoded.hash, Number(modelInfo.weight || 0));
      added.push(decoded.hash);
    }

    this.currentIdMode = idMode || this.currentIdMode;
    this.currentEpoch = Number(epoch || this.currentEpoch || 0);
    const stats = {
      epoch: this.currentEpoch,
      workingSetSize: this.workingSetHashes.size,
      activeEvaluationSize: added.length + removed.length,
      evaluatedCount: added.length + removed.length,
      addedCount: added.length,
      removedCount: removed.length,
      weightChangedCount: 0,
      shownCount: 0,
      hiddenCount: 0,
      attachedCount: 0,
      detachedCount: 0,
      missingResidentCount: 0,
      visibleCount: 0,
      skippedByGate: false,
      duplicateCullRemoved: true,
      deltaApplied: true,
      durationMs: 0,
    };
    for (const hash of removed) this._applyHash(hash, false, 0, stats);
    for (const hash of added) this._applyHash(hash, true, this.workingSetWeights.get(hash), stats);
    stats.visibleCount = this.visibleHashes.size;
    stats.durationMs = nowMs() - startedAt;
    this.lastStats = stats;
    this.loader.requestRender?.('render-working-set-delta');
    return stats;
  }

  markResidentChanged(hash) {
    if (hash == null) return;
    if (!this._isNeuralGlobalMode()) return;
    this.dirtyHashes.add(hash);
    const stats = {
      ...(this.lastStats || {}),
      activeEvaluationSize: 1,
      evaluatedCount: 1,
      shownCount: 0,
      hiddenCount: 0,
      attachedCount: 0,
      detachedCount: 0,
      missingResidentCount: 0,
      residentChanged: true,
    };
    const startedAt = nowMs();
    this._applyHash(
      hash,
      this.workingSetHashes.has(hash),
      this.workingSetWeights.get(hash),
      stats,
    );
    this.dirtyHashes.delete(hash);
    stats.visibleCount = this.visibleHashes.size;
    stats.durationMs = nowMs() - startedAt;
    this.lastStats = stats;
    this.loader.requestRender?.('resident-render-state');
  }

  syncVisibleResidentsFromCache() {
    const pool = this.loader?.modelCacheMgr?.objectsPool || {};
    for (const hash in pool) {
      const item = pool[hash];
      if (item?.meshObject && item.isInScene) this.dirtyHashes.add(hash);
    }
  }

  update() {
    if (!this._isNeuralGlobalMode() || this.dirtyHashes.size === 0) return this.lastStats;
    const stats = {
      ...(this.lastStats || {}),
      activeEvaluationSize: this.dirtyHashes.size,
      evaluatedCount: this.dirtyHashes.size,
      shownCount: 0,
      hiddenCount: 0,
      attachedCount: 0,
      detachedCount: 0,
      missingResidentCount: 0,
    };
    const startedAt = nowMs();
    for (const hash of Array.from(this.dirtyHashes)) {
      this._applyHash(
        hash,
        this.workingSetHashes.has(hash),
        this.workingSetWeights.get(hash),
        stats,
      );
      this.dirtyHashes.delete(hash);
    }
    stats.visibleCount = this.visibleHashes.size;
    stats.durationMs = nowMs() - startedAt;
    this.lastStats = stats;
    return stats;
  }
}
