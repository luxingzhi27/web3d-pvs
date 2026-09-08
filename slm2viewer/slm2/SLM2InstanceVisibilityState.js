import { applyDenseInstancedDelta, clearDenseInstancedState, initializeDenseInstancedState, replaceDenseInstancedSlots } from '../src/DenseInstancedSlots.js';
import { SLM2MaterialSystem } from './SLM2MaterialSystem.js';

export class SLM2InstanceVisibilityState extends SLM2MaterialSystem {
  _recordVisibilityMetrics(metrics)
  {
    this.visibilityMetricsSerial++;
    this.lastVisibilityMetrics = {
      serial: this.visibilityMetricsSerial,
      timestamp: performance.now(),
      mode: metrics.mode,
      idMode: metrics.idMode || null,
      latencyMs: Number(metrics.latencyMs || 0),
      backend: metrics.backend || null,
      visibleCount: Number(metrics.visibleCount || 0),
      rawCount: Number(metrics.rawCount != null ? metrics.rawCount : metrics.visibleCount || 0),
      visibleGlbCount: Number(metrics.visibleGlbCount != null ? metrics.visibleGlbCount : metrics.visibleCount || 0),
      rawGlbCount: Number(metrics.rawGlbCount != null ? metrics.rawGlbCount : metrics.rawCount != null ? metrics.rawCount : metrics.visibleCount || 0),
      visibleInstanceCount: Number(metrics.visibleInstanceCount != null ? metrics.visibleInstanceCount : metrics.visibleCount || 0),
      rawInstanceCount: Number(metrics.rawInstanceCount != null ? metrics.rawInstanceCount : metrics.rawCount != null ? metrics.rawCount : metrics.visibleCount || 0),
      requestId: metrics.requestId != null ? Number(metrics.requestId) : null,
      notes: metrics.notes || null,
      renderPolicy: this.getNeuralRenderPolicy(),
      cache: this.modelCacheMgr.getSnapshot(),
      loadQueueLength: this.useNeuralPVS
        ? this.glbResourceScheduler.getSnapshot().queuedImmediate
        : this.modelToLoadList.length,
      pendingSceneInsertions: this._getPendingSceneInsertionCount(),
    };
    this.lastBenchmarkVisibilityIds.serial = this.visibilityMetricsSerial;
    this.lastBenchmarkVisibilityIds.mode = metrics.mode || null;
    this.requestRender('visibility-result');
    if (this.neuralDebugLogs) console.log('[SLM2Loader] Visibility metrics', this.lastVisibilityMetrics);
  }

  _normalizeIdList(values)
  {
    return Array.from(new Set((values || []).map(function(value)
    {
      return Number(value);
    }).filter(function(value)
    {
      return Number.isFinite(value);
    }))).sort(function(a, b){ return a - b; });
  }

  _getComponentBitCount()
  {
    var runtimeCount = Number(this.runtimeVisibilityMeta && this.runtimeVisibilityMeta.instanceCount);
    if (Number.isInteger(runtimeCount) && runtimeCount > 0) return runtimeCount;
    return Math.max(0, this.componentVisibilityRecords.length);
  }

  _getGlbBitCount()
  {
    var runtimeCount = Number(this.runtimeVisibilityMeta && this.runtimeVisibilityMeta.globalGlbCount);
    if (Number.isInteger(runtimeCount) && runtimeCount > 0) return runtimeCount;
    if (this.glbIndex && Number.isInteger(Number(this.glbIndex.total))) return Number(this.glbIndex.total);
    return Math.max(0, this.globalGlbEntries.length);
  }

  _buildInstancedVisibilityBindings()
  {
    this.instancedVisibilityBindingsByHash = {};
    this.instancedVisibilityBindingByComponentId = {};

    if (!this.sceneConfig || !Array.isArray(this.sceneConfig.groups))
    {
      return;
    }

    for (var groupIndex = 0; groupIndex < this.sceneConfig.groups.length; ++groupIndex)
    {
      var group = this.sceneConfig.groups[groupIndex];
      if (!group || !Array.isArray(group.idRange) || group.idRange.length < 2)
      {
        continue;
      }

      var instancesMap = group.instances || {};
      var reverseMap = {};
      for (var localIdKey in instancesMap)
      {
        var localId = Number(localIdKey);
        var baseId = Number(instancesMap[localIdKey]);
        if (!Number.isFinite(localId) || !Number.isFinite(baseId))
        {
          continue;
        }
        if (!reverseMap[baseId])
        {
          reverseMap[baseId] = [];
        }
        reverseMap[baseId].push(localId);
      }

      for (var baseIdKey in reverseMap)
      {
        var instancedBaseId = Number(baseIdKey);
        if (!Number.isFinite(instancedBaseId))
        {
          continue;
        }

        var orderedLocalIds = [instancedBaseId].concat(reverseMap[instancedBaseId].sort(function(a, b){ return a - b; }));
        var hash = groupIndex + '-' + instancedBaseId;
        var componentIds = [];
        for (var i = 0; i < orderedLocalIds.length; ++i)
        {
          var componentId = Number(group.idRange[0]) + Number(orderedLocalIds[i]);
          componentIds.push(componentId);
          this.instancedVisibilityBindingByComponentId[componentId] = {
            hash: hash,
            instanceIndex: i,
          };
        }
        this.instancedVisibilityBindingsByHash[hash] = componentIds;
      }
    }
  }

  _isInstancedBindingExpected(hash)
  {
    return Boolean(hash && this.expectedInstancedVisibilityHashes &&
      this.expectedInstancedVisibilityHashes.has(hash));
  }

  _markMissingInstancedVisibilityBinding(hash, item)
  {
    if (!hash || !item || !item.meshObject)
    {
      return null;
    }

    var existing = this.loadedInstancedVisibilityStatesByHash[hash];
    if (existing && existing.disabled && existing.bindingError === 'missing-runtime-binding')
    {
      this._hideInvalidInstancedVisibilityState(existing);
      return existing;
    }

    var meshStates = [];
    var originalCount = 0;
    item.meshObject.traverse(function(node)
    {
      if (!node || !node.isInstancedMesh)
      {
        return;
      }

      originalCount += Number(node.count || 0);
      meshStates.push({
        mesh: node,
        originalMatrixArray: new Float32Array(),
        originalCount: Number(node.count || 0),
      });
    });

    var state = {
      hash: hash,
      item: item,
      meshStates: meshStates,
      originalCount: originalCount,
      componentIds: [],
      disabled: true,
      bindingError: 'missing-runtime-binding',
      activeSourceIndices: [],
    };
    this.loadedInstancedVisibilityStatesByHash[hash] = state;
    item.instancedBindingInvalid = true;
    item.isVisible = false;
    this._hideInvalidInstancedVisibilityState(state);
    if (this.modelCacheMgr && typeof this.modelCacheMgr.applyRenderState === 'function')
    {
      this.modelCacheMgr.applyRenderState(item);
    }
    if (this.neuralDebugLogs || this.DebugMode)
    {
      console.warn('[SLM2Loader] Missing instanced visibility binding; hiding the entire GLB.', {
        hash: hash,
        expectedInstanceCount: originalCount,
      });
    }
    return state;
  }

  _validateResidentInstancedBindings()
  {
    var pool = this.modelCacheMgr && this.modelCacheMgr.objectsPool
      ? this.modelCacheMgr.objectsPool
      : {};
    var expected = this.expectedInstancedVisibilityHashes || new Set();
    expected.forEach(function(hash)
    {
      if (this.instancedVisibilityBindingsByHash[hash])
      {
        return;
      }
      var item = pool[hash];
      if (item && item.meshObject)
      {
        this._markMissingInstancedVisibilityBinding(hash, item);
      }
    }, this);
  }

  _getInstancedBindingStats()
  {
    var states = this.loadedInstancedVisibilityStatesByHash || {};
    var invalidCount = 0;
    var loadedCount = 0;
    for (var hash in states)
    {
      if (!states[hash]) continue;
      loadedCount++;
      if (states[hash].disabled) invalidCount++;
    }
    return {
      expectedCount: this.expectedInstancedVisibilityHashes
        ? this.expectedInstancedVisibilityHashes.size
        : 0,
      mappedCount: Object.keys(this.instancedVisibilityBindingsByHash || {}).length,
      loadedCount: loadedCount,
      invalidCount: invalidCount,
    };
  }

  _ensureLoadedInstancedVisibilityState(hash)
  {
    if (!hash)
    {
      return null;
    }

    var pool = this.modelCacheMgr && this.modelCacheMgr.objectsPool ? this.modelCacheMgr.objectsPool : null;
    var item = pool ? pool[hash] : null;
    var hasBinding = Boolean(this.instancedVisibilityBindingsByHash[hash]);
    if (!hasBinding && !this._isInstancedBindingExpected(hash))
    {
      return null;
    }

    if (!hasBinding)
    {
      return this._markMissingInstancedVisibilityBinding(hash, item);
    }

    var existing = this.loadedInstancedVisibilityStatesByHash[hash];
    var componentIds = this.instancedVisibilityBindingsByHash[hash] || [];

    if (!item || !item.meshObject)
    {
      if (existing)
      {
        delete this.loadedInstancedVisibilityStatesByHash[hash];
      }
      return null;
    }

    if (existing && existing.item === item)
    {
      item.instancedBindingInvalid = Boolean(existing.disabled);
      if (existing.disabled)
      {
        this._hideInvalidInstancedVisibilityState(existing);
      }
      return existing;
    }

    var failClosed = (reason, meshStates, expectedCount) =>
    {
      var state = {
        hash: hash,
        item: item,
        meshStates: meshStates || [],
        originalCount: expectedCount == null ? 0 : expectedCount,
        componentIds: componentIds,
        disabled: true,
        bindingError: reason,
        activeSourceIndices: [],
      };
      this.loadedInstancedVisibilityStatesByHash[hash] = state;
      item.instancedBindingInvalid = true;
      this._hideInvalidInstancedVisibilityState(state);
      if (typeof console !== 'undefined' && typeof console.warn === 'function')
      {
        console.warn('[SLM2Loader] Instanced visibility binding is invalid; hiding the entire GLB.', {
          hash: hash,
          reason: reason,
          componentCount: componentIds.length,
          meshCount: state.meshStates.length,
          meshInstanceCount: state.originalCount,
        });
      }
      return state;
    };

    var meshStates = [];
    var expectedCount = null;
    var countMismatch = false;
    item.meshObject.traverse(function(node)
    {
      if (!node || !node.isInstancedMesh)
      {
        return;
      }

      var originalMatrixArray = node.instanceMatrix && node.instanceMatrix.array
        ? new Float32Array(node.instanceMatrix.array.slice(0, node.count * 16))
        : new Float32Array();
      node.frustumCulled = false;

      if (expectedCount == null)
      {
        expectedCount = node.count;
      }
      else if (expectedCount !== node.count)
      {
        countMismatch = true;
      }

      meshStates.push({
        mesh: node,
        originalMatrixArray: originalMatrixArray,
        originalCount: node.count,
      });
    });

    if (meshStates.length === 0)
    {
      return failClosed('missing-instanced-mesh', meshStates, expectedCount);
    }

    var bindingError = null;
    if (countMismatch)
    {
      bindingError = 'instanced-mesh-count-mismatch';
    }
    else if (componentIds.length !== expectedCount)
    {
      bindingError = 'component-count-mismatch';
    }
    else
    {
      for (var bindingIndex = 0; bindingIndex < componentIds.length; ++bindingIndex)
      {
        var componentBinding = this.instancedVisibilityBindingByComponentId[componentIds[bindingIndex]];
        if (!componentBinding || componentBinding.hash !== hash ||
            Number(componentBinding.instanceIndex) !== bindingIndex)
        {
          bindingError = 'component-index-mismatch';
          break;
        }
      }
    }

    if (bindingError)
    {
      return failClosed(bindingError, meshStates, expectedCount);
    }

    var state = {
      hash: hash,
      item: item,
      meshStates: meshStates,
      originalCount: expectedCount == null ? 0 : expectedCount,
      componentIds: componentIds,
      disabled: false,
      bindingError: null,
      activeSourceIndices: [],
    };
    initializeDenseInstancedState(state);
    item.instancedBindingInvalid = false;
    this.loadedInstancedVisibilityStatesByHash[hash] = state;
    return state;
  }

  _hideInvalidInstancedVisibilityState(state)
  {
    if (!state)
    {
      return;
    }

    if (state.item)
    {
      state.item.instancedBindingInvalid = true;
      if (state.item.meshObject)
      {
        state.item.meshObject.visible = false;
      }
    }
    if (this.renderVisibilitySystem && this.renderVisibilitySystem.visibleHashes)
    {
      this.renderVisibilitySystem.visibleHashes.delete(state.hash);
    }

    for (var meshIdx = 0; meshIdx < (state.meshStates || []).length; ++meshIdx)
    {
      var mesh = state.meshStates[meshIdx].mesh;
      if (!mesh)
      {
        continue;
      }
      mesh.visible = false;
      mesh.count = 0;
      if (mesh.instanceMatrix)
      {
        mesh.instanceMatrix.needsUpdate = true;
      }
    }
    clearDenseInstancedState(state);
  }

  _syncInstancedVisibilityHash(hash)
  {
    var state = this._ensureLoadedInstancedVisibilityState(hash);
    if (!state)
    {
      return;
    }
    var wanted = this.instancedWantedIndicesByHash.get(hash);
    var activeIndices = wanted ? Array.from(wanted) : [];
    activeIndices = activeIndices.filter(function(index)
    {
      return Number.isInteger(index) && index >= 0 && index < state.originalCount;
    });
    replaceDenseInstancedSlots(state, activeIndices);
  }

  _syncAllLoadedInstancedVisibilityStates()
  {
    var states = this.loadedInstancedVisibilityStatesByHash || {};
    for (var hash in states)
    {
      this._syncInstancedVisibilityHash(hash);
    }
  }

  _updateWantedInstancedComponent(componentId, add, deltasByHash)
  {
    var binding = this.instancedVisibilityBindingByComponentId[componentId];
    if (!binding || !binding.hash)
    {
      return;
    }
    var wanted = this.instancedWantedIndicesByHash.get(binding.hash);
    if (!wanted)
    {
      wanted = new Set();
      this.instancedWantedIndicesByHash.set(binding.hash, wanted);
    }
    if (add)
    {
      if (wanted.has(Number(binding.instanceIndex))) return;
      wanted.add(Number(binding.instanceIndex));
    }
    else
    {
      if (!wanted.has(Number(binding.instanceIndex))) return;
      wanted.delete(Number(binding.instanceIndex));
      if (wanted.size === 0)
      {
        this.instancedWantedIndicesByHash.delete(binding.hash);
      }
    }
    var delta = deltasByHash.get(binding.hash);
    if (!delta)
    {
      delta = { added: [], removed: [] };
      deltasByHash.set(binding.hash, delta);
    }
    (add ? delta.added : delta.removed).push(Number(binding.instanceIndex));
  }

  _applyInstancedDeltasByHash(deltasByHash)
  {
    deltasByHash.forEach((delta, hash) =>
    {
      var state = this._ensureLoadedInstancedVisibilityState(hash);
      if (!state || state.disabled) return;
      applyDenseInstancedDelta(state, delta.added, delta.removed);
    });
  }

  _applyInstancedVisibility(componentIds)
  {
    var delta = this.renderComponentState.replace(componentIds, this._getComponentBitCount());
    var deltasByHash = new Map();
    for (var removedIndex = 0; removedIndex < delta.removed.length; ++removedIndex)
    {
      this._updateWantedInstancedComponent(delta.removed[removedIndex], false, deltasByHash);
    }
    for (var addedIndex = 0; addedIndex < delta.added.length; ++addedIndex)
    {
      this._updateWantedInstancedComponent(delta.added[addedIndex], true, deltasByHash);
    }
    this._applyInstancedDeltasByHash(deltasByHash);
    return delta.count;
  }

  _applyInstancedVisibilityDelta(addedComponentIds, removedComponentIds)
  {
    var added = this._normalizeIdList(addedComponentIds);
    var removed = this._normalizeIdList(removedComponentIds);
    var delta = this.renderComponentState.applyDelta(added, removed);
    var deltasByHash = new Map();
    for (var removedIndex = 0; removedIndex < delta.removed.length; ++removedIndex)
    {
      this._updateWantedInstancedComponent(delta.removed[removedIndex], false, deltasByHash);
    }
    for (var addedIndex = 0; addedIndex < delta.added.length; ++addedIndex)
    {
      this._updateWantedInstancedComponent(delta.added[addedIndex], true, deltasByHash);
    }
    this._applyInstancedDeltasByHash(deltasByHash);
    return delta.count;
  }

  _componentIdsToGlbIds(componentIds)
  {
    var scope = this;
    return this._normalizeIdList((componentIds || []).map(function(componentId)
    {
      var record = scope.componentVisibilityRecords[componentId];
      return record ? record.globalGlbId : null;
    }));
  }

  _filterComponentIdsByGlbIds(componentIds, glbIds)
  {
    var glbSet = new Set(this._normalizeIdList(glbIds));
    if (glbSet.size === 0)
    {
      return [];
    }

    var scope = this;
    return this._normalizeIdList((componentIds || []).filter(function(componentId)
    {
      var record = scope.componentVisibilityRecords[componentId];
      return record && glbSet.has(record.globalGlbId);
    }));
  }

  _getComponentIdsInFrustum(cameraOverride)
  {
    if (!this.runtimeVisibilityMeta || !Array.isArray(this.runtimeVisibilityMeta.componentRecords))
    {
      return [];
    }

    this._updateRuntimeVisibilityFrustum(cameraOverride);
    var records = this.runtimeVisibilityMeta.componentRecords;
    var componentIds = [];
    for (var index = 0; index < records.length; ++index)
    {
      var record = records[index];
      var componentId = Number(record && record.componentGlobalId);
      if (record && Number.isFinite(componentId) && componentId >= 0 &&
          this._intersectsFrustumWithCenterSize(record.bounds))
      {
        componentIds.push(componentId);
      }
    }
    return this._normalizeIdList(componentIds);
  }

  _applyLocalFrustumCulling()
  {
    if (!this.runtimeVisibilityMeta)
    {
      return {
        ready: false,
        candidateComponentIds: [],
        renderComponentIds: [],
        candidateGlbIds: [],
        renderGlbIds: [],
      };
    }

    // 后退视锥负责提前下载，真实视锥负责最终显示；两者都只使用实例 AABB。
    var candidateComponentIds = this._getComponentIdsInFrustum(this.backCamera);
    var renderComponentIds = this._getComponentIdsInFrustum(this.activeCamera);
    var candidateGlbIds = this._componentIdsToGlbIds(candidateComponentIds).filter(function(globalGlbId)
    {
      return this.globalGlbEntries[globalGlbId] != null;
    }, this);
    var renderGlbIds = this._componentIdsToGlbIds(renderComponentIds).filter(function(globalGlbId)
    {
      return this.globalGlbEntries[globalGlbId] != null;
    }, this);
    var candidateWeights = candidateGlbIds.map(function(){ return 1; });

    this.refreshLoadingTask(candidateGlbIds, candidateWeights, 'global-glb');

    var renderInfos = this._makeGlobalGlbModelInfos(renderGlbIds, null, 'global-glb');
    this.lastRenderRefreshStats = this.modelCacheMgr.refreshVisible(
      renderInfos,
      this._getRenderVisibilityOptions('global-glb')
    );
    this.renderGlbState.replace(renderGlbIds, this._getGlbBitCount());
    this._applyInstancedVisibility(renderComponentIds);
    this._updateRuntimeVisibilityFrustum(this.activeCamera);

    var schedulerStats = {
      scheduler: 'local-aabb-frustum-v1',
      backend: 'main-thread-aabb',
      candidateComponentCount: candidateComponentIds.length,
      renderComponentCount: renderComponentIds.length,
      candidateGlbCount: candidateGlbIds.length,
      renderGlbCount: renderGlbIds.length,
      downloadCamera: 'back-camera',
      renderCamera: 'active-camera',
    };
    this.lastLightweightPVSSchedulerStats = schedulerStats;
    this._updateBenchmarkVisibilityIds({
      mode: 'frustum',
      rawComponentIds: candidateComponentIds,
      rawGlbIds: candidateGlbIds,
      scheduledComponentIds: candidateComponentIds,
      scheduledGlbIds: candidateGlbIds,
      renderComponentIds: renderComponentIds,
      renderGlbIds: renderGlbIds,
      prefetchComponentIds: [],
      prefetchGlbIds: [],
    });
    this._recordVisibilityMetrics({
      mode: 'frustum',
      idMode: 'global-glb',
      backend: 'main-thread-aabb',
      latencyMs: 0,
      rawCount: candidateComponentIds.length,
      visibleCount: renderComponentIds.length,
      rawGlbCount: candidateGlbIds.length,
      visibleGlbCount: renderGlbIds.length,
      rawInstanceCount: candidateComponentIds.length,
      visibleInstanceCount: renderComponentIds.length,
      notes: schedulerStats,
    });

    return {
      ready: true,
      candidateComponentIds: candidateComponentIds,
      renderComponentIds: renderComponentIds,
      candidateGlbIds: candidateGlbIds,
      renderGlbIds: renderGlbIds,
    };
  }

  _sanitizePriorityItems(items)
  {
    return (items || []).map(function(item)
    {
      return {
        globalGlbId: item.globalGlbId != null ? Number(item.globalGlbId) : null,
        priority: Number(item.priority || 0),
        heuristicPriority: Number(item.heuristicPriority != null ? item.heuristicPriority : item.priority || 0),
        projectedArea: Number(item.projectedArea || 0),
        confidence: Number(item.confidence || 0),
        importance: Number(item.importance || item.neuralImportance || 0),
        downloadPriority: item.downloadPriority == null ? null : Number(item.downloadPriority),
        visibilityScore: item.visibilityScore == null ? null : Number(item.visibilityScore),
        distance: Number.isFinite(Number(item.distance)) ? Number(item.distance) : null,
        centerBias: Number(item.centerBias || 0),
        componentCount: Number(item.componentCount || 0),
        testedComponents: Number(item.testedComponents || 0),
        usedGlobalFallback: Boolean(item.usedGlobalFallback),
        predictedUtility: item.predictedUtility == null ? null : Number(item.predictedUtility),
        estimatedCost: item.estimatedCost == null ? null : Number(item.estimatedCost),
        densityScore: item.densityScore == null ? null : Number(item.densityScore),
        tier: item.tier || null,
      };
    }).filter(function(item)
    {
      return Number.isFinite(item.globalGlbId);
    });
  }

  _updateBenchmarkVisibilityIds(payload = {})
  {
    this.lastBenchmarkVisibilityIds = {
      serial: Number(payload.serial != null ? payload.serial : this.lastBenchmarkVisibilityIds.serial || 0),
      mode: payload.mode || this.lastBenchmarkVisibilityIds.mode || null,
      rawComponentIds: this._normalizeIdList(payload.rawComponentIds),
      rawGlbIds: this._normalizeIdList(payload.rawGlbIds),
      scheduledComponentIds: this._normalizeIdList(payload.scheduledComponentIds),
      scheduledGlbIds: this._normalizeIdList(payload.scheduledGlbIds),
      prefetchComponentIds: this._normalizeIdList(payload.prefetchComponentIds),
      prefetchGlbIds: this._normalizeIdList(payload.prefetchGlbIds),
      priorityItems: this._sanitizePriorityItems(payload.priorityItems),
      candidateSelection: payload.candidateSelection || null,
    };
  }

  getBenchmarkVisibilityIds()
  {
    return {
      serial: Number(this.lastBenchmarkVisibilityIds.serial || 0),
      mode: this.lastBenchmarkVisibilityIds.mode || null,
      rawComponentIds: this.lastBenchmarkVisibilityIds.rawComponentIds.slice(),
      rawGlbIds: this.lastBenchmarkVisibilityIds.rawGlbIds.slice(),
      scheduledComponentIds: this.lastBenchmarkVisibilityIds.scheduledComponentIds.slice(),
      scheduledGlbIds: this.lastBenchmarkVisibilityIds.scheduledGlbIds.slice(),
      renderComponentIds: this.renderComponentState.toIds(),
      renderGlbIds: this.renderGlbState.toIds(),
      prefetchComponentIds: this.lastBenchmarkVisibilityIds.prefetchComponentIds.slice(),
      prefetchGlbIds: this.lastBenchmarkVisibilityIds.prefetchGlbIds.slice(),
      priorityItems: this.lastBenchmarkVisibilityIds.priorityItems.slice(),
      candidateSelection: this.lastBenchmarkVisibilityIds.candidateSelection
        ? Object.assign({}, this.lastBenchmarkVisibilityIds.candidateSelection)
        : null,
    };
  }

  getRuntimeStats()
  {
    var schedulerStats = this.lastLightweightPVSSchedulerStats || null;
    var resourceSchedule = this.glbResourceScheduler.getSnapshot();
    return {
      startup: Object.assign({}, this.startupMetrics),
      visibility: this.lastVisibilityMetrics,
      cache: this.modelCacheMgr.getSnapshot(),
      load: {
        queueLength: this.useNeuralPVS ? resourceSchedule.queuedImmediate : this.modelToLoadList.length,
        queuePreview: this.useNeuralPVS
          ? resourceSchedule.immediatePreview
          : this.modelToLoadList.slice(Math.max(0, this.modelToLoadList.length - 8)),
        inflightCount: this.inflightModelHashes ? this.inflightModelHashes.size : 0,
        pendingHashCount: this.pendingSceneInsertionHashes ? this.pendingSceneInsertionHashes.size : 0,
        wantedHashCount: this.useNeuralPVS ? resourceSchedule.wanted : 0,
        urgentWantedCount: this.useNeuralPVS ? resourceSchedule.urgentWanted : 0,
        urgentResidentCount: this.useNeuralPVS ? resourceSchedule.urgentResident : 0,
        urgentMissingCount: this.useNeuralPVS ? resourceSchedule.urgentMissing : 0,
        urgentFailedCount: this.useNeuralPVS ? resourceSchedule.urgentFailed : 0,
        pendingSceneInsertions: this._getPendingSceneInsertionCount(),
        activeDirectLoadCount: this.activeDirectLoadCount,
        pendingParseCount: this._getPendingGlbParseCount(),
        pendingParseMB: this.pendingGlbParseBytes / 1048576,
        activeParseCount: this.activeGlbParseCount,
        parseConcurrency: this.glbParseConcurrency,
        prefetchQueueLength: this.useNeuralPVS ? resourceSchedule.queuedPrefetch : 0,
        prefetchPreview: this.useNeuralPVS ? resourceSchedule.prefetchPreview : [],
        resourceSchedule: resourceSchedule,
        totalIntegrated: this.integratedSceneCount,
        lastIntegratedCount: this.lastLoadMetrics.lastIntegratedCount,
        lastBatchMs: this.lastLoadMetrics.lastBatchMs,
        lastIntegratedAt: this.lastLoadMetrics.lastIntegratedAt,
        lastTextureTaskMs: this.lastTextureTaskMs,
        staleCachedCount: this.lastLoadMetrics.staleCachedCount,
        staleDroppedCount: this.lastLoadMetrics.staleDroppedCount,
        promotedGlbCount: this.lastLoadMetrics.promotedGlbCount,
        promotedParseCount: this.lastLoadMetrics.promotedParseCount,
        promotedInsertionCount: this.lastLoadMetrics.promotedInsertionCount,
        httpAdaptive: Object.assign({}, this.httpAdaptiveState, {
          enabled: this.httpAdaptiveConcurrencyEnabled,
          minConcurrency: this.httpAdaptiveMinConcurrency,
          maxConcurrency: this.httpAdaptiveMaxConcurrency,
          prefetchMinConcurrency: this.httpAdaptivePrefetchMinConcurrency,
          prefetchMaxConcurrency: this.httpAdaptivePrefetchMaxConcurrency,
        }),
      },
      neural: {
        cullingMode: this.cullingMode,
        enabled: this.useNeuralPVS,
        neuralBackend: this.neuralBackend,
        debugNoCache: this.neuralDebugNoCache,
        backend: this.neuralPVS ? this.neuralPVS.backend : null,
        ready: this.neuralPVS ? this.neuralPVS.isReady : false,
        modelInfo: this.neuralPVS ? (this.neuralPVS.modelInfo || (this.neuralPVS.lastPredictTimings ? this.neuralPVS.lastPredictTimings.modelInfo : null)) : null,
        initTimings: this.neuralPVS ? this.neuralPVS.lastInitTimings : null,
        predictTimings: this.neuralPVS ? this.neuralPVS.lastPredictTimings : null,
        filterTimings: this.neuralPVS ? this.neuralPVS.lastFilterTimings : null,
        idMode: this.neuralPVSIdMode,
        resourceTransport: 'http',
        cpuPerfMode: this.cpuPerfMode,
        renderVisibility: this.renderVisibilitySystem ? this.renderVisibilitySystem.lastStats : null,
        renderRefresh: this.lastRenderRefreshStats,
        staticBatching: this.staticSceneOptimizer.getStats(),
        instancedBindings: this._getInstancedBindingStats(),
        actualRender: this._getActualRenderStats(),
        renderPolicy: this.getNeuralRenderPolicy(),
        downloadPlanMode: this.getNeuralDownloadPlanMode(),
        priorityScheduler: schedulerStats,
        predictionGate: this.neuralPredictionGate ? {
          ...this.neuralPredictionGate.getState(),
          forceNext: Boolean(this.forceNextNeuralPrediction),
        } : null,
        frozenInspect: {
          active: this.frozenPredictionInspectActive,
          queued: this.frozenPredictionInspectQueued,
          total: this.frozenPredictionInspectTotal,
          componentCount: this.frozenPredictionInspectComponentIds.length,
        },
        currentEpoch: this.currentNeuralPredictionEpoch,
      }
    };
  }
}

