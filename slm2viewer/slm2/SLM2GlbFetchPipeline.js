import { SLM2VisibilityRuntime } from './SLM2VisibilityRuntime.js';

export class SLM2GlbFetchPipeline extends SLM2VisibilityRuntime {
  _getPendingSceneInsertionCount()
  {
    return Math.max(0, this.pendingSceneInsertions.length - Number(this.pendingSceneInsertionCursor || 0));
  }

  _compactPendingSceneInsertionsIfNeeded(force = false)
  {
    var cursor = Number(this.pendingSceneInsertionCursor || 0);
    if (cursor <= 0)
    {
      return;
    }

    if (force || cursor >= 64 || cursor >= this.pendingSceneInsertions.length * 0.5)
    {
      this.pendingSceneInsertions = this.pendingSceneInsertions.slice(cursor);
      this.pendingSceneInsertionCursor = 0;
    }
  }

  _getAdaptiveIntegrationBudget()
  {
    if (!this.useNeuralPVS)
    {
      return {
        budgetMs: Math.max(1, this.loadIntegrationBudgetMs),
        maxCount: Number.POSITIVE_INFINITY,
      };
    }

    var dt = Number(this.lastFrameDt || 0);
    var budgetMs = this.loadIntegrationBudgetMs;
    var maxCount = this.cpuPerfMode === 'mobile' ? 2 : 4;

    if (dt > 33)
    {
      budgetMs = 1;
      maxCount = 1;
    }
    else if (dt > 20)
    {
      budgetMs = 3;
      maxCount = 2;
    }

    if (this.cpuPerfMode === 'mobile')
    {
      budgetMs = Math.min(budgetMs, dt > 20 ? budgetMs : 2);
      maxCount = Math.min(maxCount, 2);
    }

    return {
      budgetMs: Math.max(1, budgetMs),
      maxCount: Math.max(1, maxCount),
    };
  }

  _parseIntParam(params, key, fallback, minValue, maxValue)
  {
    var value = params && params[key] != null ? Number(params[key]) : fallback;
    if (!Number.isFinite(value))
    {
      value = fallback;
    }
    value = Math.round(value);
    value = Math.max(minValue, value);
    if (maxValue != null)
    {
      value = Math.min(maxValue, value);
    }
    return value;
  }

  _getActualRenderStats()
  {
    var nowMs = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
    if (this.actualRenderStatsCache && (nowMs - Number(this.actualRenderStatsAt || 0)) < 500)
    {
      return this.actualRenderStatsCache;
    }

    var stats = {
      objectCount: 0,
      visibleObjectCount: 0,
      meshCount: 0,
      visibleMeshCount: 0,
      instancedMeshCount: 0,
      visibleInstancedMeshCount: 0,
      drawnInstanceCount: 0,
      visibleTriangleEstimate: 0,
    };
    var root = this.rootScene;
    if (!root)
    {
      this.actualRenderStatsCache = stats;
      this.actualRenderStatsAt = nowMs;
      return stats;
    }

    function estimateTriangles(mesh)
    {
      var geometry = mesh && mesh.geometry ? mesh.geometry : null;
      if (!geometry)
      {
        return 0;
      }
      if (geometry.index && geometry.index.count != null)
      {
        return Math.floor(Number(geometry.index.count || 0) / 3);
      }
      if (geometry.attributes && geometry.attributes.position && geometry.attributes.position.count != null)
      {
        return Math.floor(Number(geometry.attributes.position.count || 0) / 3);
      }
      return 0;
    }

    function visit(object, parentVisible)
    {
      if (!object)
      {
        return;
      }
      var effectiveVisible = parentVisible && object.visible !== false;
      stats.objectCount++;
      if (effectiveVisible)
      {
        stats.visibleObjectCount++;
      }
      if (object.isMesh)
      {
        stats.meshCount++;
        if (object.isInstancedMesh)
        {
          stats.instancedMeshCount++;
        }
        if (effectiveVisible)
        {
          var instanceCount = object.isInstancedMesh ? Math.max(0, Number(object.count || 0)) : 1;
          stats.visibleMeshCount++;
          if (object.isInstancedMesh)
          {
            stats.visibleInstancedMeshCount++;
            stats.drawnInstanceCount += instanceCount;
          }
          stats.visibleTriangleEstimate += estimateTriangles(object) * instanceCount;
        }
      }
      var children = object.children || [];
      for (var i = 0; i < children.length; ++i)
      {
        visit(children[i], effectiveVisible);
      }
    }

    visit(root, root.visible !== false);
    this.actualRenderStatsCache = stats;
    this.actualRenderStatsAt = nowMs;
    return stats;
  }

  _parseFloatParam(params, key, fallback, minValue, maxValue)
  {
    var value = params && params[key] != null ? Number(params[key]) : fallback;
    if (!Number.isFinite(value))
    {
      value = fallback;
    }
    value = Math.max(minValue, value);
    if (maxValue != null)
    {
      value = Math.min(maxValue, value);
    }
    return value;
  }

  _getNetworkDownlinkMbps()
  {
    if (typeof navigator === 'undefined' || navigator.connection == null)
    {
      return null;
    }
    var downlink = Number(navigator.connection.downlink);
    return Number.isFinite(downlink) && downlink > 0 ? downlink : null;
  }

  _concurrencyFromMbps(mbps, mobile = false)
  {
    if (!Number.isFinite(Number(mbps)) || Number(mbps) <= 0)
    {
      return mobile ? 8 : 16;
    }
    if (mobile)
    {
      if (mbps <= 5) return 4;
      if (mbps <= 15) return 6;
      if (mbps <= 40) return 8;
      return 10;
    }
    if (mbps <= 5) return 6;
    if (mbps <= 15) return 10;
    if (mbps <= 40) return 16;
    if (mbps <= 80) return 20;
    return 24;
  }

  _configureHttpAdaptiveConcurrency()
  {
    var mobile = this.cpuPerfMode === 'mobile';
    this.httpAdaptiveMinConcurrency = mobile ? 4 : 6;
    this.httpAdaptiveMaxConcurrency = mobile ? 10 : 24;
    this.httpAdaptivePrefetchMinConcurrency = mobile ? 2 : 4;
    this.httpAdaptivePrefetchMaxConcurrency = mobile ? 4 : 8;
    var downlink = this._getNetworkDownlinkMbps();
    var initial = this._concurrencyFromMbps(downlink, mobile);
    initial = Math.max(this.httpAdaptiveMinConcurrency, Math.min(this.httpAdaptiveMaxConcurrency, initial));
    this.httpAdaptiveState.current = initial;
    this.httpAdaptiveState.prefetchCurrent = Math.max(
      this.httpAdaptivePrefetchMinConcurrency,
      Math.min(this.httpAdaptivePrefetchMaxConcurrency, Math.ceil(initial * 0.5))
    );
    this.httpAdaptiveState.downlinkMbps = downlink;
    this.httpAdaptiveState.lastReason = downlink ? 'navigator-downlink' : 'default';
    this.neuralDirectLoadConcurrency = this.httpAdaptiveState.current;
    this.neuralPrefetchDirectLoadConcurrency = this.httpAdaptiveState.prefetchCurrent;
  }

  _getLatestResourceTiming(url, startedAt)
  {
    if (typeof performance === 'undefined' || typeof performance.getEntriesByName !== 'function' || !url)
    {
      return null;
    }
    var entries = performance.getEntriesByName(url, 'resource');
    if (!entries || entries.length === 0)
    {
      return null;
    }
    for (var i = entries.length - 1; i >= 0; --i)
    {
      var entry = entries[i];
      if (entry && (!Number.isFinite(startedAt) || entry.startTime >= startedAt - 20))
      {
        return entry;
      }
    }
    return entries[entries.length - 1] || null;
  }

  _recordHttpDirectLoadSample(url, startedAt, endedAt, modelDesc, success)
  {
    if (!this.httpAdaptiveConcurrencyEnabled)
    {
      return;
    }
    var state = this.httpAdaptiveState;
    var elapsedMs = Math.max(0, Number(endedAt || performance.now()) - Number(startedAt || endedAt || 0));
    var timing = this._getLatestResourceTiming(url, startedAt);
    var networkMs = timing && Number.isFinite(timing.duration) && timing.duration > 0
      ? timing.duration
      : elapsedMs;
    var bytes = 0;
    if (timing)
    {
      bytes = Number(timing.transferSize || timing.encodedBodySize || timing.decodedBodySize || 0);
    }
    if ((!bytes || bytes <= 0) && modelDesc && Number(modelDesc.sizeKB) > 0)
    {
      bytes = Number(modelDesc.sizeKB) * 1024;
    }
    var mbps = bytes > 0 && networkMs > 0
      ? (bytes * 8) / (networkMs * 1000)
      : 0;
    var alpha = state.samples <= 0 ? 1 : 0.25;
    state.samples += 1;
    state.ewmaLoadMs = state.ewmaLoadMs > 0 ? (state.ewmaLoadMs * (1 - alpha) + elapsedMs * alpha) : elapsedMs;
    state.ewmaNetworkMs = state.ewmaNetworkMs > 0 ? (state.ewmaNetworkMs * (1 - alpha) + networkMs * alpha) : networkMs;
    if (mbps > 0)
    {
      state.ewmaMbps = state.ewmaMbps > 0 ? (state.ewmaMbps * (1 - alpha) + mbps * alpha) : mbps;
      state.lastMbps = mbps;
    }
    state.lastBytes = bytes;
    state.lastResourceUrl = url || null;
    state.lastSuccess = Boolean(success);
    this._updateAdaptiveHttpConcurrency('sample');
  }

  _updateAdaptiveHttpConcurrency(reason = 'tick')
  {
    if (!this.httpAdaptiveConcurrencyEnabled)
    {
      return;
    }
    var now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    var state = this.httpAdaptiveState;
    if (reason !== 'sample' && state.lastAdjustmentAt > 0 && now - state.lastAdjustmentAt < 500)
    {
      return;
    }
    var mobile = this.cpuPerfMode === 'mobile';
    var measuredMbps = Number(state.ewmaMbps || 0);
    var downlink = this._getNetworkDownlinkMbps();
    if (downlink)
    {
      state.downlinkMbps = downlink;
    }
    var bandwidthMbps = measuredMbps > 0 ? measuredMbps : (downlink || state.downlinkMbps || 0);
    var desired = this._concurrencyFromMbps(bandwidthMbps, mobile);
    var pressure = 'bandwidth';
    var pendingInsertions = this._getPendingSceneInsertionCount();
    var frameDt = Number(this.lastFrameDt || 0);
    var pendingRatio = pendingInsertions / Math.max(1, Number(this.maxPendingSceneInsertions || 1));

    if (frameDt > 45 || pendingRatio >= 0.75)
    {
      desired = Math.min(desired, Math.max(this.httpAdaptiveMinConcurrency, Number(state.current || desired) - 3));
      pressure = frameDt > 45 ? 'frame-pressure' : 'integration-backpressure';
    }
    else if (frameDt > 32 || pendingRatio >= 0.5)
    {
      desired = Math.min(desired, Math.max(this.httpAdaptiveMinConcurrency, Number(state.current || desired) - 1));
      pressure = frameDt > 32 ? 'frame-soft-pressure' : 'integration-soft-backpressure';
    }
    else if (state.samples >= 3 && state.ewmaLoadMs > 4500)
    {
      desired = Math.min(desired, Math.max(this.httpAdaptiveMinConcurrency, Number(state.current || desired) - 1));
      pressure = 'slow-load';
    }
    else if (measuredMbps <= 0 && state.samples >= 4 && state.ewmaLoadMs > 0 && frameDt <= 28 && pendingRatio < 0.35)
    {
      if (state.ewmaLoadMs < 1000)
      {
        desired = Math.max(desired, Number(state.current || desired) + 2);
        pressure = 'fast-load-ramp-up';
      }
      else if (state.ewmaLoadMs < 1800)
      {
        desired = Math.max(desired, Number(state.current || desired) + 1);
        pressure = 'load-ramp-up';
      }
    }

    desired = Math.max(this.httpAdaptiveMinConcurrency, Math.min(this.httpAdaptiveMaxConcurrency, desired));
    var current = Number(state.current || desired);
    var next = current;
    if (desired > current)
    {
      next = Math.min(desired, current + 2);
    }
    else if (desired < current)
    {
      next = Math.max(desired, current - 2);
    }
    state.current = Math.round(next);
    state.prefetchCurrent = Math.max(
      this.httpAdaptivePrefetchMinConcurrency,
      Math.min(this.httpAdaptivePrefetchMaxConcurrency, Math.ceil(state.current * 0.5))
    );
    state.lastAdjustmentAt = now;
    state.lastReason = pressure;
    this.neuralDirectLoadConcurrency = state.current;
    this.neuralPrefetchDirectLoadConcurrency = state.prefetchCurrent;
  }

  _getHttpDirectLoadConcurrency(baseConcurrency, prefetchOnly = false)
  {
    if (!this.httpAdaptiveConcurrencyEnabled || (!this.useNeuralPVS && !this.fullLoadMode))
    {
      return baseConcurrency;
    }
    this._updateAdaptiveHttpConcurrency('tick');
    var state = this.httpAdaptiveState;
    var adaptive = prefetchOnly ? state.prefetchCurrent : state.current;
    if (!Number.isFinite(Number(adaptive)) || adaptive <= 0)
    {
      adaptive = baseConcurrency;
    }
    return Math.max(1, Math.min(baseConcurrency, Math.round(adaptive)));
  }

  _configureGlbResourcePipelineOptions(params)
  {
    var mobile = this.cpuPerfMode === 'mobile';
    this.glbParseConcurrency = this._parseIntParam(params, 'glbParseConcurrency', mobile ? 1 : 2, 1, 4);
    var maxPendingParseMB = this._parseIntParam(params, 'glbMaxPendingParseMB', mobile ? 48 : 96, 8, 256);
    this.maxPendingGlbParseBytes = maxPendingParseMB * 1024 * 1024;
  }

  _promoteCurrentRenderGlbs(glbIds)
  {
    var promotedIds = this._normalizeIdList(glbIds);
    if (promotedIds.length === 0)
    {
      return { promotedGlbCount: 0, parseCount: 0, insertionCount: 0 };
    }

    var promotedInfos = [];
    var promotedHashes = new Set();
    for (var index = 0; index < promotedIds.length; ++index)
    {
      var globalGlbId = promotedIds[index];
      var stored = this.neuralDownloadInfoByGlbId.get(globalGlbId);
      var modelInfo = Object.assign({}, stored || {
        id: globalGlbId,
        weight: 1,
        idMode: 'global-glb',
      }, {
        prefetch: false,
        deferredVisible: false,
        predictionEpoch: this.currentNeuralPredictionEpoch,
        renderPriorityPromotion: true,
      });
      var decoded = this.decodeModelInfo(modelInfo);
      if (!decoded || !decoded.hash)
      {
        continue;
      }
      promotedInfos.push(modelInfo);
      promotedHashes.add(decoded.hash);
      this.neuralDownloadInfoByGlbId.set(globalGlbId, modelInfo);
    }

    if (promotedInfos.length === 0)
    {
      return { promotedGlbCount: 0, parseCount: 0, insertionCount: 0 };
    }

    var promotedCount = this.glbResourceScheduler.promote(
      this._makeGlbScheduleEntries(promotedInfos, 'urgent')
    );

    var parsePrefix = this.pendingGlbParseQueue.slice(0, this.pendingGlbParseCursor);
    var parseUrgent = [];
    var parseRest = [];
    for (var parseIndex = this.pendingGlbParseCursor; parseIndex < this.pendingGlbParseQueue.length; ++parseIndex)
    {
      var parseItem = this.pendingGlbParseQueue[parseIndex];
      if (parseItem && parseItem.modelDesc && promotedHashes.has(parseItem.modelDesc.hash))
      {
        parseItem.modelDesc.prefetch = false;
        parseUrgent.push(parseItem);
      }
      else
      {
        parseRest.push(parseItem);
      }
    }
    this.pendingGlbParseQueue = parsePrefix.concat(parseUrgent, parseRest);

    var insertionPrefix = this.pendingSceneInsertions.slice(0, this.pendingSceneInsertionCursor);
    var insertionUrgent = [];
    var insertionRest = [];
    for (var insertionIndex = this.pendingSceneInsertionCursor;
      insertionIndex < this.pendingSceneInsertions.length; ++insertionIndex)
    {
      var insertionItem = this.pendingSceneInsertions[insertionIndex];
      if (insertionItem && promotedHashes.has(insertionItem.hash))
      {
        insertionItem.prefetch = false;
        if (insertionItem.modelDesc) insertionItem.modelDesc.prefetch = false;
        insertionUrgent.push(insertionItem);
      }
      else
      {
        insertionRest.push(insertionItem);
      }
    }
    this.pendingSceneInsertions = insertionPrefix.concat(insertionUrgent, insertionRest);

    this.lastLoadMetrics.promotedGlbCount = Number(this.lastLoadMetrics.promotedGlbCount || 0)
      + promotedCount;
    this.lastLoadMetrics.promotedParseCount = Number(this.lastLoadMetrics.promotedParseCount || 0)
      + parseUrgent.length;
    this.lastLoadMetrics.promotedInsertionCount = Number(this.lastLoadMetrics.promotedInsertionCount || 0)
      + insertionUrgent.length;
    this.processGlbParseQueue();
    this.processLoadingList();
    var scheduleSnapshot = this.glbResourceScheduler.getSnapshot();
    return {
      promotedGlbCount: promotedCount,
      queuedCount: scheduleSnapshot.queuedImmediate,
      parseCount: parseUrgent.length,
      insertionCount: insertionUrgent.length,
    };
  }

  _cancelAllDirectDownloads()
  {
    if (!this.directDownloadControllers)
    {
      return;
    }
    this.directDownloadControllers.forEach(function(entry)
    {
      entry.controller.abort();
    });
    this.directDownloadControllers.clear();
  }

  _clearPendingGlbParseQueue()
  {
    for (var index = this.pendingGlbParseCursor; index < this.pendingGlbParseQueue.length; ++index)
    {
      var item = this.pendingGlbParseQueue[index];
      this._clearInflightHash(item && item.modelDesc);
    }
    this.pendingGlbParseQueue = [];
    this.pendingGlbParseCursor = 0;
    this.pendingGlbParseBytes = 0;
  }

  _setCurrentNeuralWorkingSet(modelInfos, idMode)
  {
    this.currentNeuralPredictionEpoch++;
    this._replaceRenderGlbStateFromModelInfos(modelInfos);
    this.lastRenderRefreshStats = this.renderVisibilitySystem.setWorkingSet(
      modelInfos,
      idMode,
      this.currentNeuralPredictionEpoch
    );
    return this.currentNeuralPredictionEpoch;
  }

  _updateCurrentNeuralRenderSet(modelInfos, idMode)
  {
    this._replaceRenderGlbStateFromModelInfos(modelInfos);
    this.lastRenderRefreshStats = this.renderVisibilitySystem.setWorkingSet(
      modelInfos,
      idMode,
      this.currentNeuralPredictionEpoch
    );
    return this.lastRenderRefreshStats;
  }

  _replaceRenderGlbStateFromModelInfos(modelInfos)
  {
    var glbIds = (modelInfos || []).map(function(item)
    {
      return Number(item && item.id);
    });
    return this.renderGlbState.replace(glbIds, this._getGlbBitCount());
  }

  _isHashInCurrentNeuralRenderSet(hash)
  {
    var currentIdMode = this.renderVisibilitySystem ? this.renderVisibilitySystem.currentIdMode : null;
    var gatingEnabled = currentIdMode === 'global-glb' || currentIdMode === 'global-glb-priority';
    if (!this.useNeuralPVS || !gatingEnabled || hash == null)
    {
      return true;
    }

    return Boolean(
      this.renderVisibilitySystem &&
      this.renderVisibilitySystem.workingSetHashes &&
      this.renderVisibilitySystem.workingSetHashes.has(hash)
    );
  }

  _isHashWantedForDownload(hash)
  {
    var currentIdMode = this.renderVisibilitySystem ? this.renderVisibilitySystem.currentIdMode : null;
    var gatingEnabled = currentIdMode === 'global-glb' || currentIdMode === 'global-glb-priority';
    if (!this.useNeuralPVS || !gatingEnabled)
    {
      return true;
    }

    return this.glbResourceScheduler.isWanted(hash);
  }

  _disposeUnintegratedScene(scene)
  {
    if (!scene)
    {
      return;
    }

    scene.removeFromParent();
    scene.traverse(function(node)
    {
      if (!node.isMesh)
      {
        return;
      }

      if (node.geometry)
      {
        node.geometry.dispose();
      }

      var materials = Array.isArray(node.material) ? node.material : [node.material];
      for (var i = 0; i < materials.length; ++i)
      {
        var material = materials[i];
        if (!material)
        {
          continue;
        }

        for (var key in material)
        {
          var value = material[key];
          if (value && value.isTexture)
          {
            value.dispose();
          }
        }

        if (typeof material.dispose === 'function')
        {
          material.dispose();
        }
      }
    });
  }

  _getPriorityInfoFromModelDesc(modelDesc)
  {
    if (!modelDesc)
    {
      return null;
    }

    if (modelDesc.priorityInfo)
    {
      return modelDesc.priorityInfo;
    }

    if (modelDesc.sourceModelInfo && modelDesc.sourceModelInfo.priorityInfo)
    {
      return modelDesc.sourceModelInfo.priorityInfo;
    }

    return null;
  }

  _canAcceptStaleCache(phase)
  {
    if (!this.modelCacheMgr)
    {
      return false;
    }

    this.modelCacheMgr.getStatsDetail();
    var memoryLimitKB = Number(this.modelCacheMgr.MaxMemoryUsageInKB || 0);
    var memoryUsedKB = Number(this.modelCacheMgr.statsDetail ? this.modelCacheMgr.statsDetail.memoryUsed : 0);
    if (Number.isFinite(memoryLimitKB) && memoryLimitKB > 0 &&
        Number.isFinite(memoryUsedKB) &&
        memoryUsedKB >= memoryLimitKB * this.neuralStaleCacheMaxMemoryRatio)
    {
      return false;
    }

    return true;
  }

  _shouldRetainStaleDownload(modelDesc, phase = 'parsed')
  {
    if (!this.useNeuralPVS || !this.neuralStaleCacheEnabled)
    {
      return false;
    }

    if (!modelDesc || !modelDesc.hash)
    {
      return false;
    }

    if (this.modelCacheMgr && this.modelCacheMgr.objectsPool && this.modelCacheMgr.objectsPool[modelDesc.hash])
    {
      return true;
    }

    if (!this._canAcceptStaleCache(phase))
    {
      return false;
    }

    var info = this._getPriorityInfoFromModelDesc(modelDesc);
    if (!info)
    {
      return false;
    }

    var projectedArea = Number(info.projectedArea || 0);
    var priority = Math.max(Number(info.priority || 0), Number(modelDesc.weight || 0));
    var tier = info.tier || '';
    var wasImmediate = tier === 'visible-now';
    var wasDeferredVisible = Boolean(modelDesc.deferredVisible || (info && info.deferredVisible));
    var wasUsefulPrefetch = tier === 'prefetch' &&
      projectedArea >= this.neuralStaleCachePrefetchAreaThreshold &&
      phase !== 'http-raw';

    return wasImmediate ||
      wasDeferredVisible ||
      wasUsefulPrefetch ||
      projectedArea >= this.neuralStaleCacheProjectedAreaThreshold ||
      priority >= this.neuralStaleCachePriorityThreshold;
  }

  _recordStaleDownloadDrop(modelDesc, reason)
  {
    this.lastLoadMetrics.staleDroppedCount = Number(this.lastLoadMetrics.staleDroppedCount || 0) + 1;
  }

  _cacheStaleLoadedGltf(gltf, modelDesc = null, reason = 'stale-cache')
  {
    var scene = gltf && (gltf.scene || (gltf.scenes && gltf.scenes[0]));
    if (!scene || !this._shouldRetainStaleDownload(modelDesc, 'parsed'))
    {
      return false;
    }

    var hash = modelDesc && modelDesc.hash ? modelDesc.hash : null;
    if (hash && this.modelCacheMgr.objectsPool[hash])
    {
      this._disposeUnintegratedScene(scene);
      this._clearInflightHash(modelDesc);
      this.pendingSceneInsertionHashes.delete(hash);
      this.lastLoadMetrics.staleCachedCount = Number(this.lastLoadMetrics.staleCachedCount || 0) + 1;
      return true;
    }

    if (scene.parent)
    {
      scene.removeFromParent();
    }

    var extras = gltf && gltf.asset && gltf.asset.extras ? gltf.asset.extras : null;
    if (!extras)
    {
      extras = {};
      if (!gltf.asset) gltf.asset = {};
      gltf.asset.extras = extras;
    }
    if (!extras.hashCode && hash)
    {
      extras.hashCode = hash;
    }
    if (extras.sizeKB == null)
    {
      extras.sizeKB = modelDesc && modelDesc.sizeKB != null ? modelDesc.sizeKB : 0;
    }

    if (!extras.hashCode)
    {
      return false;
    }

    this.modelCacheMgr.addObject(extras, scene, {
      isVisible: false,
      isInScene: false,
    });

    var scope = this;
    scene.traverse(function(node)
    {
      if (!node.isMesh) return;
      node.material = scope.fetchCachedMaterial(node.material, extras);
    });

    this._clearInflightHash(modelDesc);
    if (hash)
    {
      this.pendingSceneInsertionHashes.delete(hash);
    }
    this.lastLoadMetrics.staleCachedCount = Number(this.lastLoadMetrics.staleCachedCount || 0) + 1;
    if (this.neuralDebugLogs)
    {
      console.log('[SLM2Loader] Cached stale neural asset in background', {
        hash: extras.hashCode,
        reason: reason,
        priorityInfo: this._getPriorityInfoFromModelDesc(modelDesc),
      });
    }
    return true;
  }

  _pruneStalePendingInsertions()
  {
    if (!this.useNeuralPVS)
    {
      return;
    }

    this._pruneStaleGlbParseQueue();

    if (!Array.isArray(this.pendingSceneInsertions) || this._getPendingSceneInsertionCount() === 0)
    {
      return;
    }

    var kept = [];
    var cursor = Number(this.pendingSceneInsertionCursor || 0);
    for (var i = cursor; i < this.pendingSceneInsertions.length; ++i)
    {
      var item = this.pendingSceneInsertions[i];
      if (!item || item.hash == null || this._isHashWantedForDownload(item.hash))
      {
        kept.push(item);
        continue;
      }

      this.pendingSceneInsertionHashes.delete(item.hash);
      if (!this._cacheStaleLoadedGltf(item.gltf, item.modelDesc || null, 'stale-pending-integration'))
      {
        this.glbResourceScheduler.markDiscarded(item.hash);
        this._recordStaleDownloadDrop(item.modelDesc || { hash: item.hash }, 'stale-pending-integration');
        this._disposeUnintegratedScene(item.gltf && (item.gltf.scene || (item.gltf.scenes && item.gltf.scenes[0])));
      }
    }

    this.pendingSceneInsertions = kept;
    this.pendingSceneInsertionCursor = 0;
    this.lastLoadMetrics.queueLength = this._getPendingSceneInsertionCount();
  }

  _pruneStaleGlbParseQueue()
  {
    if (!Array.isArray(this.pendingGlbParseQueue) || this._getPendingGlbParseCount() === 0)
    {
      return;
    }

    var kept = [];
    var cursor = Number(this.pendingGlbParseCursor || 0);
    for (var i = cursor; i < this.pendingGlbParseQueue.length; ++i)
    {
      var item = this.pendingGlbParseQueue[i];
      if (!item || !item.modelDesc || item.modelDesc.hash == null || this._isHashWantedForDownload(item.modelDesc.hash))
      {
        kept.push(item);
        continue;
      }

      if (this._shouldRetainStaleDownload(item.modelDesc, 'http-raw'))
      {
        kept.push(item);
        continue;
      }

      this.pendingGlbParseBytes = Math.max(0, this.pendingGlbParseBytes - Number(item.byteLength || 0));
      this._clearInflightHash(item.modelDesc);
      this.glbResourceScheduler.markDiscarded(item.modelDesc.hash);
      this._recordStaleDownloadDrop(item.modelDesc, 'stale-glb-parse-queue');
    }

    this.pendingGlbParseQueue = kept;
    this.pendingGlbParseCursor = 0;
  }

  _getPendingGlbParseCount()
  {
    return Math.max(0, this.pendingGlbParseQueue.length - Number(this.pendingGlbParseCursor || 0));
  }

  _compactPendingGlbParseQueueIfNeeded(force = false)
  {
    var cursor = Number(this.pendingGlbParseCursor || 0);
    if (cursor <= 0)
    {
      return;
    }

    if (force || cursor >= 64 || cursor >= this.pendingGlbParseQueue.length * 0.5)
    {
      this.pendingGlbParseQueue = this.pendingGlbParseQueue.slice(cursor);
      this.pendingGlbParseCursor = 0;
    }
  }

  _clearInflightHash(modelDesc)
  {
    if (modelDesc && modelDesc.hash)
    {
      this.inflightModelHashes.delete(modelDesc.hash);
    }
  }

  processGlbParseQueue()
  {
    if (!this.pendingGlbParseQueue || this._getPendingGlbParseCount() === 0)
    {
      this._compactPendingGlbParseQueueIfNeeded(true);
      return;
    }

    if (this.gltfLoaders == undefined || this.gltfLoaders.length === 0)
    {
      return;
    }

    while (this.activeGlbParseCount < this.glbParseConcurrency &&
      this.pendingGlbParseCursor < this.pendingGlbParseQueue.length &&
      this._getPendingSceneInsertionCount() < this.maxPendingSceneInsertions)
    {
      var item = this.pendingGlbParseQueue[this.pendingGlbParseCursor++];
      if (!item)
      {
        continue;
      }

      if (item.modelDesc && item.modelDesc.hash && !this._isHashWantedForDownload(item.modelDesc.hash))
      {
        if (!this._shouldRetainStaleDownload(item.modelDesc, 'http-raw'))
        {
          this.pendingGlbParseBytes = Math.max(0, this.pendingGlbParseBytes - Number(item.byteLength || 0));
          this._clearInflightHash(item.modelDesc);
          this.glbResourceScheduler.markDiscarded(item.modelDesc.hash);
          this._recordStaleDownloadDrop(item.modelDesc, 'stale-glb-before-parse');
          continue;
        }
      }

      this._parseGlbBuffer(item);
    }

    this._compactPendingGlbParseQueueIfNeeded();
  }

  _parseGlbBuffer(item)
  {
    var scope = this;
    var loader = this.gltfLoaders[this.parseLoaderCursor % this.gltfLoaders.length];
    this.parseLoaderCursor++;
    this.activeGlbParseCount++;

    loader.parse(item.buffer, item.basePath || '', function(gltf)
    {
      scope.activeGlbParseCount = Math.max(0, scope.activeGlbParseCount - 1);
      if (item.pipelineSerial === scope.resourcePipelineSerial)
      {
        scope.pendingGlbParseBytes = Math.max(0, scope.pendingGlbParseBytes - Number(item.byteLength || 0));
      }

      if (item.pipelineSerial !== scope.resourcePipelineSerial)
      {
        scope._disposeUnintegratedScene(gltf && (gltf.scene || (gltf.scenes && gltf.scenes[0])));
        scope.processGlbParseQueue();
        scope.processLoadingList();
        return;
      }

      if (item.modelDesc && item.modelDesc.hash && !scope._isHashWantedForDownload(item.modelDesc.hash))
      {
        if (!scope._cacheStaleLoadedGltf(gltf, item.modelDesc, 'stale-glb-after-parse'))
        {
          scope._clearInflightHash(item.modelDesc);
          scope.glbResourceScheduler.markDiscarded(item.modelDesc.hash);
          scope._recordStaleDownloadDrop(item.modelDesc, 'stale-glb-after-parse');
          scope._disposeUnintegratedScene(gltf && (gltf.scene || (gltf.scenes && gltf.scenes[0])));
        }
        scope.processGlbParseQueue();
        scope.processLoadingList();
        return;
      }

      scope.processLoadedGltf(gltf, item.modelDesc);
      scope.processGlbParseQueue();
      scope.processLoadingList();
    }, function(err)
    {
      scope.activeGlbParseCount = Math.max(0, scope.activeGlbParseCount - 1);
      if (item.pipelineSerial === scope.resourcePipelineSerial)
      {
        scope.pendingGlbParseBytes = Math.max(0, scope.pendingGlbParseBytes - Number(item.byteLength || 0));
      }
      if (item.pipelineSerial !== scope.resourcePipelineSerial)
      {
        scope.processGlbParseQueue();
        scope.processLoadingList();
        return;
      }
      console.error('[LoadError][glb-parse]', item.modelDesc ? item.modelDesc.hash : null, err);
      scope._clearInflightHash(item.modelDesc);
      if (item.modelDesc && item.modelDesc.hash)
      {
        scope.glbResourceScheduler.markFailed(item.modelDesc.hash, err);
      }
      scope.processGlbParseQueue();
      scope.processLoadingList();
    });
  }
}

