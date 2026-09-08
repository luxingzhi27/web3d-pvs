import { LoaderUtils } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { classifyGlbSchedule } from '../src/GlbResourceScheduler.js';
import { DRACO_LOADER, KTX2_LOADER } from './SLM2RuntimeAssets.js';
import { SLM2GlbFetchPipeline } from './SLM2GlbFetchPipeline.js';

export class SLM2GlbPipeline extends SLM2GlbFetchPipeline {
  updateLoading(dt)
  {
    if (this.debugLoadingMode)
    {
      this.processLoadingList();
    }
    else
    {
      if (this.cullingUpdateDelta == undefined)
      {
        this.cullingUpdateDelta = 1000;
      }
      this.cullingUpdateDelta += dt;
      if (this.cameraUpdatePending && this.cullingUpdateDelta > 80)
      {
        this.sceneCulling();
        this.cameraUpdatePending = false;
        this.cullingUpdateDelta = 0;
      }

      this.processLoadingList();
    }
  }

  getModelDesc(modelInfo)
  {
    var decodeInfo = this.decodeModelInfo(modelInfo);

    if (decodeInfo != null)
    {
      if (decodeInfo.url == null)
      {
        return null;
      }

      if (decodeInfo.prefetch && this.modelCacheMgr.objectsPool[decodeInfo.hash])
      {
        return null;
      }

      if (!decodeInfo.prefetch && this.modelCacheMgr.tryCacheHit(decodeInfo.hash, this.rootScene))
      {
        return null;
      }

      if (this.inflightModelHashes.has(decodeInfo.hash) || this.pendingSceneInsertionHashes.has(decodeInfo.hash))
      {
        return null;
      }

      return decodeInfo;
    }
    else
    {
      return null;
    }
  }

  _getRenderVisibilityOptions(idMode)
  {
    var renderOptions = {};
    if (this.fullLoadMode)
    {
      return renderOptions;
    }

    if ((this.useNeuralPVS || this.cullingMode === 'frustum') &&
        (idMode === 'global-glb-priority' || idMode === 'global-glb'))
    {
      renderOptions = {
        attachVisibleToScene: true,
        detachHiddenFromScene: false,
        renderRoot: this.rootScene,
      };
    }

    renderOptions.fallbackVisibleHashes = new Set();
    return renderOptions;
  }

  _startDirectModelDownload(modelInfo)
  {
    var scope = this;
    var decodedInfo = modelInfo != undefined ? scope.decodeModelInfo(modelInfo) : null;
    var modelDesc = modelInfo != undefined ? scope.getModelDesc(modelInfo) : null;
    var modelURL = modelDesc != null ? modelDesc.url : null;
    if (modelURL == null)
    {
      if (scope.useNeuralPVS && decodedInfo && decodedInfo.hash)
      {
        if (scope.modelCacheMgr.objectsPool[decodedInfo.hash])
        {
          scope.glbResourceScheduler.markResident(decodedInfo.hash);
        }
        else if (scope.pendingSceneInsertionHashes.has(decodedInfo.hash))
        {
          scope.glbResourceScheduler.markMounting(decodedInfo.hash);
        }
        else if (!scope.inflightModelHashes.has(decodedInfo.hash))
        {
          scope.glbResourceScheduler.markFailed(
            decodedInfo.hash,
            new Error('GLB descriptor is unavailable for a scheduled resource.')
          );
        }
      }
      return false;
    }
    var capturedModel = modelInfo;
    var capturedModelDesc = modelDesc;
    var capturedPipelineSerial = this.resourcePipelineSerial;
    var downloadStartedAt = performance.now();
    var controller = new AbortController();
    if (capturedModelDesc && capturedModelDesc.hash)
    {
      scope.inflightModelHashes.add(capturedModelDesc.hash);
      scope.directDownloadControllers.set(capturedModelDesc.hash, {
        controller: controller,
        modelDesc: capturedModelDesc,
      });
    }
    this.activeDirectLoadCount++;

    fetch(modelURL, {
      signal: controller.signal,
      mode: 'cors',
      credentials: 'same-origin',
    }).then(function(response)
    {
      if (!response.ok)
      {
        throw new Error('HTTP ' + response.status + ' for ' + modelURL);
      }
      var contentType = String(response.headers.get('content-type') || '').toLowerCase();
      if (contentType.indexOf('text/html') >= 0)
      {
        throw new Error('Expected GLB but received HTML from ' + modelURL);
      }
      return response.arrayBuffer();
    }).then(function(buffer)
    {
      var downloadEndedAt = performance.now();
      if (capturedPipelineSerial !== scope.resourcePipelineSerial)
      {
        return;
      }
      var byteLength = Number(buffer && buffer.byteLength || 0);
      if (capturedModelDesc && capturedModelDesc.hash)
      {
        scope.directLoadRetries.delete(capturedModelDesc.hash);
      }
      var sampleDesc = byteLength > 0
        ? Object.assign({}, capturedModelDesc || {}, { sizeKB: byteLength / 1024 })
        : capturedModelDesc;
      scope._recordHttpDirectLoadSample(modelURL, downloadStartedAt, downloadEndedAt, sampleDesc, true);

      if (capturedModelDesc && capturedModelDesc.hash && !scope._isHashWantedForDownload(capturedModelDesc.hash) &&
          !scope._shouldRetainStaleDownload(capturedModelDesc, 'http-raw'))
      {
        scope._clearInflightHash(capturedModelDesc);
        scope.glbResourceScheduler.markDiscarded(capturedModelDesc.hash);
        scope._recordStaleDownloadDrop(capturedModelDesc, 'stale-http-response');
        return;
      }
      if (capturedModelDesc && capturedModelDesc.hash)
      {
        scope.glbResourceScheduler.markParsing(capturedModelDesc.hash);
      }
      scope.pendingGlbParseQueue.push({
        buffer: buffer,
        modelDesc: capturedModelDesc,
        byteLength: byteLength,
        basePath: LoaderUtils.extractUrlBase(modelURL),
        source: 'http',
        receivedAt: downloadEndedAt,
        pipelineSerial: capturedPipelineSerial,
      });
      scope.pendingGlbParseBytes += byteLength;
      scope.processGlbParseQueue();
    }).catch(function(err)
    {
      var aborted = err && err.name === 'AbortError';
      if (!aborted && capturedPipelineSerial === scope.resourcePipelineSerial)
      {
        scope._recordHttpDirectLoadSample(modelURL, downloadStartedAt, performance.now(), capturedModelDesc, false);
        scope._clearInflightHash(capturedModelDesc);
        if (scope.useNeuralPVS && capturedModelDesc && capturedModelDesc.hash)
        {
          scope.glbResourceScheduler.markFailed(capturedModelDesc.hash, err);
        }
        else if (capturedModelDesc && capturedModelDesc.hash)
        {
          var retries = Number(scope.directLoadRetries.get(capturedModelDesc.hash) || 0);
          if (retries < 4)
          {
            scope.directLoadRetries.set(capturedModelDesc.hash, retries + 1);
            scope.modelToLoadList.push(capturedModel);
            scope.requestRender('glb-download-retry');
          }
        }
      }
      if (aborted && capturedModelDesc && capturedModelDesc.hash)
      {
        scope.glbResourceScheduler.markDiscarded(capturedModelDesc.hash);
      }
      if (!aborted && capturedPipelineSerial === scope.resourcePipelineSerial)
      {
        console.error('[LoadError]', modelURL, err);
      }
    }).finally(function()
    {
      if (capturedModelDesc && capturedModelDesc.hash)
      {
        var activeEntry = scope.directDownloadControllers.get(capturedModelDesc.hash);
        if (activeEntry && activeEntry.controller === controller)
        {
          scope.directDownloadControllers.delete(capturedModelDesc.hash);
        }
      }
      scope.activeDirectLoadCount = Math.max(0, scope.activeDirectLoadCount - 1);
      scope.processLoadingList();
    });

    return true;
  }

  processLoadingList()
  {
    var scope = this;
    var prefetchOnlyDirectLoads = this.useNeuralPVS &&
      !this.glbResourceScheduler.hasForegroundWork();
    var targetDirectLoadConcurrency = this.fullLoadMode
      ? this.fullLoadDirectLoadConcurrency
      : (this.useNeuralPVS
        ? (prefetchOnlyDirectLoads
          ? this.neuralPrefetchDirectLoadConcurrency
          : this.neuralDirectLoadConcurrency)
        : this.baseDirectLoadConcurrency);
    this.maxPendingSceneInsertions = this.useNeuralPVS
      ? this.neuralMaxPendingSceneInsertions
      : 24;
    this.loadIntegrationBudgetMs = this.useNeuralPVS
      ? this.neuralLoadIntegrationBudgetMs
      : 6;
    targetDirectLoadConcurrency = this._getHttpDirectLoadConcurrency(
      targetDirectLoadConcurrency,
      prefetchOnlyDirectLoads
    );

    if (this.gltfLoaders == undefined)
    {
      this.gltfLoaders = [];
    }

    while (this.gltfLoaders.length < this.glbParseConcurrency)
    {
      this.gltfLoaders.push(new GLTFLoader()
        .setCrossOrigin('anonymous')
        .setDRACOLoader(DRACO_LOADER)
        .setKTX2Loader(KTX2_LOADER.detectSupport(this.renderer))
        .setMeshoptDecoder(MeshoptDecoder));
    }

    this.processGlbParseQueue();

    if (this._getPendingSceneInsertionCount() >= this.maxPendingSceneInsertions
        || this.pendingGlbParseBytes >= this.maxPendingGlbParseBytes)
    {
      return;
    }

    if (this.useNeuralPVS)
    {
      while (this.activeDirectLoadCount < targetDirectLoadConcurrency)
      {
        var scheduledEntry = this.glbResourceScheduler.takeNext();
        if (!scheduledEntry)
        {
          break;
        }
        this._startDirectModelDownload(scheduledEntry.modelInfo);
      }
      return;
    }

    while (this.modelToLoadList.length > 0 &&
      this.activeDirectLoadCount < targetDirectLoadConcurrency)
    {
      var nextModel = this.modelToLoadList.pop();
      this._startDirectModelDownload(nextModel);
    }
  }

  processLoadedGltf(gltf, modelDesc = null)
  {
    var hash = modelDesc && modelDesc.hash
      ? modelDesc.hash
      : (gltf && gltf.asset && gltf.asset.extras ? gltf.asset.extras.hashCode : null);

    if (hash != null && !this._isHashWantedForDownload(hash))
    {
      if (this._cacheStaleLoadedGltf(gltf, modelDesc, 'stale-after-load'))
      {
        this.inflightModelHashes.delete(hash);
        return;
      }
      this.inflightModelHashes.delete(hash);
      this.glbResourceScheduler.markDiscarded(hash);
      this._recordStaleDownloadDrop(modelDesc || { hash: hash }, 'stale-after-load');
      this._disposeUnintegratedScene(gltf && (gltf.scene || (gltf.scenes && gltf.scenes[0])));
      return;
    }

    this.pendingSceneInsertions.push({
      gltf: gltf,
      hash: hash,
      modelDesc: modelDesc,
      prefetch: Boolean(modelDesc && modelDesc.prefetch),
      enqueuedAt: performance.now(),
    });
    if (hash != null)
    {
      this.glbResourceScheduler.markMounting(hash);
      this.pendingSceneInsertionHashes.add(hash);
      this.inflightModelHashes.delete(hash);
    }
    this.lastLoadMetrics.queueLength = this._getPendingSceneInsertionCount();
    this.requestRender('glb-ready-to-mount');
  }

  _integrateLoadedGltf(gltf, modelDesc = null)
  {
    var scope = this;
    const scene = gltf.scene || gltf.scenes[0];
    var isPrefetch = Boolean(modelDesc && modelDesc.prefetch);
    var keepLoadedRendered = this._isResidentRenderPolicy();
    var extras = gltf && gltf.asset && gltf.asset.extras ? gltf.asset.extras : null;
    if ((!extras || !extras.hashCode) && modelDesc && modelDesc.hash)
    {
      extras = {
        hashCode: modelDesc.hash,
        sizeKB: modelDesc.sizeKB != null ? modelDesc.sizeKB : 0,
      };
      if (!gltf.asset) gltf.asset = {};
      gltf.asset.extras = extras;
    }
    var loadedHash = extras && extras.hashCode ? extras.hashCode : (modelDesc && modelDesc.hash ? modelDesc.hash : null);
    var shouldRenderLoaded = keepLoadedRendered || (!isPrefetch && this._isHashInCurrentNeuralRenderSet(loadedHash));

    if (this.cullingMode === 'frustum' && !isPrefetch)
    {
      this._updateRuntimeVisibilityFrustum(this.activeCamera);
      var loadedGlobalGlbId = loadedHash != null ? this.globalGlbHashToId[loadedHash] : null;
      shouldRenderLoaded = loadedGlobalGlbId == null || this._globalGlbIntersectsCurrentFrustum(loadedGlobalGlbId);
    }

    if (modelDesc && modelDesc.hash && !this._isHashWantedForDownload(modelDesc.hash))
    {
      if (this._cacheStaleLoadedGltf(gltf, modelDesc, 'stale-before-integration'))
      {
        return null;
      }
      this.glbResourceScheduler.markDiscarded(modelDesc.hash);
      this._recordStaleDownloadDrop(modelDesc, 'stale-before-integration');
      this._disposeUnintegratedScene(scene);
      return null;
    }

    if (shouldRenderLoaded)
    {
      scope.rootScene.add(scene);
    }

    if (extras && extras.hashCode)
    {
      scope.modelCacheMgr.addObject(extras, scene, {
        isVisible: shouldRenderLoaded,
        isInScene: shouldRenderLoaded,
      });

      scene.traverse((node) =>
      {
        if (!node.isMesh) return;
        node.material = scope.fetchCachedMaterial(node.material, extras);
      });

      scope.staticSceneOptimizer.optimizeResident(extras.hashCode, gltf);
    }

    if (extras && extras.hashCode &&
        (this.instancedVisibilityBindingsByHash[extras.hashCode] ||
         this._isInstancedBindingExpected(extras.hashCode)))
    {
      // A GLB that is expected to contain multiple mapped instances must be
      // validated before it can become visible. Missing metadata is a hard
      // error, never permission to render the complete GLB.
      this._ensureLoadedInstancedVisibilityState(extras.hashCode);
      this._syncInstancedVisibilityHash(extras.hashCode);
    }
    if (extras && extras.hashCode && this.renderVisibilitySystem)
    {
      // setWorkingSet may have run before this GLB became resident. Register
      // the completed insertion so the incremental reconciler tracks the
      // actual scene state without waiting for another visibility result.
      this.renderVisibilitySystem.markResidentChanged(extras.hashCode);
      this.lastRenderRefreshStats = this.renderVisibilitySystem.lastStats;
    }
    if (loadedHash && (!extras || !extras.hashCode))
    {
      this.glbResourceScheduler.markFailed(
        loadedHash,
        new Error('Loaded GLB has no cacheable hash metadata.')
      );
    }
    return scene;
  }

  processPendingSceneInsertions()
  {
    if (this._getPendingSceneInsertionCount() === 0)
    {
      this.lastLoadMetrics.queueLength = 0;
      this._compactPendingSceneInsertionsIfNeeded(true);
      return;
    }

    var budget = this._getAdaptiveIntegrationBudget();
    var budgetMs = budget.budgetMs;
    var maxCount = budget.maxCount;
    var start = performance.now();
    var integratedCount = 0;

    while (this.pendingSceneInsertionCursor < this.pendingSceneInsertions.length && integratedCount < maxCount)
    {
      var nextItem = this.pendingSceneInsertions[this.pendingSceneInsertionCursor++];
      if (nextItem && nextItem.hash != null)
      {
        this.pendingSceneInsertionHashes.delete(nextItem.hash);
      }
      var integratedScene = this._integrateLoadedGltf(nextItem.gltf, nextItem.modelDesc || null);
      if (integratedScene)
      {
        integratedCount++;
        this.integratedSceneCount++;
      }

      if ((performance.now() - start) >= budgetMs)
      {
        break;
      }
    }

    this._compactPendingSceneInsertionsIfNeeded();
    this.lastLoadMetrics.queueLength = this._getPendingSceneInsertionCount();
    this.lastLoadMetrics.totalIntegrated = this.integratedSceneCount;
    this.lastLoadMetrics.lastIntegratedCount = integratedCount;
    this.lastLoadMetrics.lastBatchMs = performance.now() - start;
    this.lastLoadMetrics.lastIntegratedAt = performance.now();

    if (integratedCount > 0 && this.DebugMode)
    {
      console.log('[SLM2Loader] Integrated loaded scenes batch', {
        integratedCount: integratedCount,
        pendingSceneInsertions: this._getPendingSceneInsertionCount(),
        batchMs: this.lastLoadMetrics.lastBatchMs,
      });
    }
    if (integratedCount > 0)
    {
      this.requestRender('glb-mounted');
    }
    this.processGlbParseQueue();
    this.processLoadingList();
  }

  _makeGlobalGlbModelInfos(ids, weights, idMode, extra = {})
  {
    var out = [];
    var sourceIds = Array.from(ids || []);
    var sourceWeights = Array.from(weights || []);
    for (var i = 0; i < sourceIds.length; ++i)
    {
      var id = Number(sourceIds[i]);
      if (!Number.isFinite(id))
      {
        continue;
      }
      out.push(Object.assign({
        id: id,
        weight: Math.max(0.000001, Number(sourceWeights[i] || 1)),
        idMode: idMode || 'global-glb',
      }, extra));
    }
    return out;
  }

  _makeGlbScheduleEntries(modelInfos, tier)
  {
    var entries = [];
    for (var index = 0; index < (modelInfos || []).length; ++index)
    {
      var modelInfo = Object.assign({}, modelInfos[index], {
        prefetch: tier !== 'urgent',
      });
      var decoded = this.decodeModelInfo(modelInfo);
      if (!decoded || !decoded.hash)
      {
        continue;
      }
      entries.push({
        hash: decoded.hash,
        glbId: Number(modelInfo.id),
        score: Number(modelInfo.weight || 0),
        modelInfo: modelInfo,
      });
    }
    return entries;
  }

  _applyLightweightNeuralPlan(predictionPayload, idMode)
  {
    var appliedIdMode = idMode === 'global-glb-priority' ? 'global-glb' : (idMode || 'global-glb');
    var predictedDownloadInfos = this._makeGlobalGlbModelInfos(
      predictionPayload && predictionPayload.modelList,
      predictionPayload && predictionPayload.weightList,
      appliedIdMode
    );
    var predictedPrefetchInfos = this._makeGlobalGlbModelInfos(
      predictionPayload && predictionPayload.prefetchGlbIds,
      predictionPayload && predictionPayload.prefetchWeights,
      appliedIdMode,
      { prefetch: true, speculative: true }
    );
    var renderComponentIds = this._normalizeIdList(
      predictionPayload && predictionPayload.renderComponentModelList
        ? Array.from(predictionPayload.renderComponentModelList)
        : []
    );
    var renderGlbIds = this._normalizeIdList(
      predictionPayload && predictionPayload.renderModelList
        ? Array.from(predictionPayload.renderModelList)
        : []
    );
    var rawVisibleMode = this.getNeuralDownloadPlanMode() === 'raw-visible';
    var scheduleGroups = classifyGlbSchedule(
      predictedDownloadInfos,
      predictedPrefetchInfos,
      renderGlbIds,
      rawVisibleMode ? 'raw-visible' : 'viewcell-priority'
    );
    var renderInfos = classifyGlbSchedule(
      predictedDownloadInfos,
      [],
      renderGlbIds,
      'viewcell-priority'
    ).urgent;
    var immediateInfos = scheduleGroups.urgent;
    var warmInfos = scheduleGroups.warm;
    var speculativeInfos = scheduleGroups.speculative;
    var prefetchInfos = warmInfos.concat(speculativeInfos);

    var predictionEpoch = this._setCurrentNeuralWorkingSet(renderInfos, appliedIdMode);
    this.neuralDownloadInfoByGlbId = new Map();
    var allDownloadInfos = immediateInfos.concat(prefetchInfos);
    for (var downloadInfoIndex = 0; downloadInfoIndex < allDownloadInfos.length; ++downloadInfoIndex)
    {
      var downloadInfo = allDownloadInfos[downloadInfoIndex];
      downloadInfo.predictionEpoch = predictionEpoch;
      this.neuralDownloadInfoByGlbId.set(Number(downloadInfo.id), downloadInfo);
    }
    var resourceSchedule = this.glbResourceScheduler.setPlan({
      urgent: this._makeGlbScheduleEntries(immediateInfos, 'urgent'),
      warm: this._makeGlbScheduleEntries(warmInfos, 'warm'),
      speculative: this._makeGlbScheduleEntries(speculativeInfos, 'speculative'),
    }, predictionEpoch);
    this._pruneStalePendingInsertions();
    this.processLoadingList();

    if (this._isResidentRenderPolicy())
    {
      this.lastRenderRefreshStats = this._showAllResidentObjects();
    }

    this._applyInstancedVisibility(renderComponentIds);

    var immediateComponentIds = this._filterComponentIdsByGlbIds(
      predictionPayload && predictionPayload.componentModelList
        ? Array.from(predictionPayload.componentModelList)
        : [],
      immediateInfos.map(function(item){ return item.id; })
    );
    var prefetchComponentIds = this._filterComponentIdsByGlbIds(
      predictionPayload && predictionPayload.componentModelList
        ? Array.from(predictionPayload.componentModelList)
        : [],
      prefetchInfos.map(function(item){ return item.id; })
    );

    var schedulerStats = {
      scheduler: 'pvs-v4-worker',
      backend: predictionPayload ? predictionPayload.backend : null,
      fallbackReason: predictionPayload ? predictionPayload.fallbackReason : null,
      candidateCount: predictionPayload ? Number(predictionPayload.candidateCount || 0) : 0,
      rawInstanceCount: predictionPayload && predictionPayload.componentModelList ? predictionPayload.componentModelList.length : 0,
      rawGlbCount: predictionPayload && predictionPayload.modelList ? predictionPayload.modelList.length : 0,
      immediate: immediateInfos.length,
      prefetch: prefetchInfos.length,
      activeVisibleGlbCount: renderInfos.length,
      viewcellPrefetchGlbCount: warmInfos.length,
      resourceSchedule: resourceSchedule,
      downloadPlanMode: predictionPayload && predictionPayload.timings ? predictionPayload.timings.downloadPlanMode : this.getNeuralDownloadPlanMode(),
      renderInstanceCount: renderComponentIds.length,
      renderGlbCount: renderInfos.length,
      hasModelDownloadPriority: predictionPayload ? Boolean(predictionPayload.hasModelDownloadPriority) : false,
      prioritySource: predictionPayload && predictionPayload.timings ? predictionPayload.timings.prioritySource : null,
      candidateSelection: predictionPayload
        ? (predictionPayload.candidateSelection || (predictionPayload.timings && predictionPayload.timings.candidateSelection) || null)
        : null,
      workerTimings: predictionPayload ? predictionPayload.timings || null : null,
    };
    this.lastLightweightPVSSchedulerStats = schedulerStats;

    return {
      modelList: immediateInfos.map(function(item){ return item.id; }),
      weightList: immediateInfos.map(function(item){ return item.weight; }),
      componentModelList: immediateComponentIds,
      renderModelList: renderInfos.map(function(item){ return item.id; }),
      renderWeightList: renderInfos.map(function(item){ return item.weight; }),
      renderComponentModelList: renderComponentIds,
      renderVisibleCount: renderInfos.length,
      prefetchList: prefetchInfos,
      prefetchComponentModelList: prefetchComponentIds,
      deferredSkippedList: [],
      deferredSkippedComponentModelList: [],
      schedulerStats: schedulerStats,
      priorityItems: [],
    };
  }

  _applyLightweightNeuralRefilter(filterPayload, idMode)
  {
    var appliedIdMode = idMode === 'global-glb-priority' ? 'global-glb' : (idMode || 'global-glb');
    var componentAddedIds = this._normalizeIdList(
      filterPayload && filterPayload.renderComponentAddedIds
        ? Array.from(filterPayload.renderComponentAddedIds)
        : []
    );
    var componentRemovedIds = this._normalizeIdList(
      filterPayload && filterPayload.renderComponentRemovedIds
        ? Array.from(filterPayload.renderComponentRemovedIds)
        : []
    );
    var glbAddedIds = this._normalizeIdList(
      filterPayload && filterPayload.renderGlbAddedIds
        ? Array.from(filterPayload.renderGlbAddedIds)
        : []
    );
    var glbRemovedIds = this._normalizeIdList(
      filterPayload && filterPayload.renderGlbRemovedIds
        ? Array.from(filterPayload.renderGlbRemovedIds)
        : []
    );
    var glbDelta = this.renderGlbState.applyDelta(glbAddedIds, glbRemovedIds);
    glbAddedIds = Array.from(glbDelta.added);
    glbRemovedIds = Array.from(glbDelta.removed);
    var renderComponentCount = this._applyInstancedVisibilityDelta(
      componentAddedIds,
      componentRemovedIds
    );

    if (filterPayload && Number.isFinite(Number(filterPayload.renderInstanceCount)) &&
        renderComponentCount !== Number(filterPayload.renderInstanceCount))
    {
      this.forceNextNeuralPrediction = true;
      this.neuralPendingVisibilityUpdate = true;
      throw new Error('Cached neural refilter component delta is inconsistent with its result count.');
    }
    if (filterPayload && Number.isFinite(Number(filterPayload.renderGlbCount)) &&
        glbDelta.count !== Number(filterPayload.renderGlbCount))
    {
      this.forceNextNeuralPrediction = true;
      this.neuralPendingVisibilityUpdate = true;
      throw new Error('Cached neural refilter GLB delta is inconsistent with its result count.');
    }

    var addedRenderInfos = glbAddedIds.map((globalGlbId) =>
    {
      return Object.assign({}, this.neuralDownloadInfoByGlbId.get(globalGlbId) || {
        id: globalGlbId,
        weight: 1,
        idMode: appliedIdMode,
      }, { prefetch: false, deferredVisible: false });
    });
    var removedRenderInfos = this._makeGlobalGlbModelInfos(glbRemovedIds, null, appliedIdMode);
    var demotedRenderInfos = glbRemovedIds.map((globalGlbId) =>
    {
      return Object.assign({}, this.neuralDownloadInfoByGlbId.get(globalGlbId) || {
        id: globalGlbId,
        weight: 1,
        idMode: appliedIdMode,
      }, { prefetch: true, deferredVisible: true });
    });
    if (this._isResidentRenderPolicy())
    {
      this.lastRenderRefreshStats = this._showAllResidentObjects();
    }
    else
    {
      this.lastRenderRefreshStats = this.renderVisibilitySystem.applyDelta(
        addedRenderInfos,
        removedRenderInfos,
        appliedIdMode,
        this.currentNeuralPredictionEpoch
      );
    }
    var promotionStats = this._promoteCurrentRenderGlbs(glbAddedIds);
    var demotedGlbCount = this.glbResourceScheduler.reclassify(
      this._makeGlbScheduleEntries(demotedRenderInfos, 'warm'),
      'warm'
    );

    var previousScheduler = this.lastLightweightPVSSchedulerStats || {};
    this.lastLightweightPVSSchedulerStats = Object.assign({}, previousScheduler, {
      scheduler: 'pvs-v4-worker-cached-render-filter',
      backend: filterPayload ? filterPayload.backend : null,
      renderInstanceCount: renderComponentCount,
      renderGlbCount: glbDelta.count,
      filterTimings: filterPayload ? filterPayload.timings || null : null,
      reusedModelPrediction: true,
      transferredIdCount: filterPayload && filterPayload.timings
        ? Number(filterPayload.timings.transferredIdCount || 0)
        : 0,
      promotedGlbCount: promotionStats.promotedGlbCount,
      promotedParseCount: promotionStats.parseCount,
      promotedInsertionCount: promotionStats.insertionCount,
      demotedGlbCount: demotedGlbCount,
    });
    return {
      renderComponentCount: renderComponentCount,
      renderGlbCount: glbDelta.count,
      renderVisibleCount: glbDelta.count,
      schedulerStats: this.lastLightweightPVSSchedulerStats,
    };
  }

  refreshLoadingTask(modelList, weightList, idMode, predictionPayload = null)
  {
    if ((idMode || this.defaultVisibilityIdMode) === 'global-glb-priority')
    {
      ++this.neuralScheduleSerial;
      return this.refreshLoadingTask(modelList, weightList, 'global-glb', predictionPayload);
    }

    var appliedVisibility = this.fullLoadMode
      ? {
          modelList: modelList,
          weightList: Array.isArray(weightList) ? weightList : [],
        }
      : this._filterVisibilityByCameraFrustum(modelList, weightList, idMode);

    if (this.useNeuralPVS && this.neuralDebugNoCache && (idMode || this.defaultVisibilityIdMode) === 'global-glb')
    {
      var desiredHashes = new Set();
      for (var hIdx = 0; hIdx < appliedVisibility.modelList.length; ++hIdx)
      {
        var decoded = this.decodeModelInfo({
          id: appliedVisibility.modelList[hIdx],
          weight: appliedVisibility.weightList[hIdx],
          idMode: idMode || this.defaultVisibilityIdMode
        });
        if (decoded && decoded.hash)
        {
          desiredHashes.add(decoded.hash);
        }
      }
      this.modelCacheMgr.pruneToHashes(desiredHashes);
      var pendingCursor = Number(this.pendingSceneInsertionCursor || 0);
      this.pendingSceneInsertions = this.pendingSceneInsertions.slice(pendingCursor).filter(function(item)
      {
        return !item || item.hash == null || desiredHashes.has(item.hash);
      });
      this.pendingSceneInsertionCursor = 0;
      this.pendingSceneInsertionHashes = new Set(this.pendingSceneInsertions.map(function(item)
      {
        return item && item.hash != null ? item.hash : null;
      }).filter(function(hash){ return hash != null; }));
      this.lastLoadMetrics.queueLength = this._getPendingSceneInsertionCount();
      console.log('[SLM2Loader] Neural debug no-cache pruned stale cached objects.', {
        keepCount: desiredHashes.size,
        pendingSceneInsertions: this._getPendingSceneInsertionCount(),
      });
    }

    this.modelToLoadList = [];

    var neuralGlobalMode = this.useNeuralPVS && ((idMode || this.defaultVisibilityIdMode) === 'global-glb');
    var neuralWorkingInfos = [];
    for (var tIdx = appliedVisibility.modelList.length - 1; tIdx >= 0; tIdx--)
    {
      var modelInfo =
      {
        id: appliedVisibility.modelList[tIdx],
        weight: appliedVisibility.weightList[tIdx],
        idMode: idMode || this.defaultVisibilityIdMode
      };
      if (!neuralGlobalMode)
      {
        this.modelToLoadList.push(modelInfo);
      }
      neuralWorkingInfos.push(modelInfo);
    }

    if (neuralGlobalMode)
    {
      var neuralPredictionEpoch = this._setCurrentNeuralWorkingSet(neuralWorkingInfos, idMode || this.defaultVisibilityIdMode);
      this.glbResourceScheduler.setPlan({
        urgent: this._makeGlbScheduleEntries(neuralWorkingInfos, 'urgent'),
      }, neuralPredictionEpoch);
      this._pruneStalePendingInsertions();
      this.processLoadingList();
      var renderStatsV1 = this._isResidentRenderPolicy()
        ? this._showAllResidentObjects()
        : this.renderVisibilitySystem.lastStats;
      this.lastRenderRefreshStats = renderStatsV1;
      appliedVisibility.renderVisibleCount = renderStatsV1 ? renderStatsV1.visibleCount : null;
      var rawComponentIds = predictionPayload && predictionPayload.componentModelList
        ? Array.from(predictionPayload.componentModelList)
        : [];
      appliedVisibility.componentModelList = this._filterComponentIdsByGlbIds(rawComponentIds, appliedVisibility.modelList || []);
      appliedVisibility.renderModelList = Array.isArray(appliedVisibility.renderModelList)
        ? appliedVisibility.renderModelList
        : (appliedVisibility.modelList || []).slice();
      appliedVisibility.renderComponentModelList = this._filterComponentIdsByGlbIds(rawComponentIds, appliedVisibility.renderModelList || []);
      appliedVisibility.prefetchComponentModelList = [];
    }
    else
    {
      this.renderVisibilitySystem.clear();
      this.lastRenderRefreshStats = this.modelCacheMgr.refreshVisible(
        this.modelToLoadList,
        this._getRenderVisibilityOptions(idMode || this.defaultVisibilityIdMode)
      );
      if ((idMode || this.defaultVisibilityIdMode) === 'component')
      {
        appliedVisibility.componentModelList = (appliedVisibility.modelList || []).slice();
        appliedVisibility.renderComponentModelList = (appliedVisibility.modelList || []).slice();
        appliedVisibility.renderModelList = this._componentIdsToGlbIds(appliedVisibility.modelList || []);
      }
      else
      {
        appliedVisibility.componentModelList = [];
        appliedVisibility.renderComponentModelList = [];
        appliedVisibility.renderModelList = Array.isArray(appliedVisibility.renderModelList)
          ? appliedVisibility.renderModelList
          : (appliedVisibility.modelList || []).slice();
      }
      appliedVisibility.prefetchComponentModelList = [];
    }
    this.renderGlbState.replace(appliedVisibility.renderModelList || [], this._getGlbBitCount());
    this._applyInstancedVisibility(appliedVisibility.renderComponentModelList || []);
    return appliedVisibility;
  }
}
