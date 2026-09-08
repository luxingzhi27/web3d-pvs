import {
  Box3,
  Frustum,
  Vector3,
  Matrix4,
  FloatType,
  ImageBitmapLoader,
} from 'three';

import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';

import { CacheMgr } from './CacheMgr.js';
import { LightweightPVSDispatcher } from '../src/LightweightPVSDispatcher.js';
import { CameraPredictionGate } from '../src/CameraPredictionGate.js';
import { RenderVisibilitySystem } from '../src/RenderVisibilitySystem.js';
import { IdBitsetState } from '../src/IdBitsetState.js';
import { StaticSceneOptimizer } from '../src/StaticSceneOptimizer.js';
import { GlbResourceScheduler } from '../src/GlbResourceScheduler.js';
import { replaceDenseInstancedSlots } from '../src/DenseInstancedSlots.js';
import { startupLog } from '../src/startupTimeline.js';
import { SLM2GlbPipeline } from './SLM2GlbPipeline.js';
import { LOCAL_RUNTIME_ASSET_VERSION } from './SLM2RuntimeAssets.js';

export class SLM2Loader extends SLM2GlbPipeline
{
  constructor ()
  {
    super();
    startupLog('slm2:constructor:start');
    this.sceneConfig = null;
    this.glbIndex = null;
    this.globalGlbEntries = [];
    this.globalGlbHashToId = {};
    this.runtimeVisibilityMeta = null;
    this.componentVisibilityRecords = [];
    this.globalGlbVisibilityRecords = [];
    this.globalGlbToComponentIds = [];
    this.instancedVisibilityBindingsByHash = {};
    this.instancedVisibilityBindingByComponentId = {};
    this.expectedInstancedVisibilityHashes = new Set();
    this.loadedInstancedVisibilityStatesByHash = {};
    this.renderComponentState = new IdBitsetState();
    this.renderGlbState = new IdBitsetState();
    this.instancedWantedIndicesByHash = new Map();
    this.runtimeFrustumFilterEnabled = true;
    this.runtimeFrustumFilterSource = 'back-camera';
    this.runtimeFrustumFilterDebug = false;
    this._runtimeFrustum = new Frustum();
    this._runtimeFrustumMatrix = new Matrix4();
    this._runtimeTempBox = new Box3();
    this._runtimeTempCenter = new Vector3();
    this._runtimeTempSize = new Vector3();

    this.isSceneInitialized = false;

    this.resourcesBaseUrl = null;
    this.glbResourcesBaseUrl = null;
    this.neuralAssetGroups = [];
    this.neuralAssetGroupId = 0;
    this.neuralAssetBaseUrl = null;
    this.neuralRuntimeMetaUrl = null;
    this.glbIndexUrl = null;
    this.initialGlbLoadOrderUrl = null;
    this.neuralInitialLoadOrder = [];
    this.neuralInitialLoadPending = false;
    this.neuralPVSOptions = null;
    this.neuralGroupSwitchSerial = 0;
    this.schedulingStrategy = 'auto';

    this.modelToLoadList = [];
    this.glbResourceScheduler = new GlbResourceScheduler({
      onWorkAvailable: () => {
        if (!this.useNeuralPVS) return;
        this.processLoadingList();
        this.requestRender('glb-scheduler-work');
      },
    });
    this.pendingSceneInsertions = [];
    this.pendingSceneInsertionCursor = 0;
    this.pendingSceneInsertionHashes = new Set();
    this.inflightModelHashes = new Set();
    this.maxPendingSceneInsertions = 24;
    this.loadIntegrationBudgetMs = 6;
    this.integratedSceneCount = 0;
    this.activeDirectLoadCount = 0;
    this.directDownloadControllers = new Map();
    this.directLoadRetries = new Map();
    this.parseLoaderCursor = 0;
    this.baseDirectLoadConcurrency = 5;
    this.fullLoadDirectLoadConcurrency = 16;
    this.neuralDirectLoadConcurrency = 16;
    this.neuralPrefetchDirectLoadConcurrency = 4;
    this.httpAdaptiveConcurrencyEnabled = true;
    this.httpAdaptiveMinConcurrency = 3;
    this.httpAdaptiveMaxConcurrency = 24;
    this.httpAdaptivePrefetchMinConcurrency = 1;
    this.httpAdaptivePrefetchMaxConcurrency = 8;
    this.httpAdaptiveState = {
      current: 0,
      prefetchCurrent: 0,
      samples: 0,
      ewmaLoadMs: 0,
      ewmaNetworkMs: 0,
      ewmaMbps: 0,
      lastMbps: 0,
      lastBytes: 0,
      downlinkMbps: null,
      lastAdjustmentAt: 0,
      lastReason: 'init',
      lastResourceUrl: null,
      lastSuccess: true,
    };
    this.neuralMaxPendingSceneInsertions = 96;
    this.neuralLoadIntegrationBudgetMs = 4;
    this.lastFrameDt = 0;
    this.cameraUpdatePending = true;
    this.lastTextureTaskAt = 0;
    this.lastTextureTaskMs = 0;
    this.cpuPerfMode = 'balanced';
    this.neuralScheduleSerial = 0;

    this.modelCacheMgr = new CacheMgr({sceneMgr: this});
    this.staticSceneOptimizer = new StaticSceneOptimizer(this);

    this.clientWidth = 800;
    this.clientHeight = 600;

    this.DebugMode = false;

    this.MeshLodLevel = 0; // -1: raw, >= 0: LODi

    //TODO: Async image loading:
    // https://stackoverflow.com/questions/67775759/cant-use-three-js-texture-loader-in-javascript-worker
    this.textureLoader = new ImageBitmapLoader();//TextureLoader();
    this.hdrLoader = new HDRLoader().setDataType(FloatType);
    this.exrLoader = new EXRLoader().setDataType(FloatType);
    this.hdrjpgLoader = null;

    this.loadedTextures = {};
    this.isMaterialConfigReady = false;
    this.lastMaterialRebindCount = 0;

    
    // Custom full-load parameters
    this.fullLoadMode = false;
    this.hasFullLoaded = false;
    this.defaultVisibilityIdMode = 'component';
    this.neuralPVSIdMode = 'global-glb-priority';
    this.lastNeuralPrediction = null;
    this.lastLightweightPVSSchedulerStats = null;
    this.renderVisibilitySystem = new RenderVisibilitySystem(this);
    this.neuralPredictionGate = new CameraPredictionGate({
      mode: 'viewcell',
      positionDelta: 2,
      angleDeltaDeg: 6,
      yawDeltaDeg: 6,
      pitchDeltaDeg: 5,
      fovDeltaDeg: 2,
      aspectDelta: 0.08,
      minIntervalMs: 350,
    });
    this.neuralDebugNoCache = false;
    this.neuralRenderPolicy = 'culled';
    this.neuralDownloadPlanMode = 'viewcell-priority';
    this.lastRenderRefreshStats = null;
    this.actualRenderStatsCache = null;
    this.actualRenderStatsAt = 0;
    this.currentNeuralPredictionEpoch = 0;
    this.neuralDownloadInfoByGlbId = new Map();
    this.resourcePipelineSerial = 0;
    this.frozenPredictionInspectActive = false;
    this.frozenPredictionInspectQueued = false;
    this.frozenPredictionInspectTotal = 0;
    this.frozenPredictionInspectComponentIds = [];
    this.frozenPredictionInspectGlbIds = [];
    this.neuralPredictionLifecycleSerial = 0;
    this.forceNextNeuralPrediction = false;
    this.neuralPredictionInFlight = false;
    this.neuralPendingVisibilityUpdate = false;
    this.lastNeuralRefilter = null;
    this.lastVisibilityMetrics = null;
    this.visibilityMetricsSerial = 0;
    this.lastBenchmarkVisibilityIds = {
      serial: 0,
      mode: null,
      rawComponentIds: [],
      rawGlbIds: [],
      scheduledComponentIds: [],
      scheduledGlbIds: [],
      prefetchComponentIds: [],
      prefetchGlbIds: [],
      priorityItems: [],
      candidateSelection: null,
    };
    this.lastLoadMetrics = {
      queueLength: 0,
      totalIntegrated: 0,
      lastIntegratedCount: 0,
      lastBatchMs: 0,
      lastIntegratedAt: 0,
      staleCachedCount: 0,
      staleDroppedCount: 0,
      promotedGlbCount: 0,
      promotedParseCount: 0,
      promotedInsertionCount: 0,
    };
    this.startupMetrics = {
      sceneWebMs: 0,
      sceneWebParseMs: 0,
      instancedBindingMs: 0,
      glbIndexMs: 0,
      glbIndexParseMs: 0,
      glbIndexBuildMs: 0,
      runtimeVisibilityMetaMs: 0,
      runtimeVisibilityMetaParseMs: 0,
      runtimeVisibilityMetaBuildMs: 0,
      materialConfigMs: 0,
      firstPredictionMs: 0,
      firstPredictionRawMs: 0,
      firstPredictionApplyMs: 0,
    };
    this.neuralPVSInitScheduled = false;
    this.materialConfigLoadScheduled = false;
    this.materialConfigLoadStarted = false;
    this.runtimeVisibilityMetaLoadScheduled = false;
    this.runtimeVisibilityMetaLoadInFlight = false;
    this.startupSceneCullingLogCount = 0;
    // NeuralPVS parameters
    this.cullingMode = 'neural';
    this.neuralBackend = 'instance-pvs';
    this.useNeuralPVS = false;
    this.neuralPVS = null;
    this.neuralDebugLogs = false;
    this.neuralPVSReadyCullingScheduled = false;
    this.neuralPVSInitFallbackWarningShown = false;
    this.glbParseConcurrency = 2;
    this.maxPendingGlbParseBytes = 96 * 1024 * 1024;
    this.neuralStaleCacheEnabled = true;
    this.neuralStaleCachePriorityThreshold = 0.12;
    this.neuralStaleCacheProjectedAreaThreshold = 0.02;
    this.neuralStaleCachePrefetchAreaThreshold = 0.005;
    this.neuralStaleCacheMaxMemoryRatio = 0.85;
    this.pendingGlbParseQueue = [];
    this.pendingGlbParseCursor = 0;
    this.pendingGlbParseBytes = 0;
    this.activeGlbParseCount = 0;
    startupLog('slm2:constructor:end');
  }

  getNeuralRenderPolicy()
  {
    return this.neuralRenderPolicy || 'culled';
  }

  requestRender(reason = 'loader')
  {
    if (this.options && typeof this.options.requestRender === 'function')
    {
      this.options.requestRender(reason);
    }
  }

  notifyCameraChanged()
  {
    this.cameraUpdatePending = true;
    this.requestRender('camera-change');
  }

  getCullingMode()
  {
    return this.cullingMode || (this.useNeuralPVS ? 'neural' : 'frustum');
  }

  _resetCullingWorkingSet()
  {
    this.resourcePipelineSerial++;
    this._cancelAllDirectDownloads();
    this._clearPendingGlbParseQueue();
    this.inflightModelHashes = new Set();
    this.directLoadRetries.clear();
    this.modelToLoadList = [];
    this.glbResourceScheduler.reset();
    this.neuralDownloadInfoByGlbId = new Map();
    this.lastNeuralPrediction = null;
    this.lastRenderRefreshStats = null;
    this.lastLightweightPVSSchedulerStats = null;
    if (this.renderVisibilitySystem && typeof this.renderVisibilitySystem.clear === 'function')
    {
      this.renderVisibilitySystem.clear();
    }
    if (this.modelCacheMgr && typeof this.modelCacheMgr.refreshVisible === 'function')
    {
      this.modelCacheMgr.refreshVisible([], this._getRenderVisibilityOptions('global-glb'));
    }
    this.renderGlbState.replace([], this._getGlbBitCount());
    this._applyInstancedVisibility([]);
  }

  _createNeuralPVSDispatcher()
  {
    if (!this.neuralAssetBaseUrl)
    {
      return null;
    }

    var options = Object.assign({}, this.neuralPVSOptions || {}, {
      debugLogging: this.neuralDebugLogs,
      assetVersion: LOCAL_RUNTIME_ASSET_VERSION,
    });
    this.neuralPVS = new LightweightPVSDispatcher(this.neuralAssetBaseUrl, options);
    return this.neuralPVS;
  }

  setCullingMode(value)
  {
    var requested = String(value == null ? '' : value).toLowerCase();
    var normalized = requested === 'frustum' || requested === 'aabb'
      ? 'frustum'
      : 'neural';
    var wasNeural = Boolean(this.useNeuralPVS);

    this.cullingMode = normalized;
    this.fullLoadMode = false;
    this.hasFullLoaded = false;
    this.useNeuralPVS = normalized === 'neural';
    this.neuralPVSIdMode = this.useNeuralPVS ? 'global-glb-priority' : 'global-glb';
    this.neuralPredictionInFlight = false;
    this.neuralPendingVisibilityUpdate = false;
    this.lastNeuralRefilter = null;
    this.forceNextNeuralPrediction = this.useNeuralPVS;
    this.neuralPVSInitScheduled = false;
    this.neuralPVSReadyCullingScheduled = false;

    if (!this.useNeuralPVS)
    {
      if (this.neuralPVS && typeof this.neuralPVS.dispose === 'function')
      {
        this.neuralPVS.dispose();
      }
      this.neuralPVS = null;
    }
    else if (!this.neuralPVS)
    {
      this._createNeuralPVSDispatcher();
    }

    if (wasNeural !== this.useNeuralPVS || normalized === 'frustum')
    {
      this._resetCullingWorkingSet();
    }

    if (this.isSceneInitialized)
    {
      var scope = this;
      var rerun = function()
      {
        if (scope.cullingMode !== normalized)
        {
          return;
        }
        if (scope.useNeuralPVS)
        {
          scope.forceNextNeuralPrediction = true;
          scope._scheduleNeuralPVSInit();
        }
        else
        {
          scope.forceSceneCullingNow({ forceNeural: false });
        }
      };
      if (this.runtimeVisibilityMeta)
      {
        rerun();
      }
      else
      {
        this._scheduleRuntimeVisibilityMetaLoad(rerun);
      }
    }

    if (this.neuralDebugLogs)
    {
      console.log('[SLM2Loader] Culling mode set to', normalized);
    }
    return normalized;
  }

  getNeuralDownloadPlanMode()
  {
    return this.neuralDownloadPlanMode || 'viewcell-priority';
  }

  setNeuralDownloadPlanMode(value)
  {
    var normalized = value === 'raw-visible' ? 'raw-visible' : 'viewcell-priority';
    this.neuralDownloadPlanMode = normalized;
    if (this.neuralPVS && typeof this.neuralPVS.setDownloadPlanMode === 'function')
    {
      this.neuralPVS.setDownloadPlanMode(normalized);
    }
    this.forceNextNeuralPrediction = true;
    if (this.neuralDebugLogs) console.log('[SLM2Loader] Neural download plan mode set to', normalized);
    return normalized;
  }

  setNeuralRenderPolicy(value)
  {
    var normalized = value === 'resident' || value === true || value === 'true'
      ? 'resident'
      : 'culled';
    if (this.frozenPredictionInspectActive && normalized === 'resident')
    {
      console.warn('[SLM2Loader] Resident rendering is unavailable while a final visibility snapshot is frozen.');
      return this.neuralRenderPolicy;
    }
    this.neuralRenderPolicy = normalized;
    if (this.neuralDebugLogs) console.log('[SLM2Loader] Neural render policy set to', normalized);

    if (normalized === 'resident')
    {
      this._showAllResidentObjects();
    }
    else if (this.renderVisibilitySystem)
    {
      // Resident inspection may have exposed objects outside the exact GPU
      // working set. Reconcile once when returning to normal culling.
      this.renderVisibilitySystem.syncVisibleResidentsFromCache();
      this.renderVisibilitySystem.update();
      this._syncAllLoadedInstancedVisibilityStates();
    }

    return normalized;
  }

  setUseNeuralPVS(enabled)
  {
    var nextEnabled = Boolean(enabled);
    var wasEnabled = Boolean(this.useNeuralPVS);

    if (this.isSceneInitialized && nextEnabled !== wasEnabled &&
        (this.cullingMode === 'neural' || this.cullingMode === 'frustum'))
    {
      return this.setCullingMode(nextEnabled ? 'neural' : 'frustum');
    }

    this.useNeuralPVS = nextEnabled;

    if (nextEnabled && !wasEnabled && this.renderVisibilitySystem && typeof this.renderVisibilitySystem.syncVisibleResidentsFromCache === 'function')
    {
      this.renderVisibilitySystem.syncVisibleResidentsFromCache();
    }

    if (!nextEnabled && this.renderVisibilitySystem)
    {
      this.renderVisibilitySystem.clear();
    }

    if (nextEnabled !== wasEnabled)
    {
      this.forceSceneCullingNow();
    }

    return this.useNeuralPVS;
  }

  _isResidentRenderPolicy()
  {
    return this.useNeuralPVS && this.neuralRenderPolicy === 'resident';
  }

  _showAllResidentObjects()
  {
    if (!this.modelCacheMgr || !this.modelCacheMgr.objectsPool)
    {
      return {
        visibleCount: 0,
        attachedCount: 0,
      };
    }

    var pool = this.modelCacheMgr.objectsPool;
    var visibleCount = 0;
    var attachedCount = 0;
    for (var hash in pool)
    {
      var item = pool[hash];
      if (!item || !item.meshObject)
      {
        continue;
      }

      if (!item.isInScene && this.rootScene)
      {
        this.rootScene.add(item.meshObject);
        item.isInScene = true;
        attachedCount++;
      }

      item.isVisible = true;
      item.lastVisibleAt = this.modelCacheMgr.getNowMs();
      this.modelCacheMgr.applyRenderState(item);
      visibleCount++;
    }

    this._showAllResidentInstancedMeshes();

    return {
      visibleCount: visibleCount,
      attachedCount: attachedCount,
    };
  }

  _showAllResidentInstancedMeshes()
  {
    var bindings = this.instancedVisibilityBindingsByHash || {};
    var pool = this.modelCacheMgr && this.modelCacheMgr.objectsPool ? this.modelCacheMgr.objectsPool : {};
    for (var boundHash in bindings)
    {
      if (pool[boundHash] && pool[boundHash].meshObject)
      {
        this._ensureLoadedInstancedVisibilityState(boundHash);
      }
    }

    var states = this.loadedInstancedVisibilityStatesByHash || {};
    for (var hash in states)
    {
      var state = states[hash];
      if (!state || !state.meshStates)
      {
        continue;
      }

      if (state.disabled)
      {
        // Resident/debug mode must preserve the same safety boundary as the
        // normal visibility path.  An invalid mapping is never shown whole.
        this._hideInvalidInstancedVisibilityState(state);
        continue;
      }

      var allIndices = new Array(state.originalCount);
      for (var instanceIndex = 0; instanceIndex < state.originalCount; ++instanceIndex)
      {
        allIndices[instanceIndex] = instanceIndex;
      }
      replaceDenseInstancedSlots(state, allIndices);
    }
  }

  _applyFrozenPredictionInspectSnapshot()
  {
    var renderInfos = this._makeGlobalGlbModelInfos(
      this.frozenPredictionInspectGlbIds,
      null,
      'global-glb'
    );

    if (!this.frozenPredictionInspectQueued)
    {
      var predictionEpoch = this._setCurrentNeuralWorkingSet(renderInfos, 'global-glb');
      this.glbResourceScheduler.setPlan({
        urgent: this._makeGlbScheduleEntries(renderInfos, 'urgent'),
      }, predictionEpoch);
      this._pruneStalePendingInsertions();
      this.processLoadingList();

      this.frozenPredictionInspectQueued = true;
      this.frozenPredictionInspectTotal = renderInfos.length;
    }

    this._updateCurrentNeuralRenderSet(renderInfos, 'global-glb');
    this._applyInstancedVisibility(this.frozenPredictionInspectComponentIds);

    return {
      queued: this.frozenPredictionInspectQueued,
      total: this.frozenPredictionInspectTotal,
      componentCount: this.frozenPredictionInspectComponentIds.length,
    };
  }

  startFrozenPredictionInspectSession(componentIds, glbIds)
  {
    this.neuralPredictionLifecycleSerial++;
    this.frozenPredictionInspectActive = true;
    this.frozenPredictionInspectQueued = false;
    this.frozenPredictionInspectComponentIds = this._normalizeIdList(componentIds);
    this.frozenPredictionInspectGlbIds = this._normalizeIdList(glbIds);
    this.frozenPredictionInspectTotal = this.frozenPredictionInspectGlbIds.length;
    this.forceNextNeuralPrediction = false;
    this.neuralPendingVisibilityUpdate = false;
    this.modelCacheMgr.setSchedulingStrategy('manual');
    this._applyFrozenPredictionInspectSnapshot();
    return {
      active: this.frozenPredictionInspectActive,
      total: this.frozenPredictionInspectTotal,
      componentCount: this.frozenPredictionInspectComponentIds.length,
    };
  }

  stopFrozenPredictionInspectSession()
  {
    this.neuralPredictionLifecycleSerial++;
    this.frozenPredictionInspectActive = false;
    this.frozenPredictionInspectQueued = false;
    this.frozenPredictionInspectTotal = 0;
    this.frozenPredictionInspectComponentIds = [];
    this.frozenPredictionInspectGlbIds = [];
    if (!this.fullLoadMode)
    {
      this.modelCacheMgr.setSchedulingStrategy('auto');
    }

    if (this.useNeuralPVS)
    {
      this.forceNextNeuralPrediction = true;
      this.forceSceneCullingNow({ forceNeural: true });
    }
    else if (!this.fullLoadMode)
    {
      this.forceSceneCullingNow({ forceNeural: false });
    }

    return {
      active: this.frozenPredictionInspectActive,
      total: this.frozenPredictionInspectTotal,
    };
  }

}



