import {
  AmbientLight,
  AnimationMixer,
  AxesHelper,
  Box3,
  Cache,
  DirectionalLight,
  Frustum,
  GridHelper,
  HemisphereLight,
  LinearEncoding,
  LoaderUtils,
  LoadingManager,
  PMREMGenerator,
  PerspectiveCamera,
  REVISION,
  Scene,
  SkeletonHelper,
  Vector3,
  WebGLRenderer,
  sRGBEncoding,
  MeshStandardMaterial,
  DoubleSide,
  Color,
  FrontSide,
  ClampToEdgeWrapping,
  Object3D,
  Matrix4,
  FileLoader,
  TextureLoader,
  FloatType,
  MeshBasicMaterial,
  ConstantColorFactor,
  SRGBColorSpace,
  LinearSRGBColorSpace,
  LinearFilter,
  RepeatWrapping,
  ImageBitmapLoader,
  CanvasTexture
} from 'three';

import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { MODEL_INPUT_FOV_Y_DEG } from '../src/neuralPvsFovProtocol.js';

const MANAGER = new LoadingManager();
const THREE_PATH = `https://unpkg.com/three@0.${REVISION}.x`
const DRACO_LOADER = new DRACOLoader( MANAGER ).setDecoderPath( `${THREE_PATH}/examples/js/libs/draco/gltf/` );
const KTX2_LOADER = new KTX2Loader( MANAGER ).setTranscoderPath( `${THREE_PATH}/examples/js/libs/basis/` );
const LOCAL_RUNTIME_ASSET_VERSION = 'pvs-mainline-v4-hkust-20260825';
const INSTANCED_VISIBILITY_MATRIX = new Matrix4();

function withLocalRuntimeVersion(url)
{
  if (!url) return url;
  return url + (url.indexOf('?') >= 0 ? '&' : '?') + 'v=' + encodeURIComponent(LOCAL_RUNTIME_ASSET_VERSION);
}

function joinUrlPath(baseUrl, pathPart)
{
  var base = String(baseUrl || '').replace(/\/+$/, '');
  var suffix = String(pathPart || '').replace(/^\/+/, '');
  return base + '/' + suffix;
}

import { CacheMgr } from './CacheMgr.js';
import { Vector2 } from 'three';
import { getInstancePVSAssetBaseUrl } from '../src/neuralCullingBackendMode.js';
import { LightweightPVSDispatcher } from '../src/LightweightPVSDispatcher.js';
import { CameraPredictionGate } from '../src/CameraPredictionGate.js';
import { RenderVisibilitySystem } from '../src/RenderVisibilitySystem.js';
import { NeuralResourceWSPool } from '../src/NeuralResourceWSPool.js';
import { startupLog, stopStartupLog } from '../src/startupTimeline.js';

import { HDRJPGLoader } from '@monogrid/gainmap-js'

const sequentialPromiseMap = require('sequential-promise-map');

export class SLM2Loader 
{
  constructor () 
  {
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
    this.lastRenderableComponentIdsForInstancing = [];
    this.instancedRetainUntilByComponentId = new Map();
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
    this.neuralInitialLoadOrder = [];
    this.neuralInitialLoadPending = false;
    this.neuralPVSOptions = null;
    this.neuralGroupSwitchSerial = 0;
    this.rcServerAddress = null;
    this.remoteResourcesWS = null;
    this.schedulingStrategy = 'auto';
    this.connectedAssets = false;

    this.modelToLoadList = [];
    this.neuralWSHTTPFallbackQueue = [];
    this.modelIsLoading = false;
    this.batchModelDescs = {};
    this.pendingSceneInsertions = [];
    this.pendingSceneInsertionCursor = 0;
    this.pendingSceneInsertionHashes = new Set();
    this.inflightModelHashes = new Set();
    this.maxPendingSceneInsertions = 24;
    this.loadIntegrationBudgetMs = 6;
    this.integratedSceneCount = 0;
    this.activeDirectLoadCount = 0;
    this.directLoaderCursor = 0;
    this.baseDirectLoadConcurrency = 5;
    this.fullLoadDirectLoadConcurrency = 16;
    this.neuralDirectLoadConcurrency = 16;
    this.neuralPrefetchDirectLoadConcurrency = 4;
    this.neuralDisableLoadLimits = true;
    this.neuralBandwidthSaturationMaxConcurrency = 96;
    this.neuralBandwidthSaturationPrefetchMaxConcurrency = 64;
    this.httpAdaptiveConcurrencyEnabled = true;
    this.httpAdaptiveMinConcurrency = 3;
    this.httpAdaptiveMaxConcurrency = 96;
    this.httpAdaptivePrefetchMinConcurrency = 1;
    this.httpAdaptivePrefetchMaxConcurrency = 64;
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
    this.neuralLoadIntegrationBudgetMs = 12;
    this.neuralV2LoadIntegrationBudgetMs = 8;
    this.lastFrameDt = 0;
    this.lastTextureTaskAt = 0;
    this.lastTextureTaskMs = 0;
    this.cpuPerfMode = 'balanced';
    this.neuralScheduleSerial = 0;

    this.modelCacheMgr = new CacheMgr({sceneMgr: this});

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

    this.rvCameraHash = null;
    
    // Custom full-load parameters
    this.fullLoadMode = false;
    this.hasFullLoaded = false;
    this.defaultVisibilityIdMode = 'component';
    this.rcServerIdMode = 'component';
    this.neuralPVSIdMode = 'global-glb-priority';
    this.lastNeuralPrediction = null;
    this.lastLightweightPVSSchedulerStats = null;
    this.pendingPrefetchList = [];
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
    this.neuralRenderRetainMs = 0;
    this.neuralRenderMissTolerance = 0;
    this.lastRenderRefreshStats = null;
    this.actualRenderStatsCache = null;
    this.actualRenderStatsAt = 0;
    this.currentNeuralPredictionEpoch = 0;
    this.currentDownloadWantedHashes = new Set();
    this.frozenPredictionInspectActive = false;
    this.frozenPredictionInspectQueued = false;
    this.frozenPredictionInspectTotal = 0;
    this.frozenPredictionInspectComponentIds = [];
    this.frozenPredictionInspectGlbIds = [];
    this.neuralPredictionLifecycleSerial = 0;
    this.forceNextNeuralPrediction = false;
    this.neuralPredictionInFlight = false;
    this.neuralPendingPredictionAfterCurrent = false;
    this.lastVisibilityMetrics = null;
    this.visibilityMetricsSerial = 0;
    this.lastBenchmarkVisibilityIds = {
      serial: 0,
      mode: null,
      rawComponentIds: [],
      rawGlbIds: [],
      scheduledComponentIds: [],
      scheduledGlbIds: [],
      renderComponentIds: [],
      renderGlbIds: [],
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
    this.neuralDispatcherMode = 'lightweight-worker';
    this.useNeuralPVS = false;
    this.neuralPVS = null;
    this.neuralDebugLogs = false;
    this.neuralPVSReadyCullingScheduled = false;
    this.neuralPVSInitFallbackWarningShown = false;
    this.neuralUseResourcesWS = false;
    this.neuralResourceWSPool = null;
    this.neuralWSConnectionCount = 4;
    this.neuralWSBatchSize = 8;
    this.neuralWSMaxInFlightModels = 32;
    this.neuralWSBatchTimeoutMs = 20000;
    this.neuralWSParseConcurrency = 2;
    this.neuralWSMaxPendingParseBytes = 96 * 1024 * 1024;
    this.neuralWSFallbackHttp = true;
    this.neuralWSFallbackDelayMs = 1500;
    this.neuralWSMaxRetries = 2;
    this.neuralWSStartedAt = 0;
    this.neuralStaleCacheEnabled = true;
    this.neuralStaleCachePriorityThreshold = 0.12;
    this.neuralStaleCacheProjectedAreaThreshold = 0.02;
    this.neuralStaleCachePrefetchAreaThreshold = 0.005;
    this.neuralStaleCacheMaxMemoryRatio = 0.85;
    this.neuralLoadSkippedAsDeferred = false;
    this.neuralDeferredSkippedLoadLimit = 0;
    this.pendingWSParseQueue = [];
    this.pendingWSParseCursor = 0;
    this.pendingWSParseBytes = 0;
    this.activeWSParseCount = 0;
    this.neuralWSRetryCounts = {};
    this.neuralWSHTTPFallbackHashes = new Set();
    this.lastNeuralWSStats = {
      configuredConnections: 0,
      readyConnections: 0,
      inFlightBatches: 0,
      inFlightModels: 0,
      pendingParseCount: 0,
      pendingParseMB: 0,
      activeParseCount: 0,
      retryCount: 0,
      fallbackHttpCount: 0,
      staleCachedCount: 0,
      staleDropCount: 0,
      timeoutCount: 0,
    };
    startupLog('slm2:constructor:end');
  }

  getNeuralRenderRetainMs()
  {
    return Number(this.neuralRenderRetainMs || 0);
  }

  getNeuralRenderPolicy()
  {
    return this.neuralRenderPolicy || 'culled';
  }

  getCullingMode()
  {
    return this.cullingMode || (this.useNeuralPVS ? 'neural' : 'rvcServer');
  }

  _resetCullingWorkingSet()
  {
    this.modelToLoadList = [];
    this.pendingPrefetchList = [];
    this.currentDownloadWantedHashes = new Set();
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
    this.neuralPendingPredictionAfterCurrent = false;
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
      if (this.neuralResourceWSPool)
      {
        this.neuralResourceWSPool.close();
        this.neuralResourceWSPool = null;
      }
      this.connectedAssets = false;
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
      if (this.neuralResourceWSPool)
      {
        this.neuralResourceWSPool.close();
        this.neuralResourceWSPool = null;
        this.connectedAssets = false;
      }
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
        state.lastKey = '__invalid__';
        continue;
      }

      for (var meshStateIndex = 0; meshStateIndex < state.meshStates.length; ++meshStateIndex)
      {
        var meshState = state.meshStates[meshStateIndex];
        var mesh = meshState.mesh;
        for (var instanceIndex = 0; instanceIndex < meshState.originalCount; ++instanceIndex)
        {
          mesh.setMatrixAt(instanceIndex, meshState.originalMatrices[instanceIndex]);
        }
        mesh.count = meshState.originalCount;
        mesh.visible = meshState.originalCount > 0;
        if (mesh.instanceMatrix)
        {
          mesh.instanceMatrix.needsUpdate = true;
        }
        this._refreshInstancedMeshBounds(mesh);
      }
      state.lastKey = '__all__';
    }
  }

  _refreshInstancedMeshBounds(mesh)
  {
    if (!mesh)
    {
      return;
    }

    // InstancedMesh 的包围体不会在 count 或实例矩阵变化后自动更新。
    // 若继续沿用上一次可见实例集合的包围球，Three.js 会在 draw call 层面
    // 提前剔除仍位于当前视锥内的新实例集合。
    mesh.boundingBox = null;
    mesh.boundingSphere = null;
    if (mesh.count > 0 && typeof mesh.computeBoundingSphere === 'function')
    {
      mesh.computeBoundingSphere();
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
      this._setCurrentDownloadWantedHashes(renderInfos);
      this._pruneStalePendingInsertions();

      this.modelToLoadList = [];
      this.pendingPrefetchList = [];
      for (var i = renderInfos.length - 1; i >= 0; --i)
      {
        renderInfos[i].predictionEpoch = predictionEpoch;
        this.modelToLoadList.push(renderInfos[i]);
      }

      this.frozenPredictionInspectQueued = true;
      this.frozenPredictionInspectTotal = renderInfos.length;
    }

    var renderOptions = this._getRenderVisibilityOptions('global-glb');
    renderOptions.retainVisibleMs = 0;
    renderOptions.missTolerance = 0;
    this.lastRenderRefreshStats = this.modelCacheMgr.refreshVisible(renderInfos, renderOptions);
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
    this.neuralPendingPredictionAfterCurrent = false;
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

  setNeuralRenderRetainMs(value)
  {
    var numeric = Number(value);
    if (!Number.isFinite(numeric))
    {
      numeric = 0;
    }

    numeric = Math.max(0, Math.min(5000, Math.round(numeric)));
    this.neuralRenderRetainMs = numeric;
    if (this.neuralDebugLogs) console.log('[SLM2Loader] Neural render hide delay set to', numeric, 'ms');
    return numeric;
  }

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
    if (this.useNeuralPVS && this.neuralDisableLoadLimits)
    {
      return {
        budgetMs: Number.POSITIVE_INFINITY,
        maxCount: Number.POSITIVE_INFINITY,
      };
    }

    if (!this.useNeuralPVS)
    {
      return {
        budgetMs: Math.max(1, this.loadIntegrationBudgetMs),
        maxCount: Number.POSITIVE_INFINITY,
      };
    }

    var dt = Number(this.lastFrameDt || 0);
    var budgetMs = this.loadIntegrationBudgetMs;
    var maxCount = this.cpuPerfMode === 'mobile' ? 4 : 8;

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
      budgetMs = Math.min(budgetMs, dt > 20 ? budgetMs : 4);
      maxCount = Math.min(maxCount, 4);
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
      return mobile ? 32 : 64;
    }
    if (mobile)
    {
      if (mbps <= 5) return 8;
      if (mbps <= 15) return 16;
      if (mbps <= 40) return 24;
      if (mbps <= 80) return 40;
      if (mbps <= 160) return 56;
      return 64;
    }
    if (mbps <= 5) return 12;
    if (mbps <= 15) return 24;
    if (mbps <= 40) return 40;
    if (mbps <= 80) return 64;
    if (mbps <= 160) return 80;
    return 96;
  }

  _configureHttpAdaptiveConcurrency()
  {
    var mobile = this.cpuPerfMode === 'mobile';
    this.httpAdaptiveMinConcurrency = mobile ? 8 : 12;
    this.httpAdaptiveMaxConcurrency = mobile ? 64 : 96;
    this.httpAdaptivePrefetchMinConcurrency = mobile ? 4 : 8;
    this.httpAdaptivePrefetchMaxConcurrency = mobile ? 32 : 64;
    this.neuralBandwidthSaturationMaxConcurrency = mobile ? 64 : 96;
    this.neuralBandwidthSaturationPrefetchMaxConcurrency = mobile ? 32 : 64;
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
    var ignoreFramePressure = this.useNeuralPVS && this.neuralDisableLoadLimits;
    var pendingInsertions = this._getPendingSceneInsertionCount();
    var frameDt = Number(this.lastFrameDt || 0);
    var pendingRatio = pendingInsertions / Math.max(1, Number(this.maxPendingSceneInsertions || 1));

    if (!ignoreFramePressure && (frameDt > 45 || pendingRatio >= 0.75))
    {
      desired = Math.min(desired, Math.max(this.httpAdaptiveMinConcurrency, Number(state.current || desired) - 3));
      pressure = frameDt > 45 ? 'frame-pressure' : 'integration-backpressure';
    }
    else if (!ignoreFramePressure && (frameDt > 32 || pendingRatio >= 0.5))
    {
      desired = Math.min(desired, Math.max(this.httpAdaptiveMinConcurrency, Number(state.current || desired) - 1));
      pressure = frameDt > 32 ? 'frame-soft-pressure' : 'integration-soft-backpressure';
    }
    else if (!ignoreFramePressure && state.samples >= 3 && state.ewmaLoadMs > 4500)
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
    if (this.useNeuralPVS && this.neuralDisableLoadLimits)
    {
      return Math.max(1, Math.round(Number(baseConcurrency || 1)));
    }

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

  _getUnlimitedNeuralDirectLoadTarget(prefetchOnly = false)
  {
    var queued = Array.isArray(this.modelToLoadList) ? this.modelToLoadList.length : 0;
    var fallbackQueued = Array.isArray(this.neuralWSHTTPFallbackQueue) ? this.neuralWSHTTPFallbackQueue.length : 0;
    var active = Math.max(0, Number(this.activeDirectLoadCount || 0));
    var adaptive = prefetchOnly
      ? Number(this.httpAdaptiveState && this.httpAdaptiveState.prefetchCurrent || 0)
      : Number(this.httpAdaptiveState && this.httpAdaptiveState.current || 0);
    var base = prefetchOnly
      ? Math.max(1, Number(this.neuralPrefetchDirectLoadConcurrency || 1))
      : Math.max(1, Number(this.neuralDirectLoadConcurrency || 1));
    var cap = prefetchOnly
      ? Math.max(base, Number(this.neuralBandwidthSaturationPrefetchMaxConcurrency || 32))
      : Math.max(base, Number(this.neuralBandwidthSaturationMaxConcurrency || 64));
    var desired = Math.max(base, adaptive || 0);
    return Math.max(1, Math.min(cap, Math.ceil(active + queued + fallbackQueued + desired)));
  }

  _configureNeuralResourceWSOptions(params)
  {
    var mobile = this.cpuPerfMode === 'mobile';
    this.neuralWSConnectionCount = this._parseIntParam(params, 'neuralWSConnections', mobile ? 2 : 4, 1, 8);
    this.neuralWSBatchSize = this._parseIntParam(params, 'neuralWSBatchSize', 8, 1, 32);
    this.neuralWSMaxInFlightModels = this._parseIntParam(params, 'neuralWSMaxInFlightModels', mobile ? 12 : 32, 1, 128);
    this.neuralWSBatchTimeoutMs = this._parseIntParam(params, 'neuralWSBatchTimeoutMs', 20000, 2000, 120000);
    this.neuralWSParseConcurrency = this._parseIntParam(params, 'neuralWSParseConcurrency', mobile ? 1 : 2, 1, 8);
    var maxPendingParseMB = this._parseIntParam(params, 'neuralWSMaxPendingParseMB', mobile ? 48 : 96, 8, 512);
    this.neuralWSMaxPendingParseBytes = maxPendingParseMB * 1024 * 1024;
    this.neuralWSFallbackHttp = !(params['neuralWSFallbackHttp'] === 'false' || params['neuralWSFallbackHttp'] === '0');
  }

  _shouldUseNeuralResourceWSPool()
  {
    return this.useNeuralPVS &&
      this.neuralUseResourcesWS &&
      this.resourcesWS != null &&
      this.resourcesWS !== '';
  }

  _ensureNeuralResourceWSPool()
  {
    if (!this._shouldUseNeuralResourceWSPool())
    {
      return null;
    }

    var scope = this;
    if (!this.neuralResourceWSPool)
    {
      this.neuralResourceWSPool = new NeuralResourceWSPool({
        url: this.resourcesWS,
        connectionCount: this.neuralWSConnectionCount,
        batchTimeoutMs: this.neuralWSBatchTimeoutMs,
        debug: this.neuralDebugLogs,
        onBatchResult: function(batch, buffers, meta)
        {
          scope._handleNeuralWSBatchResult(batch, buffers, meta);
        },
        onBatchError: function(batch, reason, error)
        {
          scope._handleNeuralWSBatchFailure(batch, reason, error);
        },
        onStatsChange: function(stats)
        {
          scope.connectedAssets = Number(stats.readyConnections || 0) > 0;
          scope.lastNeuralWSStats = Object.assign({}, scope.lastNeuralWSStats || {}, stats);
        },
      });
    }
    else
    {
      this.neuralResourceWSPool.configure({
        url: this.resourcesWS,
        connectionCount: this.neuralWSConnectionCount,
        batchTimeoutMs: this.neuralWSBatchTimeoutMs,
        debug: this.neuralDebugLogs,
      });
    }

    if (!this.neuralResourceWSPool.started)
    {
      this.neuralWSStartedAt = performance.now();
      this.neuralResourceWSPool.start();
    }
    return this.neuralResourceWSPool;
  }

  _getNeuralWSStats()
  {
    var poolStats = this.neuralResourceWSPool ? this.neuralResourceWSPool.getStats() : {};
    var stats = Object.assign({}, this.lastNeuralWSStats || {}, poolStats);
    stats.pendingParseCount = this._getPendingWSParseCount();
    stats.pendingParseMB = this.pendingWSParseBytes / 1048576;
    stats.activeParseCount = this.activeWSParseCount;
    stats.httpFallbackQueueLength = this.neuralWSHTTPFallbackQueue.length;
    stats.config = {
      connections: this.neuralWSConnectionCount,
      batchSize: this.neuralWSBatchSize,
      maxInFlightModels: this.neuralWSMaxInFlightModels,
      parseConcurrency: this.neuralWSParseConcurrency,
      maxPendingParseMB: this.neuralWSMaxPendingParseBytes / 1048576,
      fallbackHttp: this.neuralWSFallbackHttp,
    };
    return stats;
  }

  _setCurrentDownloadWantedHashes(modelInfos)
  {
    this.currentDownloadWantedHashes = new Set();

    for (var i = 0; i < (modelInfos || []).length; ++i)
    {
      var decoded = this.decodeModelInfo(modelInfos[i]);
      if (decoded && decoded.hash)
      {
        this.currentDownloadWantedHashes.add(decoded.hash);
      }
    }
  }

  _setCurrentNeuralWorkingSet(modelInfos, idMode)
  {
    this.currentNeuralPredictionEpoch++;
    this.renderVisibilitySystem.setWorkingSet(modelInfos, idMode, this.currentNeuralPredictionEpoch);
    return this.currentNeuralPredictionEpoch;
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
    if (!this.useNeuralPVS || !gatingEnabled || this.currentDownloadWantedHashes == null)
    {
      return true;
    }

    return this.currentDownloadWantedHashes.has(hash);
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

    if (phase === 'ws-raw')
    {
      if (this.pendingWSParseBytes >= this.neuralWSMaxPendingParseBytes * 0.5)
      {
        return false;
      }
      if (Number(this.lastFrameDt || 0) > 33)
      {
        return false;
      }
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
      phase !== 'ws-raw';

    return wasImmediate ||
      wasDeferredVisible ||
      wasUsefulPrefetch ||
      projectedArea >= this.neuralStaleCacheProjectedAreaThreshold ||
      priority >= this.neuralStaleCachePriorityThreshold;
  }

  _recordStaleDownloadDrop(modelDesc, reason)
  {
    this.lastLoadMetrics.staleDroppedCount = Number(this.lastLoadMetrics.staleDroppedCount || 0) + 1;
    if (this.lastNeuralWSStats)
    {
      this.lastNeuralWSStats.staleDropCount = Number(this.lastNeuralWSStats.staleDropCount || 0) + 1;
    }
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
      this.lastNeuralWSStats.staleCachedCount = Number(this.lastNeuralWSStats.staleCachedCount || 0) + 1;
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
    this.lastNeuralWSStats.staleCachedCount = Number(this.lastNeuralWSStats.staleCachedCount || 0) + 1;
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

    this._pruneStaleWSParseQueue();
    this._pruneStaleNeuralWSHttpFallbackQueue();

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
        this._recordStaleDownloadDrop(item.modelDesc || { hash: item.hash }, 'stale-pending-integration');
        this._disposeUnintegratedScene(item.gltf && (item.gltf.scene || (item.gltf.scenes && item.gltf.scenes[0])));
      }
    }

    this.pendingSceneInsertions = kept;
    this.pendingSceneInsertionCursor = 0;
    this.lastLoadMetrics.queueLength = this._getPendingSceneInsertionCount();
  }

  _pruneStaleWSParseQueue()
  {
    if (!Array.isArray(this.pendingWSParseQueue) || this._getPendingWSParseCount() === 0)
    {
      return;
    }

    var kept = [];
    var cursor = Number(this.pendingWSParseCursor || 0);
    for (var i = cursor; i < this.pendingWSParseQueue.length; ++i)
    {
      var item = this.pendingWSParseQueue[i];
      if (!item || !item.modelDesc || item.modelDesc.hash == null || this._isHashWantedForDownload(item.modelDesc.hash))
      {
        kept.push(item);
        continue;
      }

      if (this._shouldRetainStaleDownload(item.modelDesc, 'ws-raw'))
      {
        kept.push(item);
        continue;
      }

      this.pendingWSParseBytes = Math.max(0, this.pendingWSParseBytes - Number(item.byteLength || 0));
      this._clearInflightHash(item.modelDesc);
      this._recordStaleDownloadDrop(item.modelDesc, 'stale-ws-parse-queue');
    }

    this.pendingWSParseQueue = kept;
    this.pendingWSParseCursor = 0;
  }

  _pruneStaleNeuralWSHttpFallbackQueue()
  {
    if (!Array.isArray(this.neuralWSHTTPFallbackQueue) || this.neuralWSHTTPFallbackQueue.length === 0)
    {
      return;
    }

    var kept = [];
    this.neuralWSHTTPFallbackHashes = new Set();
    for (var i = 0; i < this.neuralWSHTTPFallbackQueue.length; ++i)
    {
      var item = this.neuralWSHTTPFallbackQueue[i];
      var decoded = item ? this.decodeModelInfo(item) : null;
      if (!decoded || !decoded.hash || this._isHashWantedForDownload(decoded.hash))
      {
        kept.push(item);
        if (decoded && decoded.hash)
        {
          this.neuralWSHTTPFallbackHashes.add(decoded.hash);
        }
      }
    }
    this.neuralWSHTTPFallbackQueue = kept;
  }

  _getPendingWSParseCount()
  {
    return Math.max(0, this.pendingWSParseQueue.length - Number(this.pendingWSParseCursor || 0));
  }

  _compactPendingWSParseQueueIfNeeded(force = false)
  {
    var cursor = Number(this.pendingWSParseCursor || 0);
    if (cursor <= 0)
    {
      return;
    }

    if (force || cursor >= 64 || cursor >= this.pendingWSParseQueue.length * 0.5)
    {
      this.pendingWSParseQueue = this.pendingWSParseQueue.slice(cursor);
      this.pendingWSParseCursor = 0;
    }
  }

  _buildResourceWSRequestItem(modelDesc)
  {
    if (!modelDesc || modelDesc.group == null)
    {
      return null;
    }

    var baseId = modelDesc.baseId != null
      ? Number(modelDesc.baseId)
      : (modelDesc.hash ? Number.parseInt(String(modelDesc.hash).split('-')[1]) : NaN);

    if (!Number.isFinite(baseId))
    {
      return null;
    }

    return {
      g: modelDesc.group,
      l: this.MeshLodLevel,
      s: baseId,
    };
  }

  _clearInflightHash(modelDesc)
  {
    if (modelDesc && modelDesc.hash)
    {
      this.inflightModelHashes.delete(modelDesc.hash);
    }
  }

  _queueNeuralWSHttpFallback(modelDesc, reason)
  {
    if (!this.neuralWSFallbackHttp || !modelDesc || !modelDesc.sourceModelInfo)
    {
      return;
    }

    if (modelDesc.hash && !this._isHashWantedForDownload(modelDesc.hash))
    {
      return;
    }

    if (modelDesc.hash && this.neuralWSHTTPFallbackHashes.has(modelDesc.hash))
    {
      return;
    }

    var sourceInfo = Object.assign({}, modelDesc.sourceModelInfo, {
      forceHttp: true,
    });
    this.neuralWSHTTPFallbackQueue.push(sourceInfo);
    if (modelDesc.hash)
    {
      this.neuralWSHTTPFallbackHashes.add(modelDesc.hash);
    }
    this.lastNeuralWSStats.fallbackHttpCount = Number(this.lastNeuralWSStats.fallbackHttpCount || 0) + 1;

    if (this.neuralDebugLogs)
    {
      console.warn('[SLM2Loader] resourcesWS falling back to HTTP', {
        hash: modelDesc.hash,
        reason: reason,
      });
    }
  }

  _retryOrFallbackNeuralWSModel(modelDesc, reason)
  {
    this._clearInflightHash(modelDesc);

    if (!modelDesc || (modelDesc.hash && !this._isHashWantedForDownload(modelDesc.hash)))
    {
      this.lastNeuralWSStats.staleDropCount = Number(this.lastNeuralWSStats.staleDropCount || 0) + 1;
      return;
    }

    var hash = modelDesc.hash || '';
    var retryCount = hash ? Number(this.neuralWSRetryCounts[hash] || 0) : this.neuralWSMaxRetries;
    if (hash && retryCount < this.neuralWSMaxRetries && modelDesc.sourceModelInfo)
    {
      this.neuralWSRetryCounts[hash] = retryCount + 1;
      this.modelToLoadList.push(modelDesc.sourceModelInfo);
      this.lastNeuralWSStats.retryCount = Number(this.lastNeuralWSStats.retryCount || 0) + 1;
      return;
    }

    this._queueNeuralWSHttpFallback(modelDesc, reason);
  }

  _handleNeuralWSBatchFailure(batch, reason, error)
  {
    if (reason === 'timeout')
    {
      this.lastNeuralWSStats.timeoutCount = Number(this.lastNeuralWSStats.timeoutCount || 0) + 1;
    }

    var descs = batch && Array.isArray(batch.reqDescs) ? batch.reqDescs : [];
    for (var i = 0; i < descs.length; ++i)
    {
      this._retryOrFallbackNeuralWSModel(descs[i], reason);
    }

    if (this.neuralDebugLogs)
    {
      console.warn('[SLM2Loader] resourcesWS batch failed', {
        batchId: batch ? batch.id : null,
        socketId: batch ? batch.socketId : null,
        reason: reason,
        message: error && error.message ? error.message : error,
        count: descs.length,
      });
    }
  }

  _handleNeuralWSBatchResult(batch, buffers, meta)
  {
    var descs = batch && Array.isArray(batch.reqDescs) ? batch.reqDescs : [];
    var returnedCount = Array.isArray(buffers) ? buffers.length : 0;
    var usableCount = Math.min(descs.length, returnedCount);

    for (var i = 0; i < usableCount; ++i)
    {
      var modelDesc = descs[i];
      var buffer = buffers[i];
      var byteLength = buffer && buffer.byteLength ? buffer.byteLength : 0;
      if (modelDesc && modelDesc.hash && !this._isHashWantedForDownload(modelDesc.hash))
      {
        if (!this._shouldRetainStaleDownload(modelDesc, 'ws-raw'))
        {
          this._clearInflightHash(modelDesc);
          this._recordStaleDownloadDrop(modelDesc, 'stale-ws-response');
          continue;
        }
      }

      this.pendingWSParseQueue.push({
        buffer: buffer,
        modelDesc: modelDesc,
        byteLength: byteLength,
        batchId: batch.id,
        socketId: batch.socketId,
        receivedAt: performance.now(),
      });
      this.pendingWSParseBytes += byteLength;
    }

    for (var missingIdx = usableCount; missingIdx < descs.length; ++missingIdx)
    {
      this._retryOrFallbackNeuralWSModel(descs[missingIdx], 'missing-buffer');
    }

    if (returnedCount > descs.length && this.neuralDebugLogs)
    {
      console.warn('[SLM2Loader] resourcesWS returned extra buffers', {
        batchId: batch.id,
        requested: descs.length,
        returned: returnedCount,
      });
    }

    if (this.neuralDebugLogs)
    {
      console.log('[SLM2Loader] resourcesWS batch accepted', {
        batchId: batch.id,
        socketId: batch.socketId,
        requested: descs.length,
        returned: returnedCount,
        queuedParse: usableCount,
        latencyMs: meta ? meta.latencyMs : null,
        bytes: meta ? meta.bytesReceived : null,
      });
    }

    this.processNeuralWSParseQueue();
  }

  processNeuralWSParseQueue()
  {
    if (!this.pendingWSParseQueue || this._getPendingWSParseCount() === 0)
    {
      this._compactPendingWSParseQueueIfNeeded(true);
      return;
    }

    if (this.gltfLoaders == undefined || this.gltfLoaders.length === 0)
    {
      return;
    }

    while (this.activeWSParseCount < this.neuralWSParseConcurrency &&
      this.pendingWSParseCursor < this.pendingWSParseQueue.length &&
      this._getPendingSceneInsertionCount() < this.maxPendingSceneInsertions)
    {
      var item = this.pendingWSParseQueue[this.pendingWSParseCursor++];
      if (!item)
      {
        continue;
      }

      if (item.modelDesc && item.modelDesc.hash && !this._isHashWantedForDownload(item.modelDesc.hash))
      {
        if (!this._shouldRetainStaleDownload(item.modelDesc, 'ws-raw'))
        {
          this.pendingWSParseBytes = Math.max(0, this.pendingWSParseBytes - Number(item.byteLength || 0));
          this._clearInflightHash(item.modelDesc);
          this._recordStaleDownloadDrop(item.modelDesc, 'stale-ws-before-parse');
          continue;
        }
      }

      this._parseNeuralWSBuffer(item);
    }

    this._compactPendingWSParseQueueIfNeeded();
  }

  _parseNeuralWSBuffer(item)
  {
    var scope = this;
    var loader = this.gltfLoaders[this.directLoaderCursor % this.gltfLoaders.length];
    this.directLoaderCursor++;
    this.activeWSParseCount++;
    var parseStartedAt = performance.now();

    loader.parse(item.buffer, '', function(gltf)
    {
      scope.activeWSParseCount = Math.max(0, scope.activeWSParseCount - 1);
      scope.pendingWSParseBytes = Math.max(0, scope.pendingWSParseBytes - Number(item.byteLength || 0));

      if (item.modelDesc && item.modelDesc.hash && !scope._isHashWantedForDownload(item.modelDesc.hash))
      {
        if (!scope._cacheStaleLoadedGltf(gltf, item.modelDesc, 'stale-ws-after-parse'))
        {
          scope._clearInflightHash(item.modelDesc);
          scope._recordStaleDownloadDrop(item.modelDesc, 'stale-ws-after-parse');
          scope._disposeUnintegratedScene(gltf && (gltf.scene || (gltf.scenes && gltf.scenes[0])));
        }
        scope.processNeuralWSParseQueue();
        return;
      }

      scope.processLaodedGltf(gltf, item.modelDesc);
      scope.processNeuralWSParseQueue();
    }, function(err)
    {
      scope.activeWSParseCount = Math.max(0, scope.activeWSParseCount - 1);
      scope.pendingWSParseBytes = Math.max(0, scope.pendingWSParseBytes - Number(item.byteLength || 0));
      console.error('[LoadError][resourcesWS-parse]', item.modelDesc ? item.modelDesc.hash : null, err);
      scope._retryOrFallbackNeuralWSModel(item.modelDesc, 'parse-error');
      scope.processNeuralWSParseQueue();
    });
  }

  loadMaterialConfig(callback)
  {
    startupLog('slm2:materialConfig:start');
    var scope = this;
    var materialConfigStartedAt = performance.now();
    if (this.DebugMode) console.log(this.sceneConfig);

    var loadMtlProxy = function (taskData, index/*, array*/) 
    {
      return new Promise(function(resolve)
      {
        var taskStartedAt = performance.now();
        var proxyModelUrl = scope.resourcesBaseUrl + '/' + taskData.proxyGlb;
        startupLog('slm2:materialConfig:proxy-request', {
          index: index,
          proxyGlb: taskData.proxyGlb,
        });

        var gltfLoader = new GLTFLoader().setCrossOrigin('anonymous').setDRACOLoader( DRACOLoader ).setMeshoptDecoder( MeshoptDecoder );
        gltfLoader.load(proxyModelUrl, (gltf) => 
        {
          var gltfLoadedAt = performance.now();
          startupLog('slm2:materialConfig:proxy-loaded', {
            index: index,
            ms: gltfLoadedAt - taskStartedAt,
          });
          // Load image lod config
          var fileLoader = new FileLoader();
          startupLog('slm2:materialConfig:image-config-request', {
            index: index,
            imageConfig: taskData.imageConfig,
          });
          var imageConfigUrl = joinUrlPath(
            scope.glbResourcesBaseUrl || scope.resourcesBaseUrl,
            taskData.imageConfig
          );
          fileLoader.load(imageConfigUrl, function(data)
          {
            var imageConfig = null;
            try
            {
              imageConfig = JSON.parse(data);
            }
            catch (error)
            {
              console.warn('[SLM2Loader] Invalid material image config response:', imageConfigUrl, error);
              resolve(null);
              return;
            }
            startupLog('slm2:materialConfig:image-config-loaded', {
              index: index,
              ms: performance.now() - gltfLoadedAt,
            });

            resolve(
              {
                gltf: gltf,
                imageConfig: imageConfig,
                taskIndex: index,
                proxyGlb: taskData.proxyGlb,
                taskStartedAt: taskStartedAt,
              });
          }, null, function(err)
          {
            console.warn('[SLM2Loader] Failed to load material image config:', imageConfigUrl, err);
            resolve(null);
          });
        }, null, (err) =>{
          console.warn('[SLM2Loader] Failed to load material proxy:', proxyModelUrl, err);
          startupLog('slm2:materialConfig:proxy-failed', {
            index: index,
            proxyGlb: taskData.proxyGlb,
          });
          resolve(null);
        });
      });
    };

    var collectMaterialTexture = function(mtl)
    {
      var texs = [];

      for (var key in mtl)
      {
        var item = mtl[key];
        if (item != null && item.isTexture)
        {
          var texChannel = key;

          if (key == 'aoMap')
          {
            texChannel = 'lightMap'; 
          }

          texs.push(
          {
            name: item.name,
            channel: texChannel,
            level: 0,
            data: 
            {
              "0": item
            }
          });
        }
      };

      if (scope.DebugMode) console.log(mtl, texs);

      return texs;
    }

    var mtlConfigTask = [];

    for (var i = 0; i < this.sceneConfig.materials.proxy.length; ++i)
    {
      mtlConfigTask.push(
        {
          proxyGlb: this.sceneConfig.materials.proxy[i],
          imageConfig: 'task-' + i + '/' + 'images/image_lod.json',
        });
    }

    var loadMaterialProxyList = function(tasks)
    {
      return new Promise(function(resolve)
      {
        var results = new Array(tasks.length);
        var runNext = function(index)
        {
          if (index >= tasks.length)
          {
            startupLog('slm2:materialConfig:all-proxies-loaded', {
              count: results.length,
            });
            resolve(results);
            return;
          }

          var loadCurrent = function()
          {
            loadMtlProxy(tasks[index], index).then(function(result)
            {
              results[index] = result;
              setTimeout(function()
              {
                runNext(index + 1);
              }, 0);
            });
          };

          if (typeof requestIdleCallback === 'function')
          {
            requestIdleCallback(loadCurrent, { timeout: 1000 });
          }
          else
          {
            setTimeout(loadCurrent, 0);
          }
        };

        runNext(0);
      });
    };

    loadMaterialProxyList(mtlConfigTask).then(results => 
    {
      this.sceneConfig.materials.data = new Array(this.sceneConfig.materials.proxy.length);
      this.sceneConfig.materials.config = new Array(this.sceneConfig.materials.proxy.length);

      for (var i = 0; i < results.length; ++i)
      {
        if (!results[i])
        {
          continue;
        }
        var mtls = {};
        results[i].gltf.scene.traverse((node) => 
        {
          if (!node.isMesh) return;
          if (node.material) 
          {
            node.material.flatShading = false;

            var texs = collectMaterialTexture(node.material);

            var preloadMaterial = node.material;
            preloadMaterial.side = DoubleSide;

            mtls[preloadMaterial.name] = 
            {
              material: preloadMaterial,
              textures: texs
            }
          }
        });

        this.sceneConfig.materials.config[results[i].taskIndex] = results[i].imageConfig;
        this.sceneConfig.materials.data[results[i].taskIndex] = mtls;
      }

      if (this.options.materialLoadedCallback)
      {
        this.options.materialLoadedCallback();
      }
      this.isMaterialConfigReady = true;
      this.lastMaterialRebindCount = this._rebindResidentMaterialsToCache();
      startupLog('slm2:materialConfig:ready', {
        totalMs: performance.now() - materialConfigStartedAt,
        reboundMeshes: this.lastMaterialRebindCount,
      });
      //console.log(this.sceneConfig.materials.data);

      if (callback)
      {
        scope.startupMetrics.materialConfigMs = performance.now() - materialConfigStartedAt;
        callback();
      }
    }).catch(()=>{
      this.isMaterialConfigReady = false;
      this.lastMaterialRebindCount = 0;
      startupLog('slm2:materialConfig:failed');
      if (callback)
      {
        scope.startupMetrics.materialConfigMs = performance.now() - materialConfigStartedAt;
        callback();
      }
    });
  }

  getMaterials()
  {
    var mtls = [];

    if (this.sceneConfig && this.sceneConfig.materials)
    {
      for (var key in this.sceneConfig.materials.data)
      {
        var item = this.sceneConfig.materials.data[key];

        for (var _key in item)
        {
          var _item = item[_key];

          mtls.push(_item.material);
        };
      };
    }

    return mtls;
  }

  fetchCachedMaterial(srcMtl, gltfDesc)
  {
    var matchedGroupId = parseInt(gltfDesc.hashCode.split('-')[0]);
    if (!this.sceneConfig ||
      !this.sceneConfig.materials ||
      !Array.isArray(this.sceneConfig.materials.data) ||
      this.sceneConfig.materials.data[matchedGroupId] == undefined)
    {
      return srcMtl;
    }
    var cachedMtlObj = this.sceneConfig.materials.data[matchedGroupId][srcMtl.name];

    // 添加新的贴图加载任务
    var newImageTask = 
    {
      groupId: matchedGroupId,
      mtlName: srcMtl.name,
      gltfHash: gltfDesc.hashCode,
      weightNormalized: this.modelCacheMgr.objectsPool[gltfDesc.hashCode].weight * this.screenPixelReciprocal
    };

    if (this.materialImageLoadingTasks == undefined)
    {
      this.materialImageLoadingTasks = {};
      this.isMaterialImageLoading = false;
    }

    if (this.materialImageLoadingTasks[srcMtl.name] == undefined || this.materialImageLoadingTasks[srcMtl.name].weight < newImageTask.weight)
    {
      this.materialImageLoadingTasks[srcMtl.name] = newImageTask;
    }

    if (cachedMtlObj && cachedMtlObj.material)
    {
      return cachedMtlObj.material;
    }
    else
    {
      console.log('failed to find material: ' + srcMtl.name);

      return srcMtl;
    }
  }

  _rebindResidentMaterialsToCache()
  {
    if (!this.modelCacheMgr || !this.modelCacheMgr.objectsPool)
    {
      return 0;
    }

    if (!this.sceneConfig ||
      !this.sceneConfig.materials ||
      !Array.isArray(this.sceneConfig.materials.data) ||
      this.sceneConfig.materials.data.length === 0)
    {
      return 0;
    }

    var reboundMeshes = 0;
    var pool = this.modelCacheMgr.objectsPool;
    for (var hash in pool)
    {
      var item = pool[hash];
      if (!item || !item.meshObject)
      {
        continue;
      }

      var extras = item.meshObject.asset && item.meshObject.asset.extras
        ? item.meshObject.asset.extras
        : { hashCode: hash };
      item.meshObject.traverse((node) =>
      {
        if (!node.isMesh || !node.material)
        {
          return;
        }
        node.material = this.fetchCachedMaterial(node.material, extras);
        reboundMeshes++;
      });
    }

    return reboundMeshes;
  }

  refreshMaterialTexture(groupId, materialName, textureId, textureData, isFromCache, imageMeta)
  {
    var material = this.sceneConfig.materials.data[groupId][materialName].material;
    var texture = this.sceneConfig.materials.data[groupId][materialName].textures[textureId];

    if (material != undefined && texture != undefined)
    {
      texture.level++;

      // 若贴图为空，可能是加载失败，直接跳过当前的细节度
      if (textureData != null)
      {
        if (texture.channel == 'aoMap')
        {
          if (isFromCache == false)
          {
            textureData.wrapS = material[texture.channel].wrapS;
            textureData.wrapT = material[texture.channel].wrapT;
            textureData.offset = material[texture.channel].offset;
            textureData.repeat  = material[texture.channel].repeat;
            textureData.rotation = material[texture.channel].rotation;
            textureData.center = material[texture.channel].center;
            textureData.colorSpace = material[texture.channel].colorSpace;
            textureData.flipY = material[texture.channel].flipY;
            textureData.generateMipmaps = true;
          }
          
          material[texture.channel] = textureData;
          material.needsUpdate = true;
        }
        else if (texture.channel == 'lightMap')
        {
          if (isFromCache == false)
          {
            textureData.channel = 1;

            if (imageMeta.type == 'jpgr')
            {

            }
            else
            {
              textureData.flipY = (imageMeta.raw == 'exr' ? true: false);
              textureData.type = FloatType;
              textureData.minFilter = LinearFilter;
              textureData.magFilter = LinearFilter;
              textureData.wrapS = RepeatWrapping;
              textureData.wrapT = RepeatWrapping;
            }
            
            textureData.needsUpdate = true;
          }
          
          material[texture.channel] = textureData;
          material.needsUpdate = true;
        }
        else
        {
          if (isFromCache == false)
          {
            textureData.wrapS = material[texture.channel].wrapS;
            textureData.wrapT = material[texture.channel].wrapT;
            textureData.offset = material[texture.channel].offset;
            textureData.repeat  = material[texture.channel].repeat;
            textureData.rotation = material[texture.channel].rotation;
            textureData.center = material[texture.channel].center;
            textureData.colorSpace = material[texture.channel].colorSpace;
            textureData.flipY = material[texture.channel].flipY;
            textureData.generateMipmaps = true;
          }
    
          material[texture.channel] = textureData;
          material[texture.channel].needsUpdate = true;
        }

        texture.data[texture.level] = textureData;
      }
    }
  }

  processImageTask(dt = 0)
  {
    var nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    var minIntervalMs = this.cpuPerfMode === 'mobile' ? 500 : 250;
    if (this.lastTextureTaskAt > 0 && nowMs - this.lastTextureTaskAt < minIntervalMs)
    {
      return;
    }

    if (this._getPendingSceneInsertionCount() > 0 || Number(dt || 0) > 25)
    {
      return;
    }

    this.lastTextureTaskAt = nowMs;
    var textureTaskStart = nowMs;
    if (this.materialImageLoadingTasks == undefined)
    {
      this.lastTextureTaskMs = 0;
      return;
    }
    var scope = this;
    if (this.isMaterialImageLoading == false)
    {
      var totalTasks = [];

      for (var key in this.materialImageLoadingTasks)
      {
        var item = this.materialImageLoadingTasks[key];
        var materialGroup = scope.sceneConfig.materials.data[item.groupId];
        var imageConfigGroup = scope.sceneConfig.materials.config[item.groupId];
        if (!materialGroup || !imageConfigGroup)
        {
          continue;
        }
        var mtlConfig = materialGroup[key];

        if (mtlConfig != undefined && mtlConfig.textures != undefined)
        {
          for (var i = 0; i < mtlConfig.textures.length; ++i)
          {
            var texName = mtlConfig.textures[i].name;
            var imageConfig = imageConfigGroup[texName];
            if (!imageConfig || !Array.isArray(imageConfig.lod))
            {
              continue;
            }
            var maxLodLevel = imageConfig.lod.length;
            if (mtlConfig.textures[i].level < maxLodLevel - 1)
            {
              var cachedInfo = scope.modelCacheMgr.objectsPool[item.gltfHash];

              if (cachedInfo.isVisible)
              {
                totalTasks.push(
                  {
                    groupId: item.groupId,
                    mtlName: key,
                    lodLevel: mtlConfig.textures[i].level,
                    texId: i,
                    imageUrl: joinUrlPath(
                      scope.glbResourcesBaseUrl || scope.resourcesBaseUrl,
                      'task-' + item.groupId + '/images/LOD' + (mtlConfig.textures[i].level + 1) + '/' + encodeURIComponent(imageConfig.uri)
                    ),
                    weightNormalized: 1.0 - item.weightNormalized,
                    texName: texName,
                    isLightMap: mtlConfig.textures[i].channel == 'lightMap',
                  });
                //注意：这里一定需要使用encodeURIComponent来对图片的名称进行编码，否则会出现特殊符号（如#）无法编码，导致加载失败
              }
            }
          }
        }
      };

      var fetchCachedTexture = function(taskData, loadCallback, errorCallback)
      {
        var cacheKey = taskData.groupId + '-' + taskData.texName + "-" + taskData.lodLevel;

        if (scope.loadedTextures[cacheKey])
        {
          if (scope.DebugMode) console.log('texture fectched from cache: ' + taskData.texName + ", " + cacheKey);
          if (loadCallback) loadCallback(scope.loadedTextures[cacheKey], true);
        }
        else
        {
          var loader = scope.textureLoader;

          var imageMeta = 
          {
            'raw' : 'png',
            'type': 'png',
          }
          
          if (taskData.imageUrl.endsWith('.hdr') || taskData.imageUrl.endsWith('.exr'))
          {
            if (taskData.imageUrl.endsWith('.hdr'))
            {
              imageMeta.raw = 'hdr';
              imageMeta.type = 'hdr';
            }
            else if (taskData.imageUrl.endsWith('.exr'))
            {
              imageMeta.raw = 'exr';
              imageMeta.type = 'exr';
            }

            if (scope.sceneConfig.materials.useHdrJpg)
            {
              if (scope.hdrjpgLoader == null)
              {
                scope.hdrjpgLoader = new HDRJPGLoader(scope.renderer);
              }
              
              loader = scope.hdrjpgLoader;

              imageMeta.type = 'jpgr';
            }
            else
            {
              if (taskData.imageUrl.endsWith('.hdr'))
              {
                loader = scope.hdrLoader;
              }
              else if (taskData.imageUrl.endsWith('.exr'))
              {
                loader = scope.exrLoader;
              }
            }
          }
          
          loader.load(taskData.imageUrl, function(_textureData)
          {
            var textureData = null;
            if (imageMeta.type == 'png')
            {
              textureData = new CanvasTexture( _textureData );
            }
            else if (imageMeta.type == 'jpgr')
            {
              textureData = _textureData.renderTarget.texture;
            }
            else
            {
              textureData = _textureData;
            }

            scope.loadedTextures[cacheKey] = textureData;

            if (loadCallback) loadCallback(textureData, false, imageMeta);
          }, null, function(err)
          {
            if (errorCallback) errorCallback(err);
          });
        }
      }

      var loadImage = function (taskData/*, index, array*/) 
      {
        return new Promise(resolve => 
        {
          fetchCachedTexture(taskData,
            function(textureData, isFromCache, imageType)
            {
              scope.refreshMaterialTexture(taskData.groupId, taskData.mtlName, taskData.texId, textureData, isFromCache, imageType);

              if (isFromCache == false)
              {
                // 加入延时，避免占用太多模型加载的资源
                setTimeout(function(){
                  resolve();
                }, 250);
              }
              else
              {
                resolve();
              }
            }, function(err)
            {
              console.log(err);

              scope.refreshMaterialTexture(taskData.groupId, taskData.mtlName, taskData.texId, null);

              resolve();
            }
          ); 
        });
      };

      if (totalTasks.length > 0)
      {
        function compareTexturePriority(keyA, keyB) 
        {
          function getPriority(item)
          {
            var priority = (item.isLightMap? -200 : 0) + item.weightNormalized * -10 + (item.texId + 1) * 100 + (item.lodLevel + 1) * 200;
            return priority;
          }
          
          return getPriority(keyA) - getPriority(keyB);
        }

        totalTasks.sort(compareTexturePriority);

        var AsyncTextureTaskCount = 3;

        var subTasks = totalTasks.slice(0, Math.min(AsyncTextureTaskCount, totalTasks.length)); // 分片加载，防止阻塞

        {
          this.isMaterialImageLoading = true;

          sequentialPromiseMap(subTasks, loadImage).then(results => 
            {
              this.isMaterialImageLoading = false;
            });
        }
        
      }
    }
    this.lastTextureTaskMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - textureTaskStart;
  }

  update(dt, time)
  {
    if (this.isSceneInitialized == false)
    {
      return;
    }

    this.lastFrameDt = Number(dt || 0);
    this.updateLoading(dt);
    this.processPendingSceneInsertions();
    this.syncCamera();

    this.modelCacheMgr.update(time);

    // This only reconciles resident render state.  RenderVisibilitySystem has
    // its own time/camera gate and does not start neural predictions.
    if (this.renderVisibilitySystem && typeof this.renderVisibilitySystem.update === 'function')
    {
      this.renderVisibilitySystem.update();
    }

    this.processImageTask(this.lastFrameDt);
  }

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
      cameraHash: metrics.cameraHash || null,
      requestId: metrics.requestId != null ? Number(metrics.requestId) : null,
      notes: metrics.notes || null,
      renderPolicy: this.getNeuralRenderPolicy(),
      cache: this.modelCacheMgr.getSnapshot(),
      loadQueueLength: this.modelToLoadList.length,
      pendingSceneInsertions: this._getPendingSceneInsertionCount(),
    };
    this.lastBenchmarkVisibilityIds.serial = this.visibilityMetricsSerial;
    this.lastBenchmarkVisibilityIds.mode = metrics.mode || null;
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
        originalMatrices: [],
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
      lastKey: '__invalid__',
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
        lastKey: null,
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

      var originalMatrices = [];
      for (var index = 0; index < node.count; ++index)
      {
        node.getMatrixAt(index, INSTANCED_VISIBILITY_MATRIX);
        originalMatrices.push(INSTANCED_VISIBILITY_MATRIX.clone());
      }

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
        originalMatrices: originalMatrices,
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
      lastKey: null,
    };
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
      this._refreshInstancedMeshBounds(mesh);
    }
  }

  _applyInstancedVisibility(componentIds)
  {
    var previousRenderable = this.lastRenderableComponentIdsForInstancing || [];
    var currentSet = new Set(this._normalizeIdList(componentIds));
    var nowMs = this.modelCacheMgr && typeof this.modelCacheMgr.getNowMs === 'function'
      ? this.modelCacheMgr.getNowMs()
      : (typeof performance !== 'undefined' ? performance.now() : Date.now());
    var retainMs = this.useNeuralPVS ? Math.max(0, Number(this.neuralRenderRetainMs || 0)) : 0;

    // 实例化构件的显示也需要短暂保留，否则一次预测抖动会立刻把实例矩阵移走，视觉上就是闪烁。
    if (!this.instancedRetainUntilByComponentId)
    {
      this.instancedRetainUntilByComponentId = new Map();
    }
    if (retainMs > 0)
    {
      for (var prevIdx = 0; prevIdx < previousRenderable.length; ++prevIdx)
      {
        var previousId = previousRenderable[prevIdx];
        if (!currentSet.has(previousId))
        {
          var existingRetainUntil = this.instancedRetainUntilByComponentId.get(previousId) || 0;
          this.instancedRetainUntilByComponentId.set(previousId, Math.max(existingRetainUntil, nowMs + retainMs));
        }
      }
      this.instancedRetainUntilByComponentId.forEach(function(retainUntil, componentId)
      {
        if (currentSet.has(componentId) || retainUntil <= nowMs)
        {
          this.instancedRetainUntilByComponentId.delete(componentId);
        }
        else
        {
          currentSet.add(componentId);
        }
      }, this);
    }
    else
    {
      this.instancedRetainUntilByComponentId.clear();
    }

    this.lastRenderableComponentIdsForInstancing = this._normalizeIdList(Array.from(currentSet));

    if (!this.instancedVisibilityBindingByComponentId || Object.keys(this.instancedVisibilityBindingByComponentId).length === 0)
    {
      return;
    }

    var wantedByHash = {};
    for (var i = 0; i < this.lastRenderableComponentIdsForInstancing.length; ++i)
    {
      var componentId = this.lastRenderableComponentIdsForInstancing[i];
      var binding = this.instancedVisibilityBindingByComponentId[componentId];
      if (!binding || !binding.hash)
      {
        continue;
      }
      if (!wantedByHash[binding.hash])
      {
        wantedByHash[binding.hash] = [];
      }
      wantedByHash[binding.hash].push(binding.instanceIndex);
    }

    var touchedHashes = new Set(Object.keys(this.loadedInstancedVisibilityStatesByHash || {}));
    for (var hashKey in wantedByHash)
    {
      touchedHashes.add(hashKey);
    }

    touchedHashes.forEach((hash) =>
    {
      var state = this._ensureLoadedInstancedVisibilityState(hash);
      if (!state)
      {
        return;
      }

      var activeIndices = (wantedByHash[hash] || []).filter(function(index)
      {
        return Number.isFinite(index) && index >= 0 && index < state.originalCount;
      }).sort(function(a, b){ return a - b; });
      var activeKey = activeIndices.join(',');

      if (state.disabled)
      {
        this._hideInvalidInstancedVisibilityState(state);
        state.lastKey = '__invalid__';
        return;
      }

      if (state.lastKey === activeKey)
      {
        return;
      }

      for (var meshStateIdx = 0; meshStateIdx < state.meshStates.length; ++meshStateIdx)
      {
        var meshState = state.meshStates[meshStateIdx];
        for (var visibleIndex = 0; visibleIndex < activeIndices.length; ++visibleIndex)
        {
          meshState.mesh.setMatrixAt(visibleIndex, meshState.originalMatrices[activeIndices[visibleIndex]]);
        }
        meshState.mesh.count = activeIndices.length;
        meshState.mesh.visible = activeIndices.length > 0;
        if (meshState.mesh.instanceMatrix)
        {
          meshState.mesh.instanceMatrix.needsUpdate = true;
        }
        this._refreshInstancedMeshBounds(meshState.mesh);
      }

      state.lastKey = activeKey;
    });
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

  _filterComponentIdsByCurrentFrustum(componentIds)
  {
    if (!this.runtimeFrustumFilterEnabled || this.runtimeVisibilityMeta == null)
    {
      return this._normalizeIdList(componentIds);
    }

    this._updateRuntimeVisibilityFrustum(this.activeCamera);
    var scope = this;
    return this._normalizeIdList((componentIds || []).filter(function(componentId)
    {
      var record = scope.componentVisibilityRecords[componentId];
      return record && scope._intersectsFrustumWithCenterSize(record.bounds);
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
      cameraHash: this.rvCameraHash,
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
      renderComponentIds: this._normalizeIdList(payload.renderComponentIds),
      renderGlbIds: this._normalizeIdList(payload.renderGlbIds),
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
      renderComponentIds: this.lastBenchmarkVisibilityIds.renderComponentIds.slice(),
      renderGlbIds: this.lastBenchmarkVisibilityIds.renderGlbIds.slice(),
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
    return {
      startup: Object.assign({}, this.startupMetrics),
      visibility: this.lastVisibilityMetrics,
      cache: this.modelCacheMgr.getSnapshot(),
      load: {
        queueLength: this.modelToLoadList.length,
        queuePreview: this.modelToLoadList.slice(Math.max(0, this.modelToLoadList.length - 8)).map(function(item)
        {
          return {
            id: item.id,
            weight: Number(item.weight || 0),
            prefetch: Boolean(item.prefetch),
            epoch: item.predictionEpoch,
          };
        }),
        inflightCount: this.inflightModelHashes ? this.inflightModelHashes.size : 0,
        pendingHashCount: this.pendingSceneInsertionHashes ? this.pendingSceneInsertionHashes.size : 0,
        wantedHashCount: this.currentDownloadWantedHashes ? this.currentDownloadWantedHashes.size : 0,
        pendingSceneInsertions: this._getPendingSceneInsertionCount(),
        activeDirectLoadCount: this.activeDirectLoadCount,
        prefetchQueueLength: this.pendingPrefetchList.length,
        prefetchPreview: this.pendingPrefetchList.slice(0, 8).map(function(item)
        {
          return {
            id: item.id,
            weight: Number(item.weight || 0),
            epoch: item.predictionEpoch,
          };
        }),
        totalIntegrated: this.integratedSceneCount,
        lastIntegratedCount: this.lastLoadMetrics.lastIntegratedCount,
        lastBatchMs: this.lastLoadMetrics.lastBatchMs,
        lastIntegratedAt: this.lastLoadMetrics.lastIntegratedAt,
        lastTextureTaskMs: this.lastTextureTaskMs,
        staleCachedCount: this.lastLoadMetrics.staleCachedCount,
        staleDroppedCount: this.lastLoadMetrics.staleDroppedCount,
        httpAdaptive: Object.assign({}, this.httpAdaptiveState, {
          enabled: this.httpAdaptiveConcurrencyEnabled,
          minConcurrency: this.httpAdaptiveMinConcurrency,
          maxConcurrency: this.httpAdaptiveMaxConcurrency,
          prefetchMinConcurrency: this.httpAdaptivePrefetchMinConcurrency,
          prefetchMaxConcurrency: this.httpAdaptivePrefetchMaxConcurrency,
          saturationMaxConcurrency: this.neuralBandwidthSaturationMaxConcurrency,
          saturationPrefetchMaxConcurrency: this.neuralBandwidthSaturationPrefetchMaxConcurrency,
        }),
      },
      neural: {
        cullingMode: this.cullingMode,
        enabled: this.useNeuralPVS,
        neuralBackend: this.neuralBackend,
        debugNoCache: this.neuralDebugNoCache,
        loadLimitsDisabled: Boolean(this.neuralDisableLoadLimits),
        backend: this.neuralPVS ? this.neuralPVS.backend : null,
        ready: this.neuralPVS ? this.neuralPVS.isReady : false,
        modelInfo: this.neuralPVS ? (this.neuralPVS.modelInfo || (this.neuralPVS.lastPredictTimings ? this.neuralPVS.lastPredictTimings.modelInfo : null)) : null,
        initTimings: this.neuralPVS ? this.neuralPVS.lastInitTimings : null,
        predictTimings: this.neuralPVS ? this.neuralPVS.lastPredictTimings : null,
        idMode: this.neuralPVSIdMode,
        resourceTransport: this.neuralUseResourcesWS ? 'resourcesWS' : 'http',
        resourcesWSConfigured: Boolean(this.resourcesWS),
        resourcesWSConnected: Boolean(this.connectedAssets),
        resourceWS: this._getNeuralWSStats(),
        cpuPerfMode: this.cpuPerfMode,
        renderVisibility: this.renderVisibilitySystem ? this.renderVisibilitySystem.lastStats : null,
        renderRefresh: this.lastRenderRefreshStats,
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

  forceSceneCullingNow(options = {})
  {
    startupLog('slm2:forceSceneCullingNow', options);
    this.lastCameraPosHash = null;
    this.lastCameraRotHash = null;
    this.lastScreenSizeHash = null;
    if (options.forceNeural === true || (options.forceNeural !== false && this.useNeuralPVS))
    {
      this.forceNextNeuralPrediction = true;
    }
    this.sceneCulling();
  }

  _ensureNeuralPVSInit()
  {
    startupLog('slm2:ensureNeuralInit:enter', {
      hasNeuralPVS: Boolean(this.neuralPVS),
      backend: this.neuralBackend,
      hasRuntimeMeta: Boolean(this.runtimeVisibilityMeta),
      isReady: Boolean(this.neuralPVS && this.neuralPVS.isReady),
    });
    if (!this.neuralPVS)
    {
      return null;
    }
    var initPromise = this.neuralPVS.init();
    startupLog('slm2:ensureNeuralInit:init-called', {
      hasPromise: Boolean(initPromise && typeof initPromise.then === 'function'),
    });
    if (initPromise && typeof initPromise.then === 'function' && !this.neuralPVSReadyCullingScheduled)
    {
      var scope = this;
      this.neuralPVSReadyCullingScheduled = true;
      initPromise.then(function()
      {
        scope.neuralPVSReadyCullingScheduled = false;
        if (scope.useNeuralPVS && scope.neuralPVS && scope.neuralPVS.isReady)
        {
          startupLog('slm2:ensureNeuralInit:ready');
          scope.neuralPVSInitWarningShown = false;
          scope.forceNextNeuralPrediction = true;
          scope.forceSceneCullingNow({ forceNeural: true });
        }
        else if (scope.useNeuralPVS && scope.neuralPVS && scope.neuralPVS.initError)
        {
          startupLog('slm2:ensureNeuralInit:error', {
            message: scope.neuralPVS.initError && scope.neuralPVS.initError.message
              ? scope.neuralPVS.initError.message
              : String(scope.neuralPVS.initError),
          });
          scope.forceSceneCullingNow({ forceNeural: false });
        }
      }).catch(function()
      {
        startupLog('slm2:ensureNeuralInit:promise-rejected');
        scope.neuralPVSReadyCullingScheduled = false;
        if (scope.useNeuralPVS)
        {
          scope.forceSceneCullingNow({ forceNeural: false });
        }
      });
    }

    return initPromise;
  }

  _scheduleNeuralPVSInit()
  {
    startupLog('slm2:scheduleNeuralInit:enter', {
      useNeuralPVS: this.useNeuralPVS,
      scheduled: this.neuralPVSInitScheduled,
      ready: Boolean(this.neuralPVS && this.neuralPVS.isReady),
      hasRuntimeMeta: Boolean(this.runtimeVisibilityMeta),
    });
    if (!this.useNeuralPVS || !this.neuralPVS || this.neuralPVSInitScheduled || this.neuralPVS.isReady)
    {
      return;
    }
    var scope = this;
    this.neuralPVSInitScheduled = true;
    var startInit = function()
    {
      scope.neuralPVSInitScheduled = false;
      scope._ensureNeuralPVSInit();
    };

    // requestAdapter 在个别驱动上可能触发短暂 GPU 设备枚举卡顿。
    // 这里至少让 viewer 首帧和交互控制先完成挂载，再启动 WebGPU 初始化。
    if (typeof requestAnimationFrame === 'function')
    {
      requestAnimationFrame(function()
      {
        setTimeout(startInit, 0);
      });
    }
    else
    {
      setTimeout(startInit, 0);
    }
  }

  _scheduleRuntimeVisibilityMetaLoad(callback)
  {
    startupLog('slm2:scheduleRuntimeMeta:enter', {
      loaded: Boolean(this.runtimeVisibilityMeta),
      scheduled: this.runtimeVisibilityMetaLoadScheduled,
      inFlight: this.runtimeVisibilityMetaLoadInFlight,
    });
    if (this.runtimeVisibilityMeta)
    {
      if (callback) callback();
      return;
    }
    if (this.runtimeVisibilityMetaLoadScheduled || this.runtimeVisibilityMetaLoadInFlight)
    {
      return;
    }

    var scope = this;
    this.runtimeVisibilityMetaLoadScheduled = true;
    var startLoad = function()
    {
      startupLog('slm2:runtimeMeta:load-start');
      scope.runtimeVisibilityMetaLoadScheduled = false;
      scope.runtimeVisibilityMetaLoadInFlight = true;
      scope._loadRuntimeVisibilityMeta(function()
      {
        startupLog('slm2:runtimeMeta:load-callback', {
          loaded: Boolean(scope.runtimeVisibilityMeta),
          ms: scope.startupMetrics.runtimeVisibilityMetaMs,
        });
        scope.runtimeVisibilityMetaLoadInFlight = false;
        if (callback) callback();
      });
    };

    setTimeout(function()
    {
      startupLog('slm2:runtimeMeta:schedule-timer-fired');
      if (typeof requestIdleCallback === 'function')
      {
        requestIdleCallback(function()
        {
          startupLog('slm2:runtimeMeta:idle-start');
          startLoad();
        }, { timeout: 1500 });
      }
      else
      {
        startLoad();
      }
    }, 250);
  }

  _scheduleMaterialConfigLoad(callback)
  {
    startupLog('slm2:scheduleMaterial:enter', {
      scheduled: this.materialConfigLoadScheduled,
      started: this.materialConfigLoadStarted,
    });
    if (this.materialConfigLoadScheduled || this.materialConfigLoadStarted)
    {
      return;
    }

    var scope = this;
    var startedWaitingAt = performance.now();
    this.materialConfigLoadScheduled = true;
    var initialDelayMs = this.useNeuralPVS ? 6000 : 3000;
    var maxDelayMs = this.useNeuralPVS ? 18000 : 10000;

    var tryStart = function()
    {
      startupLog('slm2:scheduleMaterial:try-start');
      if (scope.materialConfigLoadStarted)
      {
        return;
      }

      var hasPendingSceneWork = scope.pendingSceneInsertions && scope.pendingSceneInsertions.length > scope.pendingSceneInsertionCursor;
      var neuralBusy = Boolean(scope.neuralPredictionInFlight);
      var waitedMs = performance.now() - startedWaitingAt;
      if ((hasPendingSceneWork || neuralBusy) && waitedMs < maxDelayMs)
      {
        setTimeout(tryStart, 750);
        return;
      }

      var startLoad = function()
      {
        scope.materialConfigLoadStarted = true;
        scope.materialConfigLoadScheduled = false;
        startupLog('slm2:material:load-start');
        scope.loadMaterialConfig(callback);
      };

      if (typeof requestIdleCallback === 'function')
      {
        requestIdleCallback(function()
        {
          startupLog('slm2:material:idle-start');
          startLoad();
        }, { timeout: 2000 });
      }
      else
      {
        setTimeout(startLoad, 0);
      }
    };

    setTimeout(tryStart, initialDelayMs);
  }

  syncCamera()
  {
    this.backCamera.position.copy(this.activeCamera.position);
    this.backCamera.rotation.copy(this.activeCamera.rotation);

    this.backCamera.updateMatrixWorld();
  }

  sceneCulling() 
  {
    var logThisCulling = this.startupSceneCullingLogCount < 20;
    if (logThisCulling)
    {
      this.startupSceneCullingLogCount += 1;
      startupLog('slm2:sceneCulling:enter', {
        connected: this.connected,
        useNeuralPVS: this.useNeuralPVS,
        fullLoadMode: this.fullLoadMode,
        sceneInitialized: this.isSceneInitialized,
        neuralReady: Boolean(this.neuralPVS && this.neuralPVS.isReady),
      });
    }
    if(!this.connected && !this.useNeuralPVS && !this.fullLoadMode && this.cullingMode !== 'frustum')
    {
        if (logThisCulling) startupLog('slm2:sceneCulling:return:not-connected');
        return;
    }

    if (this.isSceneInitialized == false)
    {
      if (logThisCulling) startupLog('slm2:sceneCulling:return:not-initialized');
      return;
    }

    // Bypass RC-Server culling and load all objects in the scene immediately
    if (this.fullLoadMode) 
    {
      // Strategy 1: Disable CacheMgr auto-eviction in full load mode
      this.modelCacheMgr.setSchedulingStrategy('manual');

      if (!this.hasFullLoaded) {
         var fullLoadIdMode = this.defaultVisibilityIdMode || 'component';
         var objCount = this._getMaxModelId(fullLoadIdMode);
         var allModels = [];
         var allWeights = [];
         for(var i = 0; i <= objCount; i++) {
            allModels.push(i);
            allWeights.push(999); // Force high weight so decodeModelInfo doesn't skip it (weight < 10 is skipped)
         }
         this.refreshLoadingTask(allModels, allWeights, fullLoadIdMode);
         this.hasFullLoaded = true;
         this._failedRetries = {}; // Initialize retry tracker
         console.log("[FullLoadMode] Queued all " + (objCount + 1) + " objects for loading in mode: " + fullLoadIdMode + ".");
      }

      // Keep all loaded objects marked as visible so CacheMgr never evicts them
      var pool = this.modelCacheMgr.objectsPool;
      for (var key in pool) {
        if (pool[key].meshObject) {
          pool[key].isVisible = true;
          pool[key].lastUpdateTime = this.modelCacheMgr.timeSinceStartup;
          this.modelCacheMgr.applyRenderState(pool[key]);
        }
      }

      return; 
    }
    else if (this.frozenPredictionInspectActive)
    {
      this.modelCacheMgr.setSchedulingStrategy('manual');
      this._applyFrozenPredictionInspectSnapshot();
      return;
    }
    else
    {
      // Restore normal eviction when not in full load mode
      this.modelCacheMgr.setSchedulingStrategy('auto');
    }

    if (this.lastCameraPosHash == undefined)
    {
      this.lastCameraPosHash = null;
      this.lastCameraRotHash = null;
      this.lastScreenSizeHash = null;
    }

    var posHashResolution = 1;
    var rotHashResolution = 5;
    var cameraPosHash = (this.activeCamera.position.x * posHashResolution).toFixed(0) + "-" + 
                        (this.activeCamera.position.y * posHashResolution).toFixed(0) + "-" + 
                        (this.activeCamera.position.z * posHashResolution).toFixed(0);
    var cameraRotHash = (this.activeCamera.rotation.x * rotHashResolution).toFixed(0) + "-" + 
                        (this.activeCamera.rotation.y * rotHashResolution).toFixed(0) + "-" +
                        (this.activeCamera.rotation.z * rotHashResolution).toFixed(0);

    var screenSizeHash = this.clientWidth + "-" + this.clientHeight;

    if (this.disableCameraHash ||
        this.useNeuralPVS ||
        (cameraPosHash != this.lastCameraPosHash ||
        cameraRotHash != this.lastCameraRotHash ||
        screenSizeHash != this.lastScreenSizeHash) || 
        this.requestId < 2 // 初始阶段强制刷新两次
        )
    {
      
      this.lastCameraPosHash = cameraPosHash;
      this.lastCameraRotHash = cameraRotHash;
      this.lastScreenSizeHash = screenSizeHash;

      const viewMatrix = this.backCamera.matrixWorldInverse;
      const projectionMatrix = this.backCamera.projectionMatrix;
      const mvpMatrix = new Matrix4();
      mvpMatrix.multiplyMatrices(projectionMatrix, viewMatrix);

      const rootMatrix = this.rootScene.matrixWorld;

      mvpMatrix.multiply(rootMatrix);

      // 创建一个 Matrix4 对象
      let mvp = [];

      // 遍历 elements 数组，将每个元素精度转换成两位小数
      mvpMatrix.elements.map(value => {
          mvp.push(value); // 保留两位小数，并将字符串转换回数字
      });

      // 放大视口，以便获得更为富裕的可见集合，在摄像机转动时减少构件缺失
      var CameraFrameUpscale = 1.2;
      var clientWidth = Math.round(this.clientWidth * CameraFrameUpscale);
      var clientHeight = Math.round(this.clientHeight * CameraFrameUpscale);

      var newCameraHash = cameraPosHash + ':' + cameraRotHash;
      let camera_data = {
          "type": 0,
          "id": this.requestId,
          "mvp": mvp,
          "width": clientWidth, 
          "height": clientHeight,
          "cull": 0, // 0: no cull, 1: front cull, 2: back cull
          "hash": (this.rvCameraHash == null ? '0' : newCameraHash)
      };

      if (this.neuralDebugLogs) console.log(camera_data);

      // Update with new hash
      this.rvCameraHash = newCameraHash;

      let requestSentToServer = false;

      if (this.cullingMode === 'frustum')
      {
        if (!this.runtimeVisibilityMeta)
        {
          if (!this.runtimeVisibilityMetaLoadScheduled && !this.runtimeVisibilityMetaLoadInFlight)
          {
            this._scheduleRuntimeVisibilityMetaLoad(() =>
            {
              if (this.cullingMode === 'frustum')
              {
                this.forceSceneCullingNow({ forceNeural: false });
              }
            });
          }
          return;
        }
        this._applyLocalFrustumCulling();
        return;
      }

      if (this.useNeuralPVS) {
        // Use local Neural Network prediction instead of remote rcServer WebSocket
        if (this.neuralPVS && this.neuralPVS.isReady) {
          if (logThisCulling) startupLog('slm2:sceneCulling:neural-ready');
          this.neuralPVSInitWarningShown = false;
          var forceNeuralPrediction = Boolean(this.forceNextNeuralPrediction);
          if (this.neuralPredictionInFlight)
          {
            if (forceNeuralPrediction)
            {
              this.neuralPendingPredictionAfterCurrent = true;
            }
            return;
          }
          if (!forceNeuralPrediction && !this.neuralPredictionGate.shouldPredict(this.activeCamera))
          {
            return;
          }
          this.forceNextNeuralPrediction = false;
          const predictionDispatcher = this.neuralPVS;
          const predictionGroupSerial = this.neuralGroupSwitchSerial;
          const predictionLifecycleSerial = this.neuralPredictionLifecycleSerial;
          const predictionProcess = async () => {
            this.neuralPredictionInFlight = true;
            const predictionStart = performance.now();
            var isFirstStartupPrediction = this.startupMetrics.firstPredictionMs <= 0;
            try {
              startupLog('slm2:neural:predict-start');
              var rawPredictionStart = performance.now();
              const pred = await predictionDispatcher.predict(this.activeCamera);
              var rawPredictionMs = performance.now() - rawPredictionStart;
              if (predictionGroupSerial !== this.neuralGroupSwitchSerial ||
                  predictionLifecycleSerial !== this.neuralPredictionLifecycleSerial ||
                  this.frozenPredictionInspectActive)
              {
                return;
              }
              startupLog('slm2:neural:predict-raw-done', {
                ms: rawPredictionMs,
                rawGlbCount: pred && pred.modelList ? pred.modelList.length : 0,
                rawComponentCount: pred && pred.componentModelList ? pred.componentModelList.length : 0,
                candidateCount: pred && pred.candidateCount != null ? pred.candidateCount : null,
              });
              if (isFirstStartupPrediction)
              {
                this.startupMetrics.firstPredictionRawMs = rawPredictionMs;
              }
              if (pred) {
                this.neuralPredictionGate.commit(this.activeCamera);
                this.lastNeuralPrediction = pred;
                var predIdMode = pred.idMode || this.neuralPVSIdMode;
                if (this.neuralDebugLogs) console.log('[SLM2Loader] NeuralPVS raw prediction', {
                  idMode: predIdMode,
                  backend: pred.backend,
                  rawGlbCount: pred.modelList ? pred.modelList.length : 0,
                  rawComponentCount: pred.componentModelList ? pred.componentModelList.length : 0,
                  rawGlbPreview: Array.from(pred.modelList || []).slice(0, 16),
                });
                var applyStart = performance.now();
                var appliedVisibility = this._applyLightweightNeuralPlan(pred, predIdMode);
                var applyMs = performance.now() - applyStart;
                startupLog('slm2:neural:apply-done', {
                  ms: applyMs,
                  loadNowGlbCount: appliedVisibility && Array.isArray(appliedVisibility.modelList) ? appliedVisibility.modelList.length : 0,
                  renderGlbCount: appliedVisibility && Array.isArray(appliedVisibility.renderModelList) ? appliedVisibility.renderModelList.length : 0,
                });
                if (isFirstStartupPrediction)
                {
                  this.startupMetrics.firstPredictionApplyMs = applyMs;
                  this.startupMetrics.firstPredictionMs = performance.now() - predictionStart;
                  stopStartupLog('first-neural-prediction');
                }
                if (appliedVisibility && appliedVisibility.stale)
                {
                  return;
                }
                if (this.neuralDebugLogs) console.log('[SLM2Loader] NeuralPVS applied visibility', {
                  idMode: predIdMode,
                  rawGlbCount: pred.modelList ? pred.modelList.length : 0,
                  filteredGlbCount: Array.isArray(appliedVisibility.modelList) ? appliedVisibility.modelList.length : 0,
                  renderGlbCount: Array.isArray(appliedVisibility.renderModelList) ? appliedVisibility.renderModelList.length : 0,
                  renderResidentCount: appliedVisibility.renderVisibleCount != null ? appliedVisibility.renderVisibleCount : null,
                  filteredGlbPreview: Array.isArray(appliedVisibility.modelList) ? appliedVisibility.modelList.slice(0, 16) : [],
                  schedulerStats: appliedVisibility.schedulerStats || null,
                });
                this._updateBenchmarkVisibilityIds({
                  mode: 'neural',
                  rawComponentIds: pred.componentModelList || [],
                  rawGlbIds: pred.modelList || [],
                  scheduledComponentIds: appliedVisibility.componentModelList || [],
                  scheduledGlbIds: appliedVisibility.modelList || [],
                  renderComponentIds: appliedVisibility.renderComponentModelList || [],
                  renderGlbIds: appliedVisibility.renderModelList || [],
                  prefetchComponentIds: appliedVisibility.prefetchComponentModelList || [],
                  prefetchGlbIds: (appliedVisibility.prefetchList || []).map(function(item){ return item.id; }),
                  priorityItems: appliedVisibility.priorityItems || [],
                  candidateSelection: pred && (pred.candidateSelection || (pred.timings && pred.timings.candidateSelection)) || null,
                });
                this._recordVisibilityMetrics({
                  mode: 'neural',
                  idMode: predIdMode,
                  backend: pred.backend,
                  latencyMs: performance.now() - predictionStart,
                  rawCount: pred.modelList ? pred.modelList.length : 0,
                  visibleCount: appliedVisibility.renderVisibleCount != null
                    ? appliedVisibility.renderVisibleCount
                    : (Array.isArray(appliedVisibility.renderModelList) ? appliedVisibility.renderModelList.length : (Array.isArray(appliedVisibility.modelList) ? appliedVisibility.modelList.length : 0)),
                  rawGlbCount: pred.modelList ? pred.modelList.length : 0,
                  visibleGlbCount: appliedVisibility.renderVisibleCount != null
                    ? appliedVisibility.renderVisibleCount
                    : (Array.isArray(appliedVisibility.renderModelList) ? appliedVisibility.renderModelList.length : (Array.isArray(appliedVisibility.modelList) ? appliedVisibility.modelList.length : 0)),
                  rawInstanceCount: pred.componentModelList ? pred.componentModelList.length : 0,
                  visibleInstanceCount: appliedVisibility.renderComponentModelList
                    ? appliedVisibility.renderComponentModelList.length
                    : 0,
                  cameraHash: newCameraHash,
                  notes: {
                    loadNowGlbCount: Array.isArray(appliedVisibility.modelList) ? appliedVisibility.modelList.length : 0,
                    renderCandidateGlbCount: Array.isArray(appliedVisibility.renderModelList) ? appliedVisibility.renderModelList.length : 0,
                    renderResidentCount: appliedVisibility.renderVisibleCount != null ? appliedVisibility.renderVisibleCount : null,
                  },
                });
                if (this.onVisibilityReceived) this.onVisibilityReceived(appliedVisibility.modelList, appliedVisibility.weightList, predIdMode);
              }
            } finally {
              this.neuralPredictionInFlight = false;
              if (this.neuralPendingPredictionAfterCurrent)
              {
                this.neuralPendingPredictionAfterCurrent = false;
                this.forceNextNeuralPrediction = true;
              }
            }
          };
          predictionProcess();
        } else {
          var neuralInitFailed = Boolean(this.neuralPVS && this.neuralPVS.initError && !this.neuralPVS.initPromise);
          if (!neuralInitFailed)
          {
            this._ensureNeuralPVSInit();
            if (!this.neuralPVSInitWarningShown)
            {
              startupLog('slm2:sceneCulling:return:neural-loading');
              console.log('[NeuralPVS] Model is still loading; waiting for neural assets before culling.');
              this.neuralPVSInitWarningShown = true;
            }
            return;
          }

          if (!this.neuralPVSInitFallbackWarningShown) {
            const loadError = this.neuralPVS.initError && this.neuralPVS.initError.message
              ? this.neuralPVS.initError.message
              : this.neuralPVS.initError;
            console.warn('[NeuralPVS] Model failed to initialize; culling=neural does not use rvcServer fallback.',
              loadError ? `Last load error: ${loadError}` : '');
            this.neuralPVSInitFallbackWarningShown = true;
          }
          return;
        }
      } else {
        // Use rvcServer culling.
        this.ws.send(JSON.stringify(camera_data));
        requestSentToServer = true;
      }

      if (requestSentToServer) {
        this.requestMap[camera_data.id] = {
          "start": new Date().getTime(),
          "foi_received": 0,
          "foi_sent": 0,
          "end": 0,
          "mode": 'rcserver'
        }

        this.requestId++;
      }
    }
  }

  fetchCameraVisibilityList(callback)
  {
    this.disableCameraHash = true;

    this.sceneCulling();

    this.cullingCallback = function(modelList, weightList, idMode)
    {
      if (callback)
      {
        var visibilityList = 
        {
          modelList: modelList,
          weightList: weightList
        };

        var objCount = this._getMaxModelId(idMode || this.rcServerIdMode);
        var idFlagList = [];
        for (var i = 0; i <= objCount; ++i)
        {
          idFlagList.push(false);
        }

        var allIds = [];

        for (var i = 0; i < modelList.length; ++i)
        {
          idFlagList[modelList[i]] = true;

          allIds.push(modelList[i]);
        }

        // 依次添加不可见的对象
        for (var i = 0; i < idFlagList.length; ++i)
        {
          if (idFlagList[i] == false)
          {
            allIds.push(i);
          }
        }

        callback(allIds);
      }
    };

    this.disableCameraHash = false;
  }

  startConnectAsset()
  {
    startupLog('slm2:assetsWS:connect-start', {
      usePool: this._shouldUseNeuralResourceWSPool(),
      resourcesWS: this.resourcesWS,
    });
    if (this._shouldUseNeuralResourceWSPool())
    {
      this._ensureNeuralResourceWSPool();
      return;
    }

    if (this.resourcesWS != undefined && this.resourcesWS != null && this.resourcesWS !== '')
    {
      var scope = this;

      this.wsAssets = new WebSocket(this.resourcesWS)
      this.wsAssets.binaryType = 'arraybuffer';
      this.wsAssets.onopen = (evt) => {
        scope.connectedAssets = true
          startupLog('slm2:assetsWS:connect-succeed');
          console.log("assets server connect succeed")
      }
      this.wsAssets.onclose = (evt) => 
      {
        scope.connectedAssets = false;
        console.log('assets onclose');
      }

      this.wsAssets.onmessage = function (event) 
      {
        var arrayBuffer = event.data;

        var bufferOffset = 0;

        const uint32View = new Uint32Array(arrayBuffer.slice(bufferOffset, 4));
        bufferOffset += 4;

        var descJsonLength = uint32View[0];

        var jsonDescString =  new TextDecoder().decode(arrayBuffer.slice(bufferOffset, bufferOffset + descJsonLength));
        bufferOffset += descJsonLength;

        var jsonDesc = JSON.parse(jsonDescString);

        var dataBuffers = [];

        for (var i = 0; i < jsonDesc.bufLengths.length; ++i)
        {
          var gltfArrayBuffer = arrayBuffer.slice(bufferOffset, bufferOffset + jsonDesc.bufLengths[i]);
          bufferOffset += jsonDesc.bufLengths[i];

          dataBuffers.push(gltfArrayBuffer);
        }

        if (scope.wsAssetsCallbacks && scope.wsAssetsCallbacks[jsonDesc.type])
        {
          scope.wsAssetsCallbacks[jsonDesc.type](dataBuffers);

          scope.wsAssetsCallbacks[jsonDesc.type] = null;
        }
      };
    }
  }

  tryStaticLoading()
  {
    if (this.schedulingStrategy == 'static')
    {
      var scope = this;

      var fileLoader = new FileLoader();
      fileLoader.load(this.resourcesBaseUrl + "/initial.json", function(data) 
      {
        var initialJson = JSON.parse(data);

        var modelList = initialJson.id;
        var weightList  = initialJson.weight;

        scope.refreshLoadingTask(modelList, weightList, scope.defaultVisibilityIdMode);
      });
    }
  }

  startConnect()
  {
    var scope = this;

    this.loadingTimescale = 1;

    this.requestId = 0;
    this.requestMap = {};

    this.server_ip = this.rcServerAddress;//'ws://127.0.0.1:5600';

    if (this.server_ip == undefined)
    {
      this.tryStaticLoading();

      return;
    }

    this.ws = new WebSocket(this.server_ip)
    this.ws.binaryType = 'arraybuffer';
    this.ws.onopen = (evt) => {
      scope.connected = true
      console.log("rc server connect succeed")
    }
    this.ws.onclose = (evt) => 
    {
      scope.connected = false;
      console.log('onclose');
    }

    this.ws.onmessage = function (event) 
    {
      const arrayBuffer = event.data;
      const dataView = new DataView(arrayBuffer);
      const decoder = new TextDecoder('utf-8'); // 假设数据是以UTF-8编码的
      const jsonString = decoder.decode(dataView);
      const jsonObject = JSON.parse(jsonString);
      const modelList = JSON.parse(jsonObject.list);
      const weightList = JSON.parse(jsonObject.weight);
      
      if (scope.requestMap[jsonObject.id])
      {
        scope.requestMap[jsonObject.id].foi_received = jsonObject.start;
        scope.requestMap[jsonObject.id].foi_sent = jsonObject.end;
        scope.requestMap[jsonObject.id].end = new Date().getTime();
      }

      var requestMetrics = scope.requestMap[jsonObject.id];
      if (scope.useNeuralPVS && scope.neuralPVS && scope.neuralPVS.isReady)
      {
        if (scope.neuralDebugLogs) console.log('[SLM2Loader] Ignoring rcServer response because neural mode is active and ready.', {
          requestId: jsonObject.id,
          requestMode: requestMetrics ? requestMetrics.mode : null,
        });
        delete scope.requestMap[jsonObject.id];
        return;
      }

      if (scope.cullingCallback)
      {
        scope.cullingCallback(modelList, weightList, scope.rcServerIdMode);

        scope.cullingCallback = null;
      }
      else
      {
        scope.refreshLoadingTask(modelList, weightList, scope.defaultVisibilityIdMode);
      }

      requestMetrics = scope.requestMap[jsonObject.id];
      scope._recordVisibilityMetrics({
        mode: 'rcserver',
        idMode: scope.defaultVisibilityIdMode,
        backend: 'websocket',
        latencyMs: requestMetrics ? (requestMetrics.end - requestMetrics.start) : 0,
        rawCount: Array.isArray(modelList) ? modelList.length : 0,
        visibleCount: Array.isArray(modelList) ? modelList.length : 0,
        rawGlbCount: Array.isArray(modelList) ? new Set(modelList.map(function(componentId)
        {
          var componentRecord = scope.componentVisibilityRecords[componentId];
          return componentRecord ? componentRecord.globalGlbId : componentId;
        })).size : 0,
        visibleGlbCount: Array.isArray(modelList) ? new Set(modelList.map(function(componentId)
        {
          var componentRecord = scope.componentVisibilityRecords[componentId];
          return componentRecord ? componentRecord.globalGlbId : componentId;
        })).size : 0,
        rawInstanceCount: Array.isArray(modelList) ? modelList.length : 0,
        visibleInstanceCount: Array.isArray(modelList) ? modelList.length : 0,
        requestId: jsonObject.id,
        cameraHash: scope.rvCameraHash,
        notes: requestMetrics ? {
          foiReceived: requestMetrics.foi_received,
          foiSent: requestMetrics.foi_sent,
        } : null,
      });
      scope._updateBenchmarkVisibilityIds({
        mode: 'rcserver',
        rawComponentIds: modelList || [],
        rawGlbIds: (modelList || []).map(function(componentId)
        {
          var componentRecord = scope.componentVisibilityRecords[componentId];
          return componentRecord ? componentRecord.globalGlbId : null;
        }),
        scheduledComponentIds: modelList || [],
        scheduledGlbIds: (modelList || []).map(function(componentId)
        {
          var componentRecord = scope.componentVisibilityRecords[componentId];
          return componentRecord ? componentRecord.globalGlbId : null;
        }),
        renderComponentIds: modelList || [],
        renderGlbIds: (modelList || []).map(function(componentId)
        {
          var componentRecord = scope.componentVisibilityRecords[componentId];
          return componentRecord ? componentRecord.globalGlbId : null;
        }),
        prefetchComponentIds: [],
        prefetchGlbIds: [],
      });
      delete scope.requestMap[jsonObject.id];

      if (scope.DebugMode) console.time('loading');
    };
  };

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
      if (this.cullingUpdateDelta > 80)
      {
        this.sceneCulling();
        this.cullingUpdateDelta = 0;
      }
  
      this.cullingUpdateDelta += dt;

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
        retainVisibleMs: this.neuralRenderRetainMs,
        missTolerance: this.neuralRenderMissTolerance,
        attachVisibleToScene: true,
        detachHiddenFromScene: true,
        renderRoot: this.rootScene,
      };
    }

    renderOptions.fallbackVisibleHashes = new Set();
    return renderOptions;
  }

  _startDirectModelLoad(modelInfo, targetDirectLoadConcurrency)
  {
    var scope = this;
    var modelDesc = modelInfo != undefined ? scope.getModelDesc(modelInfo) : null;
    var modelURL = modelDesc != null ? modelDesc.url : null;
    if (modelURL == null)
    {
      return false;
    }
    if (modelDesc.hash && this.neuralWSHTTPFallbackHashes)
    {
      this.neuralWSHTTPFallbackHashes.delete(modelDesc.hash);
    }

    var capturedModel = modelInfo;
    var capturedLoaderIdx = this.directLoaderCursor % targetDirectLoadConcurrency;
    this.directLoaderCursor++;
    var capturedModelDesc = modelDesc;
    var directLoadStartedAt = performance.now();
    var directLoadProgressBytes = 0;
    if (capturedModelDesc && capturedModelDesc.hash)
    {
      scope.inflightModelHashes.add(capturedModelDesc.hash);
    }
    this.activeDirectLoadCount++;

    scope.gltfLoaders[capturedLoaderIdx].load(modelURL, (gltf) =>
    {
      var directLoadEndedAt = performance.now();
      var sampleDesc = directLoadProgressBytes > 0
        ? Object.assign({}, capturedModelDesc || {}, { sizeKB: directLoadProgressBytes / 1024 })
        : capturedModelDesc;
      scope._recordHttpDirectLoadSample(modelURL, directLoadStartedAt, directLoadEndedAt, sampleDesc, true);
      scope.processLaodedGltf(gltf, capturedModelDesc);
      scope.activeDirectLoadCount = Math.max(0, scope.activeDirectLoadCount - 1);
    }, function(progressEvent)
    {
      if (progressEvent && Number(progressEvent.loaded) > 0)
      {
        directLoadProgressBytes = Math.max(directLoadProgressBytes, Number(progressEvent.loaded));
      }
    }, function(err)
    {
      var directLoadFailedAt = performance.now();
      var sampleDesc = directLoadProgressBytes > 0
        ? Object.assign({}, capturedModelDesc || {}, { sizeKB: directLoadProgressBytes / 1024 })
        : capturedModelDesc;
      scope._recordHttpDirectLoadSample(modelURL, directLoadStartedAt, directLoadFailedAt, sampleDesc, false);
      console.error('[LoadError]', modelURL, err);
      if (capturedModelDesc && capturedModelDesc.hash)
      {
        scope.inflightModelHashes.delete(capturedModelDesc.hash);
      }

      if (scope.fullLoadMode && capturedModel) {
        if (!scope._failedRetries) scope._failedRetries = {};
        var modelKey = capturedModel.id !== undefined ? capturedModel.id : modelURL;
        var retries = scope._failedRetries[modelKey] || 0;
        if (retries < 3) {
          scope._failedRetries[modelKey] = retries + 1;
          scope.modelToLoadList.push(capturedModel);
          console.warn('[FullLoadMode] Re-queued failed model (attempt ' + (retries + 1) + '/3): ' + modelURL);
        } else {
          console.error('[FullLoadMode] Gave up on model after 3 retries: ' + modelURL);
        }
      }

      scope.activeDirectLoadCount = Math.max(0, scope.activeDirectLoadCount - 1);
    });

    return true;
  }

  _dispatchNeuralResourceWSBatches()
  {
    var pool = this._ensureNeuralResourceWSPool();
    if (!pool || !pool.hasReadyConnection())
    {
      return 0;
    }

    var started = 0;
    while (this.modelToLoadList.length > 0 &&
      pool.hasReadyConnection() &&
      pool.getInFlightModelCount() < this.neuralWSMaxInFlightModels &&
      this.pendingWSParseBytes < this.neuralWSMaxPendingParseBytes &&
      (this.neuralDisableLoadLimits || this._getPendingSceneInsertionCount() < this.maxPendingSceneInsertions))
    {
      var reqData = [];
      var reqDescs = [];
      while (this.modelToLoadList.length > 0 &&
        reqData.length < this.neuralWSBatchSize &&
        (pool.getInFlightModelCount() + reqData.length) < this.neuralWSMaxInFlightModels)
      {
        var nextModel = this.modelToLoadList.pop();
        if (nextModel && nextModel.forceHttp)
        {
          this.neuralWSHTTPFallbackQueue.push(nextModel);
          continue;
        }

        var modelDesc = nextModel != undefined ? this.getModelDesc(nextModel) : null;
        if (!modelDesc || modelDesc.url == null)
        {
          continue;
        }
        modelDesc.sourceModelInfo = nextModel;

        var reqItem = this._buildResourceWSRequestItem(modelDesc);
        if (!reqItem)
        {
          this._queueNeuralWSHttpFallback(modelDesc, 'invalid-ws-request-item');
          continue;
        }

        if (modelDesc.hash)
        {
          this.inflightModelHashes.add(modelDesc.hash);
        }
        reqData.push(reqItem);
        reqDescs.push(modelDesc);
      }

      if (reqData.length === 0)
      {
        break;
      }

      var dispatched = pool.dispatchBatch(reqData, reqDescs);
      if (!dispatched)
      {
        for (var i = 0; i < reqDescs.length; ++i)
        {
          this._retryOrFallbackNeuralWSModel(reqDescs[i], 'dispatch-failed');
        }
        break;
      }
      started++;
    }
    return started;
  }

  _moveNeuralModelsToHttpFallback(limit)
  {
    var moved = 0;
    while (this.modelToLoadList.length > 0 && moved < limit)
    {
      var item = this.modelToLoadList.pop();
      if (!item)
      {
        continue;
      }
      this.neuralWSHTTPFallbackQueue.push(Object.assign({}, item, { forceHttp: true }));
      moved++;
    }
    return moved;
  }

  _processNeuralWSHttpFallbackLoads(targetDirectLoadConcurrency)
  {
    var started = 0;
    while (this.neuralWSHTTPFallbackQueue.length > 0 &&
      this.activeDirectLoadCount < targetDirectLoadConcurrency &&
      (this.neuralDisableLoadLimits || this._getPendingSceneInsertionCount() < this.maxPendingSceneInsertions))
    {
      var modelInfo = this.neuralWSHTTPFallbackQueue.shift();
      if (!modelInfo)
      {
        continue;
      }

      var decodedForFallback = this.decodeModelInfo(modelInfo);
      var startedOne = this._startDirectModelLoad(modelInfo, targetDirectLoadConcurrency);
      if (!startedOne && decodedForFallback && decodedForFallback.hash && this.neuralWSHTTPFallbackHashes)
      {
        this.neuralWSHTTPFallbackHashes.delete(decodedForFallback.hash);
      }
      if (startedOne)
      {
        started++;
      }
    }
    return started;
  }

  processLoadingList()
  {
    var scope = this;

    var hasResourceWS = this.resourcesWS != undefined && this.resourcesWS != null && this.resourcesWS !== '';
    var useNeuralResourceWSPool = this._shouldUseNeuralResourceWSPool();
    var loadFromStreamingServer = hasResourceWS && !this.useNeuralPVS && !this.fullLoadMode;

    var targetDirectLoadConcurrency = this.fullLoadMode
      ? this.fullLoadDirectLoadConcurrency
      : (this.useNeuralPVS ? this.neuralDirectLoadConcurrency : this.baseDirectLoadConcurrency);
    var prefetchOnlyDirectLoads = false;
    if (this.useNeuralPVS && this.modelToLoadList.length === 0 && this.pendingPrefetchList.length > 0)
    {
      targetDirectLoadConcurrency = this.neuralPrefetchDirectLoadConcurrency;
      prefetchOnlyDirectLoads = true;
      while (this.pendingPrefetchList.length > 0 &&
        (this.neuralDisableLoadLimits || this.modelToLoadList.length < this.neuralPrefetchDirectLoadConcurrency))
      {
        this.modelToLoadList.push(this.pendingPrefetchList.shift());
      }
    }
    this.maxPendingSceneInsertions = this.useNeuralPVS
      ? (this.neuralDisableLoadLimits ? Number.POSITIVE_INFINITY : this.neuralMaxPendingSceneInsertions)
      : 24;
    this.loadIntegrationBudgetMs = this.useNeuralPVS
      ? this.neuralV2LoadIntegrationBudgetMs
      : 6;
    targetDirectLoadConcurrency = this.useNeuralPVS && this.neuralDisableLoadLimits
      ? this._getUnlimitedNeuralDirectLoadTarget(prefetchOnlyDirectLoads)
      : this._getHttpDirectLoadConcurrency(targetDirectLoadConcurrency, prefetchOnlyDirectLoads);

    if (this.gltfLoaders == undefined)
    {
      this.gltfLoaders = [];
    }

    while (this.gltfLoaders.length < targetDirectLoadConcurrency)
    {
      this.gltfLoaders.push(new GLTFLoader()
      .setCrossOrigin('anonymous')
      .setDRACOLoader( DRACO_LOADER )
      .setKTX2Loader( KTX2_LOADER.detectSupport( this.renderer ) )
      .setMeshoptDecoder( MeshoptDecoder ));
    }

    while (this.gltfLoaders.length < this.neuralWSParseConcurrency)
    {
      this.gltfLoaders.push(new GLTFLoader()
      .setCrossOrigin('anonymous')
      .setDRACOLoader( DRACO_LOADER )
      .setKTX2Loader( KTX2_LOADER.detectSupport( this.renderer ) )
      .setMeshoptDecoder( MeshoptDecoder ));
    }

    this.processNeuralWSParseQueue();

    if (useNeuralResourceWSPool)
    {
      var pool = this._ensureNeuralResourceWSPool();
      if (this.neuralDisableLoadLimits || this._getPendingSceneInsertionCount() < this.maxPendingSceneInsertions)
      {
        this._dispatchNeuralResourceWSBatches();
      }

      var wsStats = pool ? pool.getStats() : {};
      var poolUnavailable = this.neuralWSFallbackHttp &&
        this.modelToLoadList.length > 0 &&
        Number(wsStats.readyConnections || 0) === 0 &&
        Number(wsStats.inFlightBatches || 0) === 0 &&
        this.neuralWSStartedAt > 0 &&
        (performance.now() - this.neuralWSStartedAt) >= this.neuralWSFallbackDelayMs;
      if (poolUnavailable)
      {
        this._moveNeuralModelsToHttpFallback(Math.max(1, targetDirectLoadConcurrency - this.activeDirectLoadCount));
      }
      this._processNeuralWSHttpFallbackLoads(targetDirectLoadConcurrency);
      return;
    }

    if (this.modelToLoadList.length == 0 ||
        (loadFromStreamingServer && this.modelIsLoading == true) ||
        (!this.neuralDisableLoadLimits && this._getPendingSceneInsertionCount() >= this.maxPendingSceneInsertions))
    {
      return;
    }

    if (scope.batchModels == undefined)
    {
      scope.batchModels = {};
    }
    scope.batchModels = {};
    scope.batchModelDescs = {};

    var directLoadsStarted = 0;
    while (this.modelToLoadList.length > 0)
    {
      if (!loadFromStreamingServer && this.activeDirectLoadCount >= targetDirectLoadConcurrency)
      {
        break;
      }

      var nextModel = this.modelToLoadList.pop();

      var modelDesc = nextModel != undefined ? scope.getModelDesc(nextModel) : null;

      var modelURL = modelDesc != null ? modelDesc.url : null;

      if (modelURL != null)
      {
        if (loadFromStreamingServer == false)
        {
          if (this._startDirectModelLoad(nextModel, targetDirectLoadConcurrency))
          {
            directLoadsStarted++;
          }
        }
        else
        {
          var succ = scope.addNewBatchModel(modelDesc);

          if (succ)
          {
            directLoadsStarted++;
            if (directLoadsStarted >= this.gltfLoaders.length)
            {
              break;
            }
          }
        }
      }
    }

    if (loadFromStreamingServer && Object.keys(scope.batchModels).length > 0)
    {
      //console.log('loading from streaming server');
      scope.modelIsLoading = true;

      scope.loadBatchFromStreamingServer(function()
      {
        scope.modelIsLoading = false;
      });
    }

    if (this.modelToLoadList.length == 0 && this.activeDirectLoadCount === 0)
    {
      if (scope.DebugMode) console.timeEnd('loading');
    }
  }

  processLaodedGltf(gltf, modelDesc = null)
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
      this.pendingSceneInsertionHashes.add(hash);
      this.inflightModelHashes.delete(hash);
      if (this.neuralWSRetryCounts)
      {
        delete this.neuralWSRetryCounts[hash];
      }
      if (this.neuralWSHTTPFallbackHashes)
      {
        this.neuralWSHTTPFallbackHashes.delete(hash);
      }
    }
    this.lastLoadMetrics.queueLength = this._getPendingSceneInsertionCount();
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
    }

    if (extras && extras.hashCode &&
        (this.instancedVisibilityBindingsByHash[extras.hashCode] ||
         this._isInstancedBindingExpected(extras.hashCode)))
    {
      // A GLB that is expected to contain multiple mapped instances must be
      // validated before it can become visible. Missing metadata is a hard
      // error, never permission to render the complete GLB.
      this._ensureLoadedInstancedVisibilityState(extras.hashCode);
      this._applyInstancedVisibility(this.lastRenderableComponentIdsForInstancing || []);
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
  }

  addNewBatchModel(modelDesc)
  {
    if (!modelDesc || modelDesc.group == null)
    {
      return false;
    }

    var lodLevel = this.MeshLodLevel;
    var baseId = modelDesc.baseId != null
      ? Number(modelDesc.baseId)
      : (modelDesc.hash ? Number.parseInt(String(modelDesc.hash).split('-')[1]) : NaN);

    if (!Number.isFinite(baseId))
    {
      return false;
    }

    var newModel =
    {
      g: modelDesc.group,
      l: lodLevel,
      s: baseId
    }

    var modelKey = newModel.g + '-' + newModel.l + '-' + newModel.s;

    if (this.batchModels[modelKey] == undefined)
    {
      this.batchModels[modelKey] = newModel;
      this.batchModelDescs[modelKey] = modelDesc;
      if (modelDesc.hash)
      {
        this.inflightModelHashes.add(modelDesc.hash);
      }

      return true;
    }

    return false;
  }

  loadBatchFromStreamingServer(callback)
  {
    var scope = this;
    var reqData = [];
    var reqDescs = [];

    for (var key in this.batchModels)
    {
      var item = this.batchModels[key];
      reqData.push(item);
      reqDescs.push(this.batchModelDescs ? this.batchModelDescs[key] : null);
    };
    this.batchModelDescs = {};

    if (reqData.length > 0)
    {
      var batchStartedAt = performance.now();
      this.legacyResourcesWSBatchId = Number(this.legacyResourcesWSBatchId || 0) + 1;
      var batchId = 'legacy-' + this.legacyResourcesWSBatchId;
      let wsReq = {
        "type": 'req',
        "data": reqData
      };
  
      this.fectchBatchedModels(wsReq, function(gltfBuffers)
      {
        var parseTasks = [];
        var batchElapsedMs = performance.now() - batchStartedAt;

        for (let i = 0; i < gltfBuffers.length; ++i)
        {
          var newTask = new Promise((resolve, reject) => 
          {
            var loader = scope.gltfLoaders[i % scope.gltfLoaders.length];
            var modelDesc = reqDescs[i] || null;
            var buffer = gltfBuffers[i];
            var byteLength = buffer && buffer.byteLength ? buffer.byteLength : 0;
            var parseStartedAt = performance.now();
            loader.parse(buffer, '', (gltf) => 
            {
              scope.processLaodedGltf(gltf, modelDesc);
              resolve();
            }, reject);
          }).catch(function(err)
          {
            var failedDesc = reqDescs[i] || null;
            if (failedDesc && failedDesc.hash)
            {
              scope.inflightModelHashes.delete(failedDesc.hash);
            }
            console.error('[LoadError][resourcesWS-parse]', failedDesc ? failedDesc.hash : null, err);
          });

          parseTasks.push(newTask);
        }

        for (let missingIdx = gltfBuffers.length; missingIdx < reqDescs.length; ++missingIdx)
        {
          var missingDesc = reqDescs[missingIdx] || null;
          if (missingDesc && missingDesc.hash)
          {
            scope.inflightModelHashes.delete(missingDesc.hash);
          }
        }

        Promise.all(parseTasks).then((results) =>
        {
          if (callback)
          {
            callback();
          }
        });
      });
    }
    else
    {
      if (callback)
      {
        callback();
      }
    }
  }

  fectchBatchedModels(req, callback)
  {
    if (!this.wsAssets || this.wsAssets.readyState !== WebSocket.OPEN)
    {
      console.warn('[SLM2Loader] resourcesWS is not connected; skipped streaming batch.');
      if (callback)
      {
        callback([]);
      }
      return;
    }

    if (this.wsAssetsCallbacks == undefined)
    {
      this.wsAssetsCallbacks = {};
    }

    this.wsAssetsCallbacks[req.type] = callback;

    this.wsAssets.send(JSON.stringify(req));
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

  _applyLightweightNeuralPlan(predictionPayload, idMode)
  {
    var appliedIdMode = idMode === 'global-glb-priority' ? 'global-glb' : (idMode || 'global-glb');
    var predictedImmediateInfos = this._makeGlobalGlbModelInfos(
      predictionPayload && predictionPayload.immediateGlbIds,
      predictionPayload && predictionPayload.immediateWeights,
      appliedIdMode
    );
    var initialInfos = this._takeNeuralInitialLoadInfos(appliedIdMode);
    var immediateIds = new Set();
    var immediateInfos = [];
    for (var initialIndex = 0; initialIndex < initialInfos.length; ++initialIndex)
    {
      immediateInfos.push(initialInfos[initialIndex]);
      immediateIds.add(initialInfos[initialIndex].id);
    }
    for (var predictedIndex = 0; predictedIndex < predictedImmediateInfos.length; ++predictedIndex)
    {
      var predictedInfo = predictedImmediateInfos[predictedIndex];
      if (!immediateIds.has(predictedInfo.id))
      {
        immediateInfos.push(predictedInfo);
        immediateIds.add(predictedInfo.id);
      }
    }
    var predictedPrefetchInfos = this._makeGlobalGlbModelInfos(
      predictionPayload && predictionPayload.prefetchGlbIds,
      predictionPayload && predictionPayload.prefetchWeights,
      appliedIdMode,
      { prefetch: true, deferredVisible: true }
    );
    var prefetchInfos = predictedPrefetchInfos.filter(function(item){
      return !immediateIds.has(item.id);
    });
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
    var renderInfos = this._makeGlobalGlbModelInfos(
      renderGlbIds,
      null,
      appliedIdMode
    );

    var predictionEpoch = this._setCurrentNeuralWorkingSet(renderInfos, appliedIdMode);
    for (var immediateTagIdx = 0; immediateTagIdx < immediateInfos.length; ++immediateTagIdx)
    {
      immediateInfos[immediateTagIdx].predictionEpoch = predictionEpoch;
    }
    for (var prefetchTagIdx = 0; prefetchTagIdx < prefetchInfos.length; ++prefetchTagIdx)
    {
      prefetchInfos[prefetchTagIdx].predictionEpoch = predictionEpoch;
    }

    this._setCurrentDownloadWantedHashes(immediateInfos.concat(prefetchInfos));
    this._pruneStalePendingInsertions();

    this.modelToLoadList = [];
    for (var immediateIndex = immediateInfos.length - 1; immediateIndex >= 0; --immediateIndex)
    {
      this.modelToLoadList.push(immediateInfos[immediateIndex]);
    }
    this.pendingPrefetchList = prefetchInfos.slice();

    this.lastRenderRefreshStats = this.modelCacheMgr.refreshVisible(
      renderInfos,
      this._getRenderVisibilityOptions(appliedIdMode)
    );

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
      activeVisibleGlbCount: predictionPayload && predictionPayload.timings ? predictionPayload.timings.activeVisibleGlbCount : null,
      viewcellPrefetchGlbCount: predictionPayload && predictionPayload.timings ? predictionPayload.timings.viewcellPrefetchGlbCount : null,
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
      this.modelToLoadList.push(modelInfo);
      neuralWorkingInfos.push(modelInfo);
    }

    if (neuralGlobalMode)
    {
      var neuralPredictionEpoch = this._setCurrentNeuralWorkingSet(neuralWorkingInfos, idMode || this.defaultVisibilityIdMode);
      for (var infoIdx = 0; infoIdx < this.modelToLoadList.length; ++infoIdx)
      {
        this.modelToLoadList[infoIdx].predictionEpoch = neuralPredictionEpoch;
      }
      this._setCurrentDownloadWantedHashes(this.modelToLoadList);
      this._pruneStalePendingInsertions();
      var renderStatsV1 = this._isResidentRenderPolicy()
        ? this._showAllResidentObjects()
        : this.modelCacheMgr.refreshVisible(
          neuralWorkingInfos,
          this._getRenderVisibilityOptions(idMode || this.defaultVisibilityIdMode)
        );
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
      this.currentDownloadWantedHashes = new Set();
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
    this._applyInstancedVisibility(appliedVisibility.renderComponentModelList || []);
    return appliedVisibility;
  }

  setSize(clientWidth, clientHeight)
  {
    this.clientWidth = clientWidth;
    this.clientHeight = clientHeight;

    this.backCamera.aspect = clientWidth / clientHeight;
    this.backCamera.updateProjectionMatrix();

    this.screenPixelReciprocal = 1.0 / (this.clientWidth * this.clientHeight);
  }

  getRVCServerUrl(sceneName, callback)
  {
    var scope = this;
    if (this.rcServerAddress != undefined && this.rcServerAddress != null)
    {
      if (callback)
      {
        callback();
      }
      return;
    }
    if (this.lbServer != undefined && this.lbServer != null)
    {
      var requestOptions = {
        method: 'GET',
        redirect: 'follow'
      };
      
      var lbsURL = this.lbServer + "/getRVC?scene=" + sceneName;
      console.log(lbsURL);
      fetch(lbsURL, requestOptions)
        .then(response => {
          if (response.status != 200)
          {
            throw new Error('LB Server no response!');
          }
          else
          {
            return response.text();
          }
        })
        .then(result => {
          if (result == undefined)
          {
            throw new Error('No avilable rvc candiate!');
          }
          var rtData = JSON.parse(result);
          if (rtData.data != null && rtData.data.url != null)
          {
            scope.rcServerAddress = rtData.data.url;
            console.log('new rc url: ' + scope.rcServerAddress);

            if (callback)
            {
              callback();
            }
          }
          else
          {
            throw new Error('No valid rvc url!');
          }
        })
        .catch(error => {
          console.log(error)
          if (callback)
          {
            callback();
          }
        });
    }
    else
    {
      if (callback)
      {
        callback();
      }
    }
  }

  loadStatic(sceneConfigUrl, renderer, camera, options, callback)
  {
    var baseConfig = 
    {
      name: 'default',
      lbServer: null,
      loader: {
        resourcesBaseUrl: sceneConfigUrl.replace('/sceneWeb.json', ''),
        schedulingStrategy: "static"
      }
    };

    return this.load(baseConfig, renderer, camera, options, callback);
  }

  load(baseConfig, renderer, camera, options, callback)
  {
    startupLog('slm2:load:start');
    this.resourcesBaseUrl = baseConfig.loader.resourcesBaseUrl;
    this.glbResourcesBaseUrl = baseConfig.loader.glbResourcesBaseUrl || baseConfig.loader.modelResourcesBaseUrl || this.resourcesBaseUrl;
    this.neuralAssetGroups = Array.isArray(baseConfig.loader.neuralAssetGroups)
      ? baseConfig.loader.neuralAssetGroups.slice()
      : [];
    this.neuralAssetGroupId = Number.isFinite(Number(baseConfig.loader.neuralAssetGroupId))
      ? Number(baseConfig.loader.neuralAssetGroupId)
      : 0;
    this.neuralAssetBaseUrl = baseConfig.loader.neuralAssetBaseUrl || baseConfig.loader.neuralAssetBase || null;
    this.neuralRuntimeMetaUrl = baseConfig.loader.neuralRuntimeMetaUrl || null;
    this.glbIndexUrl = baseConfig.loader.glbIndexUrl || null;
    var hasConfiguredGlbResourcesBaseUrl = Boolean(baseConfig.loader.glbResourcesBaseUrl || baseConfig.loader.modelResourcesBaseUrl);
    this.resourcesWS = baseConfig.loader.resourcesWS;
    this.remoteResourcesWS = baseConfig.loader.remoteResourcesWS || baseConfig.loader.resourcesWS || null;
    this.rcServerAddress = baseConfig.loader.rcServerAddress;
    this.schedulingStrategy = baseConfig.loader.schedulingStrategy;
    this.lbServer = baseConfig.lbServer;
    this.options = options || {};
    var params = this.options.paramJson || {};
    var cullingParam = params['culling'] == null ? 'neural' : String(params['culling']);
    var cullingParamLower = cullingParam.toLowerCase();
    var cullingMode = cullingParamLower === 'rvcserver' ? 'rvcServer' : cullingParamLower;
    if (cullingMode === 'aabb')
    {
      cullingMode = 'frustum';
    }
    if (cullingMode !== 'neural' && cullingMode !== 'frustum' && cullingMode !== 'rvcServer' && cullingMode !== 'false')
    {
      console.warn('[SLM2Loader] Unknown culling mode "' + cullingParam + '"; expected "neural", "frustum", "rvcServer", or "false". Falling back to neural.');
      cullingMode = 'neural';
    }
    this.cullingMode = cullingMode;
    this.useNeuralPVS = this.cullingMode === 'neural';
    this.fullLoadMode = this.cullingMode === 'false';
    this.hasFullLoaded = false;
    this.neuralPredictionInFlight = false;
    this.neuralPendingPredictionAfterCurrent = false;
    this.neuralDebugLogs = params['neuralDebugLogs'] === 'true' || params['debugNeural'] === 'true';
    this.neuralPVSInitScheduled = false;
    this.materialConfigLoadScheduled = false;
    this.materialConfigLoadStarted = false;
    this.runtimeVisibilityMetaLoadScheduled = false;
    this.runtimeVisibilityMetaLoadInFlight = false;
    this.startupSceneCullingLogCount = 0;
    startupLog('slm2:load:mode', {
      cullingMode: this.cullingMode,
      backend: this.neuralBackend,
      resourcesBaseUrl: this.resourcesBaseUrl,
      glbResourcesBaseUrl: this.glbResourcesBaseUrl,
      resourcesWS: this.resourcesWS,
    });
    if (params['resourcesWS'])
    {
      this.resourcesWS = params['resourcesWS'];
      this.remoteResourcesWS = params['resourcesWS'];
    }
    if (params['resourcesBaseUrl'])
    {
      this.resourcesBaseUrl = params['resourcesBaseUrl'];
      if (!hasConfiguredGlbResourcesBaseUrl)
      {
        this.glbResourcesBaseUrl = this.resourcesBaseUrl;
      }
    }
    if (params['glbResourcesBaseUrl'] || params['modelResourcesBaseUrl'])
    {
      this.glbResourcesBaseUrl = params['glbResourcesBaseUrl'] || params['modelResourcesBaseUrl'];
    }
    if (params['rcServerAddress'] || params['rvcServerAddress'])
    {
      this.rcServerAddress = params['rcServerAddress'] || params['rvcServerAddress'];
    }
    var neuralResourceTransport = String(params['neuralResourceTransport'] || params['neuralTransport'] || '').toLowerCase();
    this.neuralUseResourcesWS = neuralResourceTransport === 'ws' ||
      neuralResourceTransport === 'websocket' ||
      params['neuralUseResourcesWS'] === 'true' ||
      params['neuralUseResourcesWS'] === '1' ||
      params['neuralResourcesWS'] === 'true' ||
      params['neuralResourcesWS'] === '1';
    if (this.fullLoadMode)
    {
      this.neuralUseResourcesWS = false;
      this.resourcesWS = null;
    }
    if (this.neuralUseResourcesWS && (this.resourcesWS == null || this.resourcesWS === '') && this.remoteResourcesWS)
    {
      this.resourcesWS = this.remoteResourcesWS;
    }
    this.cpuPerfMode = params['cpuPerfMode'] === 'mobile' ? 'mobile' : 'balanced';
    this._configureHttpAdaptiveConcurrency();
    this._configureNeuralResourceWSOptions(params);
    this.setNeuralRenderPolicy(params['neuralRenderPolicy'] || params['neuralAlwaysRender']);
    this.renderVisibilitySystem.configure(this.cpuPerfMode === 'mobile'
      ? { mobileMinUpdateMs: 100, desktopMinUpdateMs: 100 }
      : { desktopMinUpdateMs: 50 });
    if (this.neuralDebugLogs) console.log('[SLM2Loader] CPU perf mode:', this.cpuPerfMode);
    if (this.neuralDebugLogs) console.log('[SLM2Loader] Neural resource transport:', this.neuralUseResourcesWS ? 'resourcesWS' : 'http', this.resourcesWS || null);
    if (params['neuralRenderRetainMs'] != null)
    {
      this.setNeuralRenderRetainMs(params['neuralRenderRetainMs']);
    }
    if (params['neuralLoadSkippedAsDeferred'] != null)
    {
      this.neuralLoadSkippedAsDeferred = !(params['neuralLoadSkippedAsDeferred'] === 'false' ||
        params['neuralLoadSkippedAsDeferred'] === '0');
    }
    if (params['neuralDeferredSkippedLoadLimit'] != null)
    {
      this.neuralDeferredSkippedLoadLimit = Math.max(0, Number(params['neuralDeferredSkippedLoadLimit'] || 0));
    }
    var gateMode = String(params['neuralPredictionGateMode'] || params['neuralPredictionMode'] || 'viewcell').toLowerCase();
    if (gateMode !== 'delta' && gateMode !== 'viewcell')
    {
      gateMode = 'viewcell';
    }
    var defaultPositionDelta = this.cpuPerfMode === 'mobile' ? 2.5 : 2.0;
    var defaultMinIntervalMs = this.cpuPerfMode === 'mobile' ? 500 : 350;
    this.neuralPredictionGate.configure({
      mode: gateMode,
      positionDelta: this._parseFloatParam(
        params,
        params['neuralPredictionPositionDelta'] != null ? 'neuralPredictionPositionDelta' : 'neuralViewcellRadius',
        defaultPositionDelta,
        0.25,
        30
      ),
      angleDeltaDeg: this._parseFloatParam(params, 'neuralPredictionAngleDeltaDeg', 6, 0.5, 45),
      yawDeltaDeg: this._parseFloatParam(params, 'neuralPredictionYawDeltaDeg', 6, 0.5, 45),
      pitchDeltaDeg: this._parseFloatParam(params, 'neuralPredictionPitchDeltaDeg', 5, 0.5, 45),
      fovDeltaDeg: this._parseFloatParam(params, 'neuralPredictionFovDeltaDeg', 2, 0.25, 30),
      aspectDelta: this._parseFloatParam(params, 'neuralPredictionAspectDelta', 0.08, 0.005, 1),
      minIntervalMs: this._parseIntParam(params, 'neuralPredictionMinIntervalMs', defaultMinIntervalMs, 50, 5000),
    });
    this.neuralBackend = 'lightweight-worker-pvs';
    this.neuralDispatcherMode = 'lightweight-worker';
    var configuredNeuralScene = baseConfig.loader.neuralScene || baseConfig.loader.neuralSceneName || null;
    var configuredNeuralAssetBase = this.neuralAssetBaseUrl || baseConfig.loader.neuralAssetBaseUrl || baseConfig.loader.neuralAssetBase || null;
    var neuralSceneName = String(params['neuralScene'] || params['scene'] || configuredNeuralScene || baseConfig.name || 'hkust-v3');
    var registeredNeuralAssetBase = getInstancePVSAssetBaseUrl(neuralSceneName);
    var requestedNeuralAssetBase = params['neuralAssetBaseUrl'] || params['neuralAssetBase'] || configuredNeuralAssetBase || registeredNeuralAssetBase;
    if (this.useNeuralPVS && !requestedNeuralAssetBase)
    {
      console.warn('[NeuralPVS] 当前场景没有 V4 模型，已切换为实例 AABB 视锥剔除。', neuralSceneName);
      this.useNeuralPVS = false;
      this.cullingMode = 'frustum';
    }
    var neuralAssetBase = requestedNeuralAssetBase ? String(requestedNeuralAssetBase) : '';
    var neuralRuntimeMetaUrl = this.neuralRuntimeMetaUrl || (this.resourcesBaseUrl
      ? this.resourcesBaseUrl + "/runtimeVisibilityMeta.json"
      : "./assets/runtimeVisibilityMeta.json");
    this.neuralAssetBaseUrl = neuralAssetBase;
    this.neuralRuntimeMetaUrl = neuralRuntimeMetaUrl;
    this.neuralPVSOptions = {
      debugLogging: this.neuralDebugLogs,
      assetVersion: LOCAL_RUNTIME_ASSET_VERSION,
      cpuPerfMode: this.cpuPerfMode,
      maxImmediate: this.cpuPerfMode === 'mobile' ? 160 : 384,
      maxPrefetch: this.cpuPerfMode === 'mobile' ? 768 : 2048,
      prefetchThreshold: params['neuralPrefetchThreshold'] != null
        ? Number(params['neuralPrefetchThreshold'])
        : undefined,
      downloadPlanMode: this.getNeuralDownloadPlanMode(),
    };
    this.neuralPVSIdMode = this.useNeuralPVS
      ? 'global-glb-priority'
      : (this.cullingMode === 'frustum' ? 'global-glb' : 'component');
    this.neuralPVS = this.useNeuralPVS
      ? new LightweightPVSDispatcher(neuralAssetBase, this.neuralPVSOptions)
      : null;
    startupLog('slm2:neural-object-created', {
      hasNeuralPVS: Boolean(this.neuralPVS),
      backend: this.neuralBackend,
      scene: neuralSceneName,
      assetBase: neuralAssetBase,
      runtimeMetaUrl: neuralRuntimeMetaUrl,
    });
    this.neuralPVSInitWarningShown = false;
    this.neuralPVSInitFallbackWarningShown = false;
    this.neuralPVSReadyCullingScheduled = false;
    if (this.neuralPVS)
    {
      this.neuralPVS.debugLogging = this.neuralDebugLogs;
    }

    this.modelCacheMgr.setSchedulingStrategy(this.schedulingStrategy);

    this.options = options || {};

    this.renderer = renderer;
  
    this.backCamera = new PerspectiveCamera(MODEL_INPUT_FOV_Y_DEG, camera.aspect, camera.near, camera.far);
    this.activeCamera = camera;
    this.syncCamera();

    var initalSize = new Vector2();
    this.renderer.getSize(initalSize);
    this.setSize(initalSize.x, initalSize.y);

    var scope = this;
    var fileLoader = new FileLoader();
    var sceneWebStartedAt = performance.now();
    startupLog('slm2:sceneWeb:request', {
      url: this.resourcesBaseUrl + "/sceneWeb.json",
    });
		fileLoader.load(this.resourcesBaseUrl + "/sceneWeb.json", function(data) 
    {
      scope.startupMetrics.sceneWebMs = performance.now() - sceneWebStartedAt;
      startupLog('slm2:sceneWeb:loaded', {
        ms: scope.startupMetrics.sceneWebMs,
        bytes: data ? data.length : 0,
      });
      var sceneParseStart = performance.now();
			scope.sceneConfig = JSON.parse(data);
      scope.startupMetrics.sceneWebParseMs = performance.now() - sceneParseStart;
      startupLog('slm2:sceneWeb:parsed', {
        ms: scope.startupMetrics.sceneWebParseMs,
      });
      var instancedStart = performance.now();
      scope._buildInstancedVisibilityBindings();
      scope.startupMetrics.instancedBindingMs = performance.now() - instancedStart;
      startupLog('slm2:instanced-bindings-built', {
        ms: scope.startupMetrics.instancedBindingMs,
      });

      scope._loadGlbIndex(function()
      {
        startupLog('slm2:glbIndex:callback');
        if (callback)
        {
          startupLog('slm2:viewer-callback:before');
          callback(scope.sceneConfig.config);
          startupLog('slm2:viewer-callback:after');
        }

        scope.isSceneInitialized = true;
        startupLog('slm2:sceneInitialized:true');
        if (scope.fullLoadMode)
        {
          setTimeout(function()
          {
            scope.forceSceneCullingNow({ forceNeural: false });
          }, 0);
        }
        else if (scope.useNeuralPVS)
        {
          scope.forceNextNeuralPrediction = true;
          scope._scheduleRuntimeVisibilityMetaLoad(function()
          {
            scope._scheduleNeuralPVSInit();
          });
        }
        else if (scope.cullingMode === 'frustum')
        {
          scope._scheduleRuntimeVisibilityMetaLoad(function()
          {
            scope.forceSceneCullingNow({ forceNeural: false });
          });
        }

        if (scope.useNeuralPVS)
        {
          if (scope.neuralDebugLogs) console.log('[SLM2Loader] culling=neural; rvcServer connection disabled.');
        }
        else if (scope.cullingMode === 'frustum')
        {
          if (scope.neuralDebugLogs) console.log('[SLM2Loader] culling=frustum; using local runtime AABB filtering.');
        }
        else if (!scope.fullLoadMode)
        {
          scope.getRVCServerUrl(baseConfig.name, function()
          {
            scope.startConnect();
          });
        }
        else if (scope.neuralDebugLogs)
        {
          console.log('[SLM2Loader] culling=false; rvcServer/resourcesWS disabled, full-load uses HTTP.');
        }

        scope.startConnectAsset();

        scope._scheduleMaterialConfigLoad(function()
        {
          if (scope.neuralDebugLogs) console.log('[SLM2Loader] Material config loaded.', {
            totalMs: scope.startupMetrics.materialConfigMs,
            reboundMeshes: scope.lastMaterialRebindCount,
          });
        });
      });
		});

    this.rootScene = new Object3D();

    return this.rootScene;
  }

  _loadGlbIndex(callback)
  {
    var scope = this;
    var fileLoader = new FileLoader();
    var glbIndexStartedAt = performance.now();
    startupLog('slm2:glbIndex:request');
    var handleSuccess = function(data)
    {
      try
      {
        startupLog('slm2:glbIndex:loaded', {
          bytes: data ? data.length : 0,
        });
        var parseStart = performance.now();
        var parsedIndex = JSON.parse(data);
        scope._mergeGlbIndex(parsedIndex, scope.globalGlbEntries.length === 0);
        scope.neuralInitialLoadOrder = Array.isArray(parsedIndex && parsedIndex.initialLoadIds)
          ? parsedIndex.initialLoadIds.slice()
          : [];
        scope.neuralInitialLoadPending = scope.neuralInitialLoadOrder.length > 0;
        scope.startupMetrics.glbIndexParseMs = performance.now() - parseStart;
        var buildStart = performance.now();
        scope.startupMetrics.glbIndexBuildMs = performance.now() - buildStart;
        startupLog('slm2:glbIndex:parsed', {
          parseMs: scope.startupMetrics.glbIndexParseMs,
          buildMs: scope.startupMetrics.glbIndexBuildMs,
          entries: (scope.glbIndex && scope.glbIndex.entries || []).length,
        });

        if (scope.neuralDebugLogs || scope.DebugMode) console.log('[SLM2Loader] Loaded GLB index entries: ' + ((scope.glbIndex && scope.glbIndex.entries || []).length));
      }
      catch (err)
      {
        startupLog('slm2:glbIndex:parse-failed', {
          message: err && err.message ? err.message : String(err),
        });
        console.warn('[SLM2Loader] Failed to parse glbIndex.json, falling back to component ids.', err);
        if (scope.globalGlbEntries.length === 0)
        {
          scope.glbIndex = null;
          scope.globalGlbEntries = [];
          scope.globalGlbHashToId = {};
        }
      }

      if (callback)
      {
        scope.startupMetrics.glbIndexMs = performance.now() - glbIndexStartedAt;
        startupLog('slm2:glbIndex:done', {
          ms: scope.startupMetrics.glbIndexMs,
        });
        callback();
      }
    };

    var handleFailure = function()
    {
      startupLog('slm2:glbIndex:failed');
      if (scope.globalGlbEntries.length === 0)
      {
        scope.glbIndex = null;
        scope.globalGlbEntries = [];
        scope.globalGlbHashToId = {};
      }
      console.warn('[SLM2Loader] glbIndex.json not found, continuing with component-id mode only.');
      if (callback)
      {
        scope.startupMetrics.glbIndexMs = performance.now() - glbIndexStartedAt;
        callback();
      }
    };

    // 当前每个场景都有独立的 glbIndex。不要在场景请求失败时回退到历史
    // 根目录 HKUST 索引，否则 Metropolis 可能用错实例到 GLB 的映射。
    var localFallbackUrl = this.glbIndexUrl || this.resourcesBaseUrl
      ? null
      : withLocalRuntimeVersion("./assets/glbIndex.json");
    var remoteUrl = this.glbIndexUrl || (this.resourcesBaseUrl + "/glbIndex.json");
    fileLoader.load(remoteUrl, handleSuccess, null, function()
    {
      if (localFallbackUrl)
      {
        fileLoader.load(localFallbackUrl, handleSuccess, null, handleFailure);
      }
      else
      {
        handleFailure();
      }
    });
  }

  _mergeGlbIndex(index, reset = false)
  {
    if (reset)
    {
      this.globalGlbEntries = [];
      this.globalGlbHashToId = {};
    }
    this.glbIndex = index || this.glbIndex || { entries: [] };
    var entries = Array.isArray(index && index.entries) ? index.entries : [];
    for (var i = 0; i < entries.length; ++i)
    {
      var entry = entries[i];
      if (!entry || entry.globalId == null) continue;
      var globalId = Number(entry.globalId);
      if (!Number.isFinite(globalId) || globalId < 0) continue;
      this.globalGlbEntries[globalId] = entry;
      if (entry.hash != null) this.globalGlbHashToId[entry.hash] = globalId;
    }
  }

  _setRuntimeVisibilityMeta(runtimeMeta)
  {
    this.runtimeVisibilityMeta = runtimeMeta || null;
    this.componentVisibilityRecords = [];
    this.globalGlbVisibilityRecords = [];
    this.globalGlbToComponentIds = [];
    this.expectedInstancedVisibilityHashes = new Set();
    if (!runtimeMeta)
    {
      return;
    }

    var componentRecords = Array.isArray(runtimeMeta.componentRecords)
      ? runtimeMeta.componentRecords
      : [];
    var glbRecords = Array.isArray(runtimeMeta.globalGlbRecords)
      ? runtimeMeta.globalGlbRecords
      : [];
    for (var componentIndex = 0; componentIndex < componentRecords.length; ++componentIndex)
    {
      var componentRecord = componentRecords[componentIndex];
      var componentId = Number(componentRecord && componentRecord.componentGlobalId);
      if (componentRecord && Number.isFinite(componentId) && componentId >= 0)
      {
        this.componentVisibilityRecords[componentId] = componentRecord;
      }
    }
    for (var glbIndex = 0; glbIndex < glbRecords.length; ++glbIndex)
    {
      var glbRecord = glbRecords[glbIndex];
      var globalGlbId = Number(glbRecord && glbRecord.globalGlbId);
      if (!glbRecord || !Number.isFinite(globalGlbId) || globalGlbId < 0) continue;
      this.globalGlbVisibilityRecords[globalGlbId] = glbRecord;
      var componentIds = Array.isArray(glbRecord.componentGlobalIds)
        ? glbRecord.componentGlobalIds.map(Number)
        : [];
      this.globalGlbToComponentIds[globalGlbId] = componentIds;
      var glbHash = glbRecord.glbHash || glbRecord.hash ||
        (this.globalGlbEntries[globalGlbId] && this.globalGlbEntries[globalGlbId].hash);
      if (componentIds.length > 1 && glbHash)
      {
        this.expectedInstancedVisibilityHashes.add(glbHash);
      }
    }
    this._validateResidentInstancedBindings();
  }

  _loadRuntimeVisibilityMeta(callback)
  {
    var scope = this;
    var fileLoader = new FileLoader();
    var runtimeMetaStartedAt = performance.now();
    startupLog('slm2:runtimeMeta:request');
    var handleSuccess = function(data)
    {
      try
      {
        startupLog('slm2:runtimeMeta:loaded', {
          bytes: data ? data.length : 0,
        });
        var parseStart = performance.now();
        var parsedRuntimeMeta = JSON.parse(data);
        scope.startupMetrics.runtimeVisibilityMetaParseMs = performance.now() - parseStart;
        var buildStart = performance.now();
        scope._setRuntimeVisibilityMeta(parsedRuntimeMeta);
        scope.startupMetrics.runtimeVisibilityMetaBuildMs = performance.now() - buildStart;
        startupLog('slm2:runtimeMeta:parsed', {
          parseMs: scope.startupMetrics.runtimeVisibilityMetaParseMs,
          buildMs: scope.startupMetrics.runtimeVisibilityMetaBuildMs,
          components: scope.componentVisibilityRecords.length,
          glbs: scope.globalGlbVisibilityRecords.length,
        });

        if (scope.neuralDebugLogs || scope.DebugMode) console.log('[SLM2Loader] Loaded runtimeVisibilityMeta.json: components=' +
          scope.componentVisibilityRecords.length + ', globalGlbs=' + scope.globalGlbVisibilityRecords.length);
      }
      catch (err)
      {
        startupLog('slm2:runtimeMeta:parse-failed', {
          message: err && err.message ? err.message : String(err),
        });
        scope.runtimeVisibilityMeta = null;
        scope.componentVisibilityRecords = [];
        scope.globalGlbVisibilityRecords = [];
        scope.globalGlbToComponentIds = [];
        scope.expectedInstancedVisibilityHashes = new Set();
        console.warn('[SLM2Loader] Failed to parse runtimeVisibilityMeta.json, frustum post-filter disabled.', err);
      }

      if (callback)
      {
        scope.startupMetrics.runtimeVisibilityMetaMs = performance.now() - runtimeMetaStartedAt;
        startupLog('slm2:runtimeMeta:done', {
          ms: scope.startupMetrics.runtimeVisibilityMetaMs,
        });
        callback();
      }
    };

    var handleFallbackFailure = function()
    {
      startupLog('slm2:runtimeMeta:failed');
      scope.runtimeVisibilityMeta = null;
      scope.componentVisibilityRecords = [];
      scope.globalGlbVisibilityRecords = [];
      scope.globalGlbToComponentIds = [];
      scope.expectedInstancedVisibilityHashes = new Set();
      console.warn('[SLM2Loader] runtimeVisibilityMeta.json not found in scene resources or local assets, continuing without frustum post-filter.');
      if (callback)
      {
        scope.startupMetrics.runtimeVisibilityMetaMs = performance.now() - runtimeMetaStartedAt;
        callback();
      }
    };

    var localFallbackUrl = this.neuralRuntimeMetaUrl ? null : withLocalRuntimeVersion("./assets/runtimeVisibilityMeta.json");
    var remoteUrl = this.neuralRuntimeMetaUrl || (this.resourcesBaseUrl + "/runtimeVisibilityMeta.json");
    fileLoader.load(remoteUrl, handleSuccess, null, function()
    {
      if (localFallbackUrl)
      {
        fileLoader.load(localFallbackUrl, handleSuccess, null, handleFallbackFailure);
      }
      else
      {
        handleFallbackFailure();
      }
    });
  }

  _fetchNeuralGroupJson(url)
  {
    var requestUrl = withLocalRuntimeVersion(url);
    return fetch(requestUrl, { cache: 'force-cache' }).then(function(response)
    {
      if (!response.ok)
      {
        throw new Error('Failed to load grouped scene asset ' + requestUrl + ': ' + response.status);
      }
      return response.text();
    }).then(function(text)
    {
      if (text.trimStart().charAt(0) === '<')
      {
        throw new Error('Expected JSON from grouped scene asset ' + requestUrl);
      }
      return JSON.parse(text);
    });
  }

  _takeNeuralInitialLoadInfos(idMode)
  {
    if (!this.neuralInitialLoadPending || !Array.isArray(this.neuralInitialLoadOrder))
    {
      return [];
    }
    this.neuralInitialLoadPending = false;
    var seen = new Set();
    var infos = [];
    for (var index = 0; index < this.neuralInitialLoadOrder.length; ++index)
    {
      var globalId = Number(this.neuralInitialLoadOrder[index]);
      if (!Number.isFinite(globalId) || seen.has(globalId) || !this.globalGlbEntries[globalId])
      {
        continue;
      }
      seen.add(globalId);
      infos.push({
        id: globalId,
        weight: 1,
        idMode: idMode || 'global-glb',
        initialPreload: true,
      });
    }
    return infos;
  }

  switchNeuralAssetGroup(groupRef)
  {
    var group = typeof groupRef === 'object'
      ? groupRef
      : this.neuralAssetGroups[Number(groupRef)];
    if (!group)
    {
      return Promise.resolve(false);
    }
    var groupId = Number(group.id != null ? group.id : group.sourceViewpointId);
    var assetBaseUrl = group.neuralAssetBaseUrl || group.assetBaseUrl || group.modelAssetBaseUrl;
    var runtimeMetaUrl = group.runtimeMetaUrl || group.neuralRuntimeMetaUrl;
    var glbIndexUrl = group.glbIndexUrl;
    if (!assetBaseUrl || !runtimeMetaUrl || !glbIndexUrl)
    {
      console.warn('[SLM2Loader] Incomplete grouped scene asset config.', group);
      return Promise.resolve(false);
    }
    if (groupId === this.neuralAssetGroupId && assetBaseUrl === this.neuralAssetBaseUrl && this.neuralPVS)
    {
      return Promise.resolve(true);
    }

    var switchSerial = ++this.neuralGroupSwitchSerial;
    this.neuralAssetGroupId = groupId;
    this.neuralAssetBaseUrl = assetBaseUrl;
    this.neuralRuntimeMetaUrl = runtimeMetaUrl;
    this.glbIndexUrl = glbIndexUrl;
    this.neuralInitialLoadOrder = [];
    this.neuralInitialLoadPending = false;
    this.neuralPVSInitScheduled = false;
    this.neuralPVSReadyCullingScheduled = false;
    this.neuralPVSInitFallbackWarningShown = false;

    if (this.neuralPVS && typeof this.neuralPVS.dispose === 'function')
    {
      this.neuralPVS.dispose();
    }
    this.neuralPVS = null;
    this.neuralPredictionInFlight = false;
    this.lastNeuralPrediction = null;
    this.modelToLoadList = [];
    this.pendingPrefetchList = [];
    this.pendingSceneInsertions = [];
    this.pendingSceneInsertionCursor = 0;
    this.pendingSceneInsertionHashes = new Set();
    this.neuralWSHTTPFallbackQueue = [];
    this.neuralWSHTTPFallbackHashes = new Set();
    this.currentDownloadWantedHashes = new Set();
    if (this.renderVisibilitySystem && typeof this.renderVisibilitySystem.clear === 'function')
    {
      this.renderVisibilitySystem.clear();
    }
    // A viewpoint switch changes the ownership of the runtime assets.  Drop
    // resident meshes from the previous region so switching does not silently
    // accumulate all nine regions in the browser cache.
    if (this.modelCacheMgr && typeof this.modelCacheMgr.clearAll === 'function')
    {
      this.modelCacheMgr.clearAll();
    }
    this.loadedInstancedVisibilityStatesByHash = {};
    this.lastRenderableComponentIdsForInstancing = [];
    if (this.instancedRetainUntilByComponentId && typeof this.instancedRetainUntilByComponentId.clear === 'function')
    {
      this.instancedRetainUntilByComponentId.clear();
    }
    try
    {
      if (this.modelCacheMgr && typeof this.modelCacheMgr.refreshVisible === 'function')
      {
        this.modelCacheMgr.refreshVisible([], this._getRenderVisibilityOptions('global-glb'));
      }
      this._applyInstancedVisibility([]);
    }
    catch (error)
    {
      if (this.neuralDebugLogs) console.warn('[SLM2Loader] Failed to clear previous grouped visibility.', error);
    }

    var scope = this;
    return Promise.all([
      this._fetchNeuralGroupJson(runtimeMetaUrl),
      this._fetchNeuralGroupJson(glbIndexUrl),
    ]).then(function(payload)
    {
      if (switchSerial !== scope.neuralGroupSwitchSerial)
      {
        return false;
      }
      scope._setRuntimeVisibilityMeta(payload[0]);
      // Group indexes are complete for the selected region.  Reset the
      // previous index so stale IDs and stale group paths cannot be scheduled.
      scope._mergeGlbIndex(payload[1], true);
      scope.neuralInitialLoadOrder = Array.isArray(payload[1] && payload[1].initialLoadIds)
        ? payload[1].initialLoadIds.slice()
        : [];
      scope.neuralInitialLoadPending = scope.neuralInitialLoadOrder.length > 0;
      var options = Object.assign({}, scope.neuralPVSOptions || {}, {
        debugLogging: scope.neuralDebugLogs,
        assetVersion: LOCAL_RUNTIME_ASSET_VERSION,
      });
      scope.neuralPVS = scope.useNeuralPVS
        ? new LightweightPVSDispatcher(assetBaseUrl, options)
        : null;
      scope.forceNextNeuralPrediction = true;
      if (scope.isSceneInitialized && scope.useNeuralPVS)
      {
        scope.forceSceneCullingNow({ forceNeural: true });
      }
      else if (scope.isSceneInitialized && scope.cullingMode === 'frustum')
      {
        scope.forceSceneCullingNow({ forceNeural: false });
      }
      startupLog('slm2:neural-group-switched', {
        groupId: groupId,
        candidateCount: payload[0] ? Number(payload[0].instanceCount || 0) : null,
      });
      return true;
    }).catch(function(error)
    {
      console.error('[SLM2Loader] Failed to switch grouped scene runtime assets.', error);
      return false;
    });
  }

  _updateRuntimeVisibilityFrustum(cameraOverride = null)
  {
    var filterCamera = cameraOverride || (this.runtimeFrustumFilterSource === 'active-camera' ? this.activeCamera : this.backCamera);
    if (filterCamera == null)
    {
      return null;
    }

    filterCamera.updateMatrixWorld();
    this._runtimeFrustumMatrix.multiplyMatrices(filterCamera.projectionMatrix, filterCamera.matrixWorldInverse);
    this._runtimeFrustum.setFromProjectionMatrix(this._runtimeFrustumMatrix);
    return this._runtimeFrustum;
  }

  _intersectsFrustumWithCenterSize(bounds)
  {
    if (bounds == null || bounds.center == null || bounds.size == null)
    {
      return true;
    }

    this._runtimeTempCenter.set(bounds.center[0], bounds.center[1], bounds.center[2]);
    this._runtimeTempSize.set(bounds.size[0], bounds.size[1], bounds.size[2]);
    this._runtimeTempBox.setFromCenterAndSize(this._runtimeTempCenter, this._runtimeTempSize);
    return this._runtimeFrustum.intersectsBox(this._runtimeTempBox);
  }

  _intersectsFrustumWithMinMax(aabb)
  {
    if (aabb == null || aabb.min == null || aabb.max == null)
    {
      return true;
    }

    this._runtimeTempBox.min.set(aabb.min[0], aabb.min[1], aabb.min[2]);
    this._runtimeTempBox.max.set(aabb.max[0], aabb.max[1], aabb.max[2]);
    return this._runtimeFrustum.intersectsBox(this._runtimeTempBox);
  }

  _globalGlbIntersectsCurrentFrustum(globalGlbId)
  {
    var glbRecord = this.globalGlbVisibilityRecords[globalGlbId];

    if (glbRecord == null)
    {
      return true;
    }

    if (!this._intersectsFrustumWithMinMax(glbRecord.aabb))
    {
      return false;
    }

    var componentIds = this.globalGlbToComponentIds[globalGlbId] || [];
    if (componentIds.length === 0)
    {
      return true;
    }

    for (var c = 0; c < componentIds.length; ++c)
    {
      var componentRecord = this.componentVisibilityRecords[componentIds[c]];
      if (componentRecord != null && this._intersectsFrustumWithCenterSize(componentRecord.bounds))
      {
        return true;
      }
    }

    return false;
  }

  _buildNeuralRenderFallbackHashes(idMode)
  {
    var sourceIdMode = idMode || this.defaultVisibilityIdMode;
    var fallbackHashes = new Set();

    if (!this.useNeuralPVS || !(sourceIdMode === 'global-glb' || sourceIdMode === 'global-glb-priority'))
    {
      return fallbackHashes;
    }

    if (!this.runtimeFrustumFilterEnabled || this.runtimeVisibilityMeta == null)
    {
      return fallbackHashes;
    }

    this._updateRuntimeVisibilityFrustum(this.activeCamera);

    var pool = this.modelCacheMgr && this.modelCacheMgr.objectsPool ? this.modelCacheMgr.objectsPool : {};
    for (var hash in pool)
    {
      var item = pool[hash];
      if (!item || !item.meshObject || !item.isInScene)
      {
        continue;
      }

      var globalGlbId = this.globalGlbHashToId[hash];
      if (globalGlbId == null)
      {
        continue;
      }

      if (this._globalGlbIntersectsCurrentFrustum(globalGlbId))
      {
        fallbackHashes.add(hash);
      }
    }

    return fallbackHashes;
  }

  _filterVisibilityByCameraFrustum(modelList, weightList, idMode)
  {
    var sourceWeights = Array.isArray(weightList) ? weightList : [];
    var sourceIdMode = idMode || this.defaultVisibilityIdMode;
    if (this.useNeuralPVS && sourceIdMode === 'global-glb')
    {
      return {
        modelList: modelList,
        weightList: sourceWeights,
      };
    }

    if (!this.runtimeFrustumFilterEnabled || !Array.isArray(modelList) || modelList.length === 0 || this.runtimeVisibilityMeta == null)
    {
      return {
        modelList: modelList,
        weightList: sourceWeights,
      };
    }

    this._updateRuntimeVisibilityFrustum();

    var filteredIds = [];
    var filteredWeights = [];
    var aabbRejectedCount = 0;
    var componentRejectedCount = 0;
    var keptByMissingMeta = 0;
    var rejectedPreview = [];

    if (sourceIdMode === 'global-glb')
    {
      for (var i = 0; i < modelList.length; ++i)
      {
        var globalGlbId = modelList[i];
        var glbRecord = this.globalGlbVisibilityRecords[globalGlbId];

        if (glbRecord == null)
        {
          filteredIds.push(globalGlbId);
          filteredWeights.push(sourceWeights[i]);
          keptByMissingMeta++;
          continue;
        }

        if (!this._intersectsFrustumWithMinMax(glbRecord.aabb))
        {
          aabbRejectedCount++;
          if (rejectedPreview.length < 12) rejectedPreview.push({ id: globalGlbId, reason: 'glb-aabb' });
          continue;
        }

        var componentIds = this.globalGlbToComponentIds[globalGlbId] || [];
        var isVisibleInFrustum = componentIds.length === 0;
        for (var c = 0; c < componentIds.length && !isVisibleInFrustum; ++c)
        {
          var componentRecord = this.componentVisibilityRecords[componentIds[c]];
          if (componentRecord != null && this._intersectsFrustumWithCenterSize(componentRecord.bounds))
          {
            isVisibleInFrustum = true;
          }
        }

        if (isVisibleInFrustum)
        {
          filteredIds.push(globalGlbId);
          filteredWeights.push(sourceWeights[i]);
        }
        else
        {
          componentRejectedCount++;
          if (rejectedPreview.length < 12) rejectedPreview.push({ id: globalGlbId, reason: 'component-aabb' });
        }
      }
    }
    else if (sourceIdMode === 'component')
    {
      for (var j = 0; j < modelList.length; ++j)
      {
        var componentId = modelList[j];
        var componentInfo = this.componentVisibilityRecords[componentId];
        if (componentInfo == null || this._intersectsFrustumWithCenterSize(componentInfo.bounds))
        {
          filteredIds.push(componentId);
          filteredWeights.push(sourceWeights[j]);
        }
      }
    }
    else
    {
      return {
        modelList: modelList,
        weightList: sourceWeights,
      };
    }

    if (this.runtimeFrustumFilterDebug)
    {
      console.log('[SLM2Loader] Frustum post-filter idMode=' + sourceIdMode +
        ' before=' + modelList.length + ' after=' + filteredIds.length +
        ' camera=' + this.runtimeFrustumFilterSource, {
          aabbRejectedCount: aabbRejectedCount,
          componentRejectedCount: componentRejectedCount,
          keptByMissingMeta: keptByMissingMeta,
          rejectedPreview: rejectedPreview,
        });
    }

    return {
      modelList: filteredIds,
      weightList: filteredWeights,
    };
  }

  _getMaxModelId(idMode)
  {
    if (idMode === 'global-glb' || idMode === 'global-glb-priority')
    {
      return this.globalGlbEntries.length > 0 ? this.globalGlbEntries.length - 1 : -1;
    }

    return this.sceneConfig.groups[this.sceneConfig.groups.length - 1].idRange[1];
  }

  decodeGlobalGlbInfo(modelInfo)
  {
    if (modelInfo.id == null || this.globalGlbEntries.length == 0)
    {
      return null;
    }

    var entry = this.globalGlbEntries[modelInfo.id];
    if (entry == undefined)
    {
      return null;
    }

    var lodPath = this.MeshLodLevel >= 0 ? ('LOD' + this.MeshLodLevel) : 'raw';
    var glbBaseUrl = this.glbResourcesBaseUrl || this.resourcesBaseUrl;
    // Grouped indexes carry a path rooted at the scene GLB directory. Keep the
    // old task/base fallback for legacy indexes that do not provide it.
    var indexedPath = entry.path || ('task-' + entry.taskId + '/glb/' + lodPath + '/sub_' + entry.baseId + '.glb');
    var modelUrl = joinUrlPath(glbBaseUrl, indexedPath);
    var isToLoad = modelInfo.weight == null ? true : modelInfo.weight > 0;

    return {
      hash: entry.hash,
      url: isToLoad ? modelUrl : null,
      weight: modelInfo.weight,
      group: entry.taskId,
      globalGlbId: entry.globalId,
      baseId: entry.baseId,
      prefetch: Boolean(modelInfo.prefetch),
      deferredVisible: Boolean(modelInfo.deferredVisible),
      predictionEpoch: modelInfo.predictionEpoch != null ? Number(modelInfo.predictionEpoch) : null,
      priorityInfo: modelInfo.priorityInfo || null,
    };
  }
  decodeModelInfo(modelInfo)
  {
    var idMode = modelInfo.idMode || this.defaultVisibilityIdMode;

    if (idMode === 'global-glb' || idMode === 'global-glb-priority')
    {
      return this.decodeGlobalGlbInfo(modelInfo);
    }

    if (modelInfo.id == null)
    {
      return null;
    }

    var onlyShowInstanced = false;

    var matchedGroup = null;
    var matchedGroupId = 0;

    for (var i = 0; i < this.sceneConfig.groups.length; ++i)
    {
      if (modelInfo.id >= this.sceneConfig.groups[i].idRange[0] && 
        modelInfo.id <= this.sceneConfig.groups[i].idRange[1])
      {
        matchedGroup = this.sceneConfig.groups[i];
        matchedGroupId = i;
        break;
      }
    }

    var isToLoad = true;

    if (matchedGroup != null)
    {
      var baseId = modelInfo.id - matchedGroup.idRange[0];

      if (matchedGroup.instances[baseId] != undefined) // 检查是否有实例化对象
      {
        baseId = matchedGroup.instances[baseId];
      }
      else
      {
        if (onlyShowInstanced)
        {
          isToLoad = false;
        }
      }

      var hashCode = matchedGroupId + '-' + baseId;

      var lodPath = this.MeshLodLevel >= 0 ? ('LOD' + this.MeshLodLevel) : 'raw';

      var glbBaseUrl = this.glbResourcesBaseUrl || this.resourcesBaseUrl;
      var modelUrl = joinUrlPath(glbBaseUrl, "task-" + matchedGroupId + "/glb/" + lodPath + "/sub_" + baseId + ".glb");

      if (modelInfo.weight < 10)
      {
        isToLoad = false;
      }

      var decoded =
      {
        hash: hashCode,
        url: isToLoad ? modelUrl: null,
        weight: modelInfo.weight,
        group: matchedGroupId,
        prefetch: Boolean(modelInfo.prefetch),
        deferredVisible: Boolean(modelInfo.deferredVisible),
        predictionEpoch: modelInfo.predictionEpoch != null ? Number(modelInfo.predictionEpoch) : null,
      };

      return decoded;
    }

    return null;
  }
}



