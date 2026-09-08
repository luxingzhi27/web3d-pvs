import { FileLoader, Object3D, PerspectiveCamera, Vector2 } from 'three';
import { MODEL_INPUT_FOV_Y_DEG } from '../src/neuralPvsFovProtocol.js';
import { getInstancePVSAssetBaseUrl } from '../src/neuralCullingBackendMode.js';
import { LightweightPVSDispatcher } from '../src/LightweightPVSDispatcher.js';
import { startupLog } from '../src/startupTimeline.js';
import { INITIAL_GLB_PRELOAD_LIMIT, LOCAL_RUNTIME_ASSET_VERSION, joinUrlPath, withLocalRuntimeVersion } from './SLM2RuntimeAssets.js';
import { SLM2VisibilityIndex } from './SLM2VisibilityIndex.js';

export class SLM2SceneCatalog extends SLM2VisibilityIndex {
  setSize(clientWidth, clientHeight)
  {
    this.clientWidth = clientWidth;
    this.clientHeight = clientHeight;

    this.backCamera.aspect = clientWidth / clientHeight;
    this.backCamera.updateProjectionMatrix();

    this.screenPixelReciprocal = 1.0 / (this.clientWidth * this.clientHeight);
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
    this.initialGlbLoadOrderUrl = baseConfig.loader.initialGlbLoadOrderUrl || null;
    var hasConfiguredGlbResourcesBaseUrl = Boolean(baseConfig.loader.glbResourcesBaseUrl || baseConfig.loader.modelResourcesBaseUrl);
    this.schedulingStrategy = baseConfig.loader.schedulingStrategy || 'auto';
    this.options = options || {};
    var params = this.options.paramJson || {};
    var cullingParam = params['culling'] == null ? 'neural' : String(params['culling']);
    var cullingParamLower = cullingParam.toLowerCase();
    var cullingMode = cullingParamLower;
    if (cullingMode === 'aabb')
    {
      cullingMode = 'frustum';
    }
    if (cullingMode !== 'neural' && cullingMode !== 'frustum' && cullingMode !== 'false')
    {
      console.warn('[SLM2Loader] Unknown culling mode "' + cullingParam + '"; expected "neural", "frustum", or "false". Falling back to neural.');
      cullingMode = 'neural';
    }
    this.cullingMode = cullingMode;
    this.useNeuralPVS = this.cullingMode === 'neural';
    this.fullLoadMode = this.cullingMode === 'false';
    this.hasFullLoaded = false;
    this.neuralPredictionInFlight = false;
    this.neuralPendingVisibilityUpdate = false;
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
    });
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
    this.cpuPerfMode = params['cpuPerfMode'] === 'mobile' ? 'mobile' : 'balanced';
    this._configureHttpAdaptiveConcurrency();
    this._configureGlbResourcePipelineOptions(params);
    this.setNeuralRenderPolicy(params['neuralRenderPolicy'] || params['neuralAlwaysRender']);
    if (this.neuralDebugLogs) console.log('[SLM2Loader] CPU perf mode:', this.cpuPerfMode);
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
      backendPreference: params['neuralRuntimeBackend'] || 'auto',
      cpuPerfMode: this.cpuPerfMode,
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
          scope._loadInitialGlbLoadOrder(function()
          {
            scope._startNeuralInitialGlbPreload();
            scope._scheduleRuntimeVisibilityMetaLoad(function()
            {
              scope._scheduleNeuralPVSInit();
            });
          });
        }
        else if (scope.cullingMode === 'frustum')
        {
          scope._scheduleRuntimeVisibilityMetaLoad(function()
          {
            scope.forceSceneCullingNow({ forceNeural: false });
          });
        }

        if (scope.cullingMode === 'frustum')
        {
          if (scope.neuralDebugLogs) console.log('[SLM2Loader] culling=frustum; using local runtime AABB filtering.');
        }
        else if (scope.neuralDebugLogs)
        {
          console.log('[SLM2Loader] GLB resources use the HTTP download/parse/mount pipeline.');
        }

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

  _loadInitialGlbLoadOrder(callback)
  {
    var scope = this;
    var fileLoader = new FileLoader();
    var startedAt = performance.now();
    var requestUrl = this.initialGlbLoadOrderUrl ||
      (this.resourcesBaseUrl + "/initialGlbLoadOrder.json");
    startupLog('slm2:initialGlbOrder:request', { url: requestUrl });

    var finish = function()
    {
      scope.startupMetrics.initialGlbLoadOrderMs = performance.now() - startedAt;
      if (callback) callback();
    };
    fileLoader.load(requestUrl, function(data)
    {
      try
      {
        var parsed = JSON.parse(data);
        var ids = Array.isArray(parsed && parsed.initialLoadIds)
          ? parsed.initialLoadIds
          : [];
        var seen = new Set();
        scope.neuralInitialLoadOrder = ids.map(Number).filter(function(globalId)
        {
          if (!Number.isInteger(globalId) || globalId < 0 || seen.has(globalId) ||
              !scope.globalGlbEntries[globalId])
          {
            return false;
          }
          seen.add(globalId);
          return true;
        }).slice(0, INITIAL_GLB_PRELOAD_LIMIT);
        scope.neuralInitialLoadPending = scope.neuralInitialLoadOrder.length > 0;
        startupLog('slm2:initialGlbOrder:loaded', {
          count: scope.neuralInitialLoadOrder.length,
        });
      }
      catch (error)
      {
        scope.neuralInitialLoadOrder = [];
        scope.neuralInitialLoadPending = false;
        console.warn('[SLM2Loader] Failed to parse initialGlbLoadOrder.json.', error);
      }
      finish();
    }, null, function()
    {
      scope.neuralInitialLoadOrder = [];
      scope.neuralInitialLoadPending = false;
      startupLog('slm2:initialGlbOrder:missing', { url: requestUrl });
      finish();
    });
  }

  _startNeuralInitialGlbPreload()
  {
    if (!this.useNeuralPVS)
    {
      return 0;
    }
    var infos = this._takeNeuralInitialLoadInfos('global-glb');
    this.glbResourceScheduler.enqueueStartup(
      this._makeGlbScheduleEntries(infos, 'startup')
    );
    if (infos.length > 0)
    {
      // Start GLB requests first. The following runtime-meta callback starts
      // the model runtime fetch in parallel without waiting for GLB completion.
      this.processLoadingList();
    }
    this.startupMetrics.initialGlbPreloadCount = infos.length;
    startupLog('slm2:initialGlbPreload:started', { count: infos.length });
    return infos.length;
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
    var loadOrder = this.neuralInitialLoadOrder.slice(0, INITIAL_GLB_PRELOAD_LIMIT);
    for (var index = 0; index < loadOrder.length; ++index)
    {
      var globalId = Number(loadOrder[index]);
      if (!Number.isFinite(globalId) || seen.has(globalId) || !this.globalGlbEntries[globalId])
      {
        continue;
      }
      seen.add(globalId);
      infos.push({
        id: globalId,
        weight: Math.max(0.000001, 1 - index / Math.max(1, loadOrder.length)),
        idMode: idMode || 'global-glb',
        initialPreload: true,
        prefetch: true,
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
    var initialGlbLoadOrderUrl = group.initialGlbLoadOrderUrl || null;
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
    this.initialGlbLoadOrderUrl = initialGlbLoadOrderUrl;
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
    this.neuralPendingVisibilityUpdate = false;
    this.lastNeuralPrediction = null;
    this.lastNeuralRefilter = null;
    this.resourcePipelineSerial++;
    this.modelToLoadList = [];
    this.glbResourceScheduler.reset();
    this._cancelAllDirectDownloads();
    this._clearPendingGlbParseQueue();
    this.pendingSceneInsertions = [];
    this.pendingSceneInsertionCursor = 0;
    this.pendingSceneInsertionHashes = new Set();
    this.neuralDownloadInfoByGlbId = new Map();
    this.inflightModelHashes = new Set();
    this.directLoadRetries.clear();
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
    this.renderComponentState.reset(this._getComponentBitCount());
    this.renderGlbState.reset(this._getGlbBitCount());
    this.instancedWantedIndicesByHash = new Map();
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
      initialGlbLoadOrderUrl
        ? this._fetchNeuralGroupJson(initialGlbLoadOrderUrl)
        : Promise.resolve({ initialLoadIds: [] }),
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
      scope.neuralInitialLoadOrder = Array.isArray(payload[2] && payload[2].initialLoadIds)
        ? payload[2].initialLoadIds.slice()
        : [];
      scope.neuralInitialLoadPending = scope.neuralInitialLoadOrder.length > 0;
      scope._startNeuralInitialGlbPreload();
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

}
