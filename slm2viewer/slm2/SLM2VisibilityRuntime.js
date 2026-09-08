import { startupLog, stopStartupLog } from '../src/startupTimeline.js';
import { SLM2InstanceVisibilityState } from './SLM2InstanceVisibilityState.js';

export class SLM2VisibilityRuntime extends SLM2InstanceVisibilityState {
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

  _cloneVisibilityCamera()
  {
    var camera = this.activeCamera.clone();
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    return camera;
  }

  _schedulePendingNeuralVisibilityUpdate()
  {
    if (!this.neuralPendingVisibilityUpdate && !this.forceNextNeuralPrediction)
    {
      return;
    }
    this.neuralPendingVisibilityUpdate = false;
    setTimeout(() =>
    {
      if (this.useNeuralPVS && !this.frozenPredictionInspectActive)
      {
        this.sceneCulling();
      }
    }, 0);
  }

  _runCachedNeuralRenderFilter(predictionDispatcher, camera, groupSerial, lifecycleSerial)
  {
    var filterProcess = async () =>
    {
      this.neuralPredictionInFlight = true;
      var filterStartedAt = performance.now();
      try
      {
        var filtered = await predictionDispatcher.refilter(camera);
        if (!filtered || filtered.stale ||
            groupSerial !== this.neuralGroupSwitchSerial ||
            lifecycleSerial !== this.neuralPredictionLifecycleSerial ||
            this.frozenPredictionInspectActive)
        {
          return;
        }
        this.neuralPredictionGate.commitRender(camera);
        this.lastNeuralRefilter = filtered;
        var applied = this._applyLightweightNeuralRefilter(
          filtered,
          filtered.idMode || this.neuralPVSIdMode
        );
        var previousIds = this.lastBenchmarkVisibilityIds || {};
        this._recordVisibilityMetrics({
          mode: 'neural',
          idMode: filtered.idMode || this.neuralPVSIdMode,
          backend: filtered.backend,
          latencyMs: performance.now() - filterStartedAt,
          rawCount: previousIds.rawGlbIds ? previousIds.rawGlbIds.length : 0,
          visibleCount: applied.renderGlbCount,
          rawGlbCount: previousIds.rawGlbIds ? previousIds.rawGlbIds.length : 0,
          visibleGlbCount: applied.renderGlbCount,
          rawInstanceCount: previousIds.rawComponentIds ? previousIds.rawComponentIds.length : 0,
          visibleInstanceCount: applied.renderComponentCount,
          notes: {
            cachedModelPrediction: true,
            renderCandidateGlbCount: applied.renderGlbCount,
            renderResidentCount: this.renderVisibilitySystem.visibleHashes.size,
          },
        });
      }
      finally
      {
        this.neuralPredictionInFlight = false;
        this._schedulePendingNeuralVisibilityUpdate();
      }
    };
    filterProcess();
  }

  sceneCulling()
  {
    var logThisCulling = this.startupSceneCullingLogCount < 20;
    if (logThisCulling)
    {
      this.startupSceneCullingLogCount += 1;
      startupLog('slm2:sceneCulling:enter', {
        useNeuralPVS: this.useNeuralPVS,
        fullLoadMode: this.fullLoadMode,
        sceneInitialized: this.isSceneInitialized,
        neuralReady: Boolean(this.neuralPVS && this.neuralPVS.isReady),
      });
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

    var posHashResolution = 1;
    var rotHashResolution = 5;
    var cameraPosHash = (this.activeCamera.position.x * posHashResolution).toFixed(0) + "-" + 
                        (this.activeCamera.position.y * posHashResolution).toFixed(0) + "-" + 
                        (this.activeCamera.position.z * posHashResolution).toFixed(0);
    var cameraRotHash = (this.activeCamera.rotation.x * rotHashResolution).toFixed(0) + "-" + 
                        (this.activeCamera.rotation.y * rotHashResolution).toFixed(0) + "-" +
                        (this.activeCamera.rotation.z * rotHashResolution).toFixed(0);

    var screenSizeHash = this.clientWidth + "-" + this.clientHeight;

    if (this.useNeuralPVS ||
        cameraPosHash != this.lastCameraPosHash ||
        cameraRotHash != this.lastCameraRotHash ||
        screenSizeHash != this.lastScreenSizeHash)
    {
      
      this.lastCameraPosHash = cameraPosHash;
      this.lastCameraRotHash = cameraRotHash;
      this.lastScreenSizeHash = screenSizeHash;

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
        // Run one local prediction for the current view-cell anchor.
        if (this.neuralPVS && this.neuralPVS.isReady) {
          if (logThisCulling) startupLog('slm2:sceneCulling:neural-ready');
          this.neuralPVSInitWarningShown = false;
          var forceNeuralPrediction = Boolean(this.forceNextNeuralPrediction);
          var needsFullPrediction = forceNeuralPrediction || this.neuralPredictionGate.shouldPredict(this.activeCamera);
          var needsRenderRefilter = !needsFullPrediction && this.neuralPredictionGate.shouldRefilter(this.activeCamera);
          if (this.neuralPredictionInFlight)
          {
            if (needsFullPrediction || needsRenderRefilter)
            {
              this.neuralPendingVisibilityUpdate = true;
            }
            return;
          }
          if (!needsFullPrediction && !needsRenderRefilter)
          {
            return;
          }
          const predictionDispatcher = this.neuralPVS;
          const predictionGroupSerial = this.neuralGroupSwitchSerial;
          const predictionLifecycleSerial = this.neuralPredictionLifecycleSerial;
          const visibilityCamera = this._cloneVisibilityCamera();
          if (needsRenderRefilter)
          {
            this._runCachedNeuralRenderFilter(
              predictionDispatcher,
              visibilityCamera,
              predictionGroupSerial,
              predictionLifecycleSerial
            );
            return;
          }
          this.forceNextNeuralPrediction = false;
          const predictionProcess = async () => {
            this.neuralPredictionInFlight = true;
            const predictionStart = performance.now();
            var isFirstStartupPrediction = this.startupMetrics.firstPredictionMs <= 0;
            try {
              startupLog('slm2:neural:predict-start');
              var rawPredictionStart = performance.now();
              const pred = await predictionDispatcher.predict(visibilityCamera);
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
                this.neuralPredictionGate.commit(visibilityCamera);
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
              this._schedulePendingNeuralVisibilityUpdate();
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
            console.warn('[NeuralPVS] Model failed to initialize; neural culling is unavailable.',
              loadError ? `Last load error: ${loadError}` : '');
            this.neuralPVSInitFallbackWarningShown = true;
          }
          return;
        }
      }
    }
  }
}
