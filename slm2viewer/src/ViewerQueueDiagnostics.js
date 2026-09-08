export const FRONTEND_RUNTIME_ASSET_ESTIMATE = Object.freeze({
  label: 'PVS V4 固定实例特征与查询网络，约 5.48 MB，不含按需 GLB',
});

export class ViewerQueueDiagnostics {
  createQueueDebugPanel()
  {
    const panel = document.createElement('div');
    panel.className = 'queue-debug-panel';
    panel.style.display = this.queueDebugEnabled ? 'block' : 'none';

    const header = document.createElement('div');
    header.className = 'queue-debug-header';
    header.innerHTML = '<span>Neural Queues</span>';

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.textContent = 'hide';
    toggle.addEventListener('click', () => {
      this.queueDebugEnabled = !this.queueDebugEnabled;
      panel.style.display = this.queueDebugEnabled ? 'block' : 'none';
      toggle.textContent = this.queueDebugEnabled ? 'hide' : 'show';
      this.requestRender('queue-panel-toggle');
    });
    header.appendChild(toggle);

    const redToggle = document.createElement('button');
    redToggle.type = 'button';
    redToggle.textContent = this.predictionDebugEnabled ? 'red on' : 'red off';
    redToggle.title = 'Toggle red highlight for instances/components predicted visible by the current neural result.';
    redToggle.addEventListener('click', () => {
      this.setPredictionDebugEnabled(!this.predictionDebugEnabled);
    });
    header.appendChild(redToggle);
    this.predictionDebugToggleButton = redToggle;

    const freezeButton = document.createElement('button');
    freezeButton.type = 'button';
    freezeButton.textContent = 'freeze view';
    freezeButton.title = 'Freeze the final instance set visible in the current 60-degree render frustum.';
    freezeButton.addEventListener('click', () => {
      this.toggleFrozenPredictionInspectMode();
    });
    header.appendChild(freezeButton);
    this.predictionDebugFreezeButton = freezeButton;

    const shotButton = document.createElement('button');
    shotButton.type = 'button';
    shotButton.textContent = 'top png';
    shotButton.title = 'Render one top-down PNG with predicted visible instances highlighted red.';
    shotButton.addEventListener('click', () => {
      this.capturePredictionDebugTopDown();
    });
    header.appendChild(shotButton);

    const content = document.createElement('pre');
    content.className = 'queue-debug-content';
    content.textContent = 'waiting for runtime stats...';

    panel.appendChild(header);
    panel.appendChild(content);
    this.el.appendChild(panel);
    this.queueDebugPanel = panel;
    this.queueDebugContent = content;

    const miniButton = document.createElement('button');
    miniButton.type = 'button';
    miniButton.className = 'queue-debug-mini-toggle';
    miniButton.textContent = 'Queues';
    miniButton.addEventListener('click', () => {
      this.queueDebugEnabled = !this.queueDebugEnabled;
      panel.style.display = this.queueDebugEnabled ? 'block' : 'none';
      toggle.textContent = this.queueDebugEnabled ? 'hide' : 'show';
      this.requestRender('queue-panel-toggle');
    });
    this.el.appendChild(miniButton);
  }

  updateQueueDebugPanel(time)
  {
    if (!this.queueDebugEnabled || !this.queueDebugContent || !this.slm2Loader)
    {
      return;
    }

    if (time - this.queueDebugLastUpdate < this.queueDebugIntervalMs)
    {
      return;
    }
    this.queueDebugLastUpdate = time;

    const stats = this.slm2Loader.getRuntimeStats();
    const startup = stats.startup || {};
    const visibility = stats.visibility || {};
    const load = stats.load || {};
    const cache = stats.cache || {};
    const neural = stats.neural || {};
    const render = neural.renderVisibility || {};
    const staticBatching = neural.staticBatching || {};
    const actualRender = neural.actualRender || {};
    const scheduler = neural.priorityScheduler || {};
    const candidateSelection = neural.predictTimings?.candidateSelection || scheduler.candidateSelection || {};
    const frozenInspect = neural.frozenInspect || {};
    const httpAdaptive = load.httpAdaptive || {};
    const initTimings = neural.initTimings || {};
    const pvsInitTimings = initTimings.pvs || initTimings;
    const webgpuAdapter = pvsInitTimings.webgpu?.adapter || {};
    const webgpuAdapterLabel = [
      webgpuAdapter.vendor,
      webgpuAdapter.architecture,
      webgpuAdapter.description,
    ].filter(Boolean).join(' / ') || '-';
    const gate = neural.predictionGate || {};
    const modelInfo = neural.modelInfo || (neural.predictTimings ? neural.predictTimings.modelInfo : null) || {};
    const workpoint = modelInfo.calibrationWorkpoint || {};
    const notes = visibility.notes || {};
    const predictionAge = visibility.timestamp ? Math.max(0, performance.now() - visibility.timestamp) : null;
    const predictDebug = this.predictionDebugLastStats || {};
    const fmt = (value, digits = 1) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '-';
    const ids = (items) => (items || []).map((item) => {
      const id = item && item.id !== undefined ? item.id : '?';
      const weight = item && item.weight !== undefined ? fmt(item.weight, 3) : '-';
      return `${id}:${weight}`;
    }).join(', ');

    this.queueDebugContent.textContent = [
      `model=${modelInfo.runtimeModelDisplayName || modelInfo.runtimeModelName || '-'} schema=${modelInfo.runtimeSchema || '-'} threshold=${fmt(modelInfo.visibilityThreshold, 3)} selection=${modelInfo.thresholdSelection || '-'} priority=${modelInfo.outputsDownloadPriority === true}`,
      `calibration weighted=${fmt(workpoint.aggregateWeightedRecall, 4)} lower=${fmt(workpoint.aggregateWeightedRecallLowerConfidenceBound, 4)} poseWeighted=${fmt(workpoint.poseWeightedRecall, 4)} testReads=${workpoint.testEvaluationCount || 0}`,
      `culling=${neural.cullingMode || '-'} mode=${visibility.mode || '-'} idMode=${neural.idMode || visibility.idMode || '-'} backend=${neural.backend || '-'} ready=${neural.ready}`,
      `neural=${neural.enabled} transport=http renderPolicy=${neural.renderPolicy} cpu=${neural.cpuPerfMode}`,
      `httpAdaptive enabled=${httpAdaptive.enabled !== false} now=${httpAdaptive.current || '-'} prefetch=${httpAdaptive.prefetchCurrent || '-'} mbps=${fmt(httpAdaptive.ewmaMbps, 1)} downlink=${fmt(httpAdaptive.downlinkMbps, 1)} load=${fmt(httpAdaptive.ewmaLoadMs, 0)}ms reason=${httpAdaptive.lastReason || '-'}`,
      `renderSurface pixelRatio=${fmt(this.renderSurface?.pixelRatio, 2)} buffer=${this.renderSurface?.drawingBufferWidth || '-'}x${this.renderSurface?.drawingBufferHeight || '-'}`,
      `startup sceneWeb=${fmt(startup.sceneWebMs, 0)}ms glbIndex=${fmt(startup.glbIndexMs, 0)} runtimeMeta=${fmt(startup.runtimeVisibilityMetaMs, 0)} material=${fmt(startup.materialConfigMs, 0)}ms`,
      `neuralInit total=${fmt(initTimings.totalMs, 0)}ms pvs=${fmt(pvsInitTimings.totalMs, 0)}ms fetch=${fmt(pvsInitTimings.fetchMs, 0)} gpu=${fmt(pvsInitTimings.gpuInitMs, 0)}ms adapter=${webgpuAdapterLabel}`,
      `gate pos=${fmt(gate.positionDelta, 1)} angle=${fmt(gate.angleDeltaDeg, 1)}deg min=${fmt(gate.minIntervalMs, 0)}ms forceNext=${gate.forceNext}`,
      `prediction serial=${visibility.serial || '-'} age=${predictionAge == null ? '-' : fmt(predictionAge, 0) + 'ms'} latency=${fmt(visibility.latencyMs)}ms`,
      `candidate source=${candidateSelection.source || '-'} count=${candidateSelection.candidateCount ?? '-'} cells=${candidateSelection.queryCellCount ?? '-'} indexed=${candidateSelection.indexedInstanceCount ?? '-'} overflow=${candidateSelection.overflowInstanceCount ?? '-'}`,
      `modelBackGlb=${visibility.rawGlbCount || 0} loadNow=${notes.loadNowGlbCount != null ? notes.loadNowGlbCount : '-'} currentFrustumGlb=${notes.renderCandidateGlbCount != null ? notes.renderCandidateGlbCount : '-'} residentVisible=${notes.renderResidentCount != null ? notes.renderResidentCount : (visibility.visibleGlbCount != null ? visibility.visibleGlbCount : '-')} modelBackInst=${visibility.rawInstanceCount || 0}`,
      `download queue=${load.queueLength || 0} wanted=${load.wantedHashCount || 0} inflight=${load.inflightCount || 0} pendingParse=${load.pendingHashCount || 0} pendingIntegrate=${load.pendingSceneInsertions || 0}`,
      `activeQueues immediate=${load.queueLength || 0} urgentMissing=${load.urgentMissingCount || 0} urgentFailed=${load.urgentFailedCount || 0} prefetch=${load.prefetchQueueLength || 0} http=${load.activeDirectLoadCount || 0} parse=${load.pendingParseCount || 0} mount=${load.pendingSceneInsertions || 0} integrated=${load.totalIntegrated || 0}`,
      `cache total=${cache.totalNums || 0} visible=${cache.visibleNums || 0} invisible=${cache.invisibleNums || 0} inSceneHidden=${cache.inSceneNums || 0} mem=${fmt((cache.memoryUsedKB || 0) / 1024, 0)}MB`,
      `render working=${render.workingSetSize || 0} activeEval=${render.activeEvaluationSize || 0} evaluated=${render.evaluatedCount || 0} visible=${render.visibleCount || 0} skipped=${render.skippedByGate}`,
      `actual visibleMesh=${actualRender.visibleMeshCount || 0}/${actualRender.meshCount || 0} instancedMesh=${actualRender.visibleInstancedMeshCount || 0}/${actualRender.instancedMeshCount || 0} drawnInst=${actualRender.drawnInstanceCount || 0} tri≈${fmt(actualRender.visibleTriangleEstimate, 0)}`,
      `staticBatch glb=${staticBatching.batchedGlbs || 0} batches=${staticBatching.batchCount || 0} pending=${staticBatching.pendingGlbs || 0} drawSaved=${staticBatching.drawCallReduction || 0} flattened=${staticBatching.flattenedGlbs || 0}`,
      `render delta=+${render.addedCount || 0}/-${render.removedCount || 0} evaluated=${render.evaluatedCount || 0} attach=${render.attachedCount || 0} detach=${render.detachedCount || 0} renderMs=${fmt(render.durationMs, 3)}`,
      `modelPlan total=${scheduler.total || 0} immediate=${scheduler.visibleNow || 0} prefetchTotal=${scheduler.prefetch || 0} skipped=${scheduler.skipped || 0} tested=${scheduler.testedComponents || 0}`,
      `predictRed enabled=${predictDebug.enabled || false} frozen=${predictDebug.frozen || false} predictedComp=${predictDebug.predicted || 0} loadedHash=${predictDebug.loaded || 0} markedInst=${predictDebug.marked || 0} missing=${predictDebug.missing || 0} attached=${predictDebug.attached || 0} restored=${predictDebug.restored || 0} shot=${predictDebug.snapshot || '-'}`,
      `freezeInspect active=${frozenInspect.active || false} queued=${frozenInspect.queued || false} glb=${frozenInspect.total || 0} inst=${frozenInspect.componentCount || 0}`,
      `nextLoad: ${ids(load.queuePreview) || '-'}`,
      `prefetch: ${ids(load.prefetchPreview) || '-'}`,
    ].join('\n');
  }

  _formatRuntimeDebugNumber(value, digits = 1, suffix = '')
  {
    const number = Number(value);
    return Number.isFinite(number) ? `${number.toFixed(digits)}${suffix}` : '-';
  }

  _formatRuntimeDebugInt(value)
  {
    const number = Number(value);
    return Number.isFinite(number) ? `${Math.round(number)}` : '0';
  }

  _describeRuntimeCullingSource(stats)
  {
    const neural = stats && stats.neural ? stats.neural : {};
    const visibility = stats && stats.visibility ? stats.visibility : {};
    const predictionBackend = String(
      (neural.predictTimings && neural.predictTimings.backend) ||
      visibility.backend ||
      ''
    );
    const currentBackend = String(neural.backend || neural.neuralBackend || '');
    const backend = predictionBackend || currentBackend;
    const fallbackReason = neural.predictTimings && neural.predictTimings.fallbackReason
      ? neural.predictTimings.fallbackReason
      : (neural.initTimings && neural.initTimings.fallbackReason ? neural.initTimings.fallbackReason : '');

    if (neural.cullingMode === 'frustum')
    {
      return '本地实例 AABB 视锥剔除';
    }

    if (!neural.enabled)
    {
      return 'AABB/普通剔除（神经关闭）';
    }

    if (!neural.ready)
    {
      return '模型未就绪';
    }

    if (!predictionBackend && currentBackend.indexOf('webgpu') >= 0)
    {
      return '模型已就绪，等待首次预测';
    }

    if (backend.indexOf('webgpu') >= 0)
    {
      return '模型剔除（Worker WebGPU）';
    }

    if (backend.indexOf('wasm-simd') >= 0)
    {
      return fallbackReason
        ? '模型剔除（Worker WASM SIMD 兼容后端）'
        : '模型剔除（Worker WASM SIMD）';
    }

    if (backend.indexOf('aabb') >= 0)
    {
      return 'AABB 剔除（Worker 降级）';
    }

    return backend ? `Worker 剔除（${backend}）` : '等待首次预测';
  }

  _describeRuntimeBandwidth(load, neural)
  {
    const httpAdaptive = load && load.httpAdaptive ? load.httpAdaptive : {};
    const resourceWS = neural && neural.resourceWS ? neural.resourceWS : {};
    const httpEwma = Number(httpAdaptive.ewmaMbps || 0);
    const httpLast = Number(httpAdaptive.lastMbps || 0);
    const downlink = Number(httpAdaptive.downlinkMbps || 0);
    const wsMBps = Number(resourceWS.throughputMBps || 0);
    const samples = Number(httpAdaptive.samples || 0);
    const lastBytes = Number(httpAdaptive.lastBytes || 0);
    const reason = httpAdaptive.lastReason || '-';

    if (httpEwma > 0 || httpLast > 0)
    {
      return `HTTP实测 ${this._formatRuntimeDebugNumber(httpEwma || httpLast, 1, 'Mbps')}，最近 ${this._formatRuntimeDebugNumber(httpLast, 1, 'Mbps')}，样本 ${samples}，原因 ${reason}`;
    }

    if (wsMBps > 0)
    {
      return `WS实测 ${this._formatRuntimeDebugNumber(wsMBps, 2, 'MB/s')}（约 ${this._formatRuntimeDebugNumber(wsMBps * 8, 1, 'Mbps')}），HTTP样本 ${samples}`;
    }

    if (downlink > 0)
    {
      return `浏览器估计 ${this._formatRuntimeDebugNumber(downlink, 1, 'Mbps')}，暂无HTTP字节样本，原因 ${reason}`;
    }

    if (samples > 0)
    {
      return `暂无可用字节样本，已完成 ${samples} 次下载，最近字节 ${Math.round(lastBytes)}，可能是缓存或跨域Timing受限`;
    }

    return '等待GLB下载样本';
  }

  _describeRuntimeConcurrency(load)
  {
    const httpAdaptive = load && load.httpAdaptive ? load.httpAdaptive : {};
    const current = httpAdaptive.current || '-';
    const max = httpAdaptive.maxConcurrency || '-';
    const prefetch = httpAdaptive.prefetchCurrent || '-';
    const prefetchMax = httpAdaptive.prefetchMaxConcurrency || '-';
    const reason = httpAdaptive.lastReason || '-';
    const samples = httpAdaptive.samples || 0;
    return `HTTP ${current}/${max}，预取 ${prefetch}/${prefetchMax}，原因 ${reason}，样本 ${samples}`;
  }

  updateRuntimeDebugGui(time, force = false)
  {
    if (!this.slm2Loader)
    {
      return;
    }

    if (!force && time - this.runtimeDebugGuiLastUpdate < this.runtimeDebugGuiIntervalMs)
    {
      return;
    }
    this.runtimeDebugGuiLastUpdate = time;

    const stats = this.slm2Loader.getRuntimeStats();
    const visibility = stats.visibility || {};
    const load = stats.load || {};
    const cache = stats.cache || {};
    const neural = stats.neural || {};
    const render = neural.renderVisibility || {};
    const actualRender = neural.actualRender || {};
    const httpAdaptive = load.httpAdaptive || {};
    const predictTimings = neural.predictTimings || {};
    const filterTimings = neural.filterTimings || {};
    const currentRenderTimings = Number(filterTimings.serial || 0) > Number(predictTimings.serial || 0)
      ? filterTimings
      : predictTimings;
    const scheduler = neural.priorityScheduler || {};
    const initTimings = neural.initTimings || {};
    const pvsInitTimings = initTimings.pvs || initTimings;
    const webgpuAdapter = pvsInitTimings.webgpu?.adapter || {};
    const webgpuAdapterLabel = [
      webgpuAdapter.vendor,
      webgpuAdapter.architecture,
      webgpuAdapter.description,
    ].filter(Boolean).join(' / ') || '-';
    const gate = neural.predictionGate || {};
    const modelInfo = neural.modelInfo || predictTimings.modelInfo || {};
    const workpoint = modelInfo.calibrationWorkpoint || {};
    const notes = visibility.notes || {};
    const predictDebug = this.predictionDebugLastStats || {};
    const predictionAge = visibility.timestamp ? Math.max(0, performance.now() - visibility.timestamp) : null;
    const candidateSelection = predictTimings.candidateSelection || scheduler.candidateSelection || {};

    this.runtimeDebugState.frontendAssetEstimate = FRONTEND_RUNTIME_ASSET_ESTIMATE.label;
    this.runtimeDebugState.cullingMode = neural.cullingMode || this.slm2Loader.getCullingMode();
    this.runtimeDebugState.pvsModelVersion = modelInfo.runtimeModelDisplayName || modelInfo.runtimeModelName || '-';
    this.runtimeDebugState.pvsModelSchema = `${modelInfo.runtimeSchema || '-'} / threshold ${this._formatRuntimeDebugNumber(modelInfo.visibilityThreshold, 3)} / ${modelInfo.thresholdSelection || '-'}`;
    this.runtimeDebugState.pvsModelWorkpoint = `weighted ${this._formatRuntimeDebugNumber(workpoint.aggregateWeightedRecall, 4)}, lower ${this._formatRuntimeDebugNumber(workpoint.aggregateWeightedRecallLowerConfidenceBound, 4)}, poseWeighted ${this._formatRuntimeDebugNumber(workpoint.poseWeightedRecall, 4)}, testReads ${workpoint.testEvaluationCount || 0}`;
    this.runtimeDebugState.pvsCullingSource = this._describeRuntimeCullingSource(stats);
    this.runtimeDebugState.pvsBackend = `${visibility.backend || neural.backend || neural.neuralBackend || '-'} / adapter ${webgpuAdapterLabel}`;
    this.runtimeDebugState.pvsReady = neural.ready ? '是' : '否';
    this.runtimeDebugState.pvsFallbackReason = predictTimings.fallbackReason || initTimings.fallbackReason || '-';
    this.runtimeDebugState.pvsPredictMs = this._formatRuntimeDebugNumber(
      predictTimings.totalMs != null ? predictTimings.totalMs : visibility.latencyMs,
      1,
      'ms'
    );
    this.runtimeDebugState.pvsInferenceMs = this._formatRuntimeDebugNumber(predictTimings.inferenceMs, 1, 'ms');
    this.runtimeDebugState.pvsPostMs = this._formatRuntimeDebugNumber(predictTimings.postMs, 1, 'ms');
    this.runtimeDebugState.pvsPredictionAgeMs = predictionAge == null ? '-' : this._formatRuntimeDebugNumber(predictionAge, 0, 'ms');
    this.runtimeDebugState.pvsCandidateSelection = `${candidateSelection.source || '-'} / ${this._formatRuntimeDebugInt(candidateSelection.candidateCount)} candidates / ${this._formatRuntimeDebugInt(candidateSelection.queryCellCount)} cells / ${this._formatRuntimeDebugInt(candidateSelection.overflowInstanceCount)} overflow`;
    this.runtimeDebugState.pvsGate = `pos ${this._formatRuntimeDebugNumber(gate.positionDelta, 1, 'm')} / yaw ${this._formatRuntimeDebugNumber(gate.yawDeltaDeg || gate.angleDeltaDeg, 1, '°')} / min ${this._formatRuntimeDebugNumber(gate.minIntervalMs, 0, 'ms')}`;
    this.runtimeDebugState.pvsRawInstances = this._formatRuntimeDebugInt(
      predictTimings.rawInstanceCount != null ? predictTimings.rawInstanceCount : visibility.rawInstanceCount
    );
    this.runtimeDebugState.pvsRawGlbs = this._formatRuntimeDebugInt(
      predictTimings.rawGlbCount != null ? predictTimings.rawGlbCount : visibility.rawGlbCount
    );
    this.runtimeDebugState.pvsImmediateGlbs = this._formatRuntimeDebugInt(
      load.urgentWantedCount != null
        ? load.urgentWantedCount
        : (predictTimings.urgentGlbCount != null ? predictTimings.urgentGlbCount : notes.loadNowGlbCount)
    );
    this.runtimeDebugState.pvsPrefetchGlbs = this._formatRuntimeDebugInt(
      predictTimings.prefetchGlbCount != null ? predictTimings.prefetchGlbCount : load.prefetchQueueLength
    );
    this.runtimeDebugState.pvsRenderInstances = this._formatRuntimeDebugInt(
      currentRenderTimings.renderInstanceCount != null
        ? currentRenderTimings.renderInstanceCount
        : visibility.visibleInstanceCount
    );
    this.runtimeDebugState.pvsRenderGlbs = this._formatRuntimeDebugInt(
      currentRenderTimings.renderGlbCount != null
        ? currentRenderTimings.renderGlbCount
        : notes.renderCandidateGlbCount
    );
    this.runtimeDebugState.pvsDownloadQueue = this._formatRuntimeDebugInt(load.queueLength);
    this.runtimeDebugState.pvsPrefetchQueue = this._formatRuntimeDebugInt(load.prefetchQueueLength);
    this.runtimeDebugState.pvsActiveLoads = this._formatRuntimeDebugInt(load.activeDirectLoadCount);
    this.runtimeDebugState.pvsInflightLoads = this._formatRuntimeDebugInt(load.inflightCount);
    this.runtimeDebugState.pvsPendingIntegrate = this._formatRuntimeDebugInt(load.pendingSceneInsertions);
    this.runtimeDebugState.pvsHttpMbps = this._describeRuntimeBandwidth(load, neural);
    this.runtimeDebugState.pvsHttpConcurrency = this._describeRuntimeConcurrency(load);
    this.runtimeDebugState.pvsCacheSummary = `total ${cache.totalNums || 0}, visible ${cache.visibleNums || 0}, hidden ${cache.invisibleNums || 0}`;
    this.runtimeDebugState.pvsRenderSummary = `work ${render.workingSetSize || 0}, resident ${render.visibleCount || 0}, delta +${render.addedCount || 0}/-${render.removedCount || 0}`;
    this.runtimeDebugState.pvsActualRender = `mesh ${actualRender.visibleMeshCount || 0}/${actualRender.meshCount || 0}, instancedMesh ${actualRender.visibleInstancedMeshCount || 0}/${actualRender.instancedMeshCount || 0}, drawnInst ${actualRender.drawnInstanceCount || 0}, tri≈${this._formatRuntimeDebugInt(actualRender.visibleTriangleEstimate)}`;
    this.runtimeDebugState.pvsPredictionDebugSummary = `red ${predictDebug.enabled ? 'on' : 'off'}, frozen ${predictDebug.frozen ? 'on' : 'off'}, marked ${predictDebug.marked || 0}, missing ${predictDebug.missing || 0}`;

    for (let i = 0; i < this.runtimeDebugControllers.length; ++i)
    {
      if (this.runtimeDebugControllers[i] && typeof this.runtimeDebugControllers[i].updateDisplay === 'function')
      {
        this.runtimeDebugControllers[i].updateDisplay();
      }
    }

    const runtimeEls = this.runtimeDebugValueEls || {};
    Object.keys(runtimeEls).forEach((property) => {
      const el = runtimeEls[property];
      if (!el) return;
      const value = this.runtimeDebugState[property] != null ? String(this.runtimeDebugState[property]) : '-';
      el.textContent = value;
      el.title = value;
    });
  }
}

