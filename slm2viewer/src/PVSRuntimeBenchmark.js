import { PerspectiveCamera, Quaternion, Vector3 } from 'three';
import { InstancePVSRuntime } from './InstancePVSRuntime.js';

const SCENE_MANIFEST_URL = './assets/benchmark/pvs_paper_runtime/scenes.json';
const RESULT_SCHEMA = 'pvs-v4-browser-runtime-result-v1';
const SOFTWARE_ADAPTER_TOKENS = ['swiftshader', 'llvmpipe', 'softpipe', 'swrast', 'software'];
const FORWARD_AXIS = new Vector3(0, 0, -1);

function nowMs() {
  return performance.now();
}

function percentile(values, quantile) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * quantile;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function summarizeSamples(samples) {
  const inference = samples.map((sample) => sample.modelInferenceMs);
  const candidates = samples.map((sample) => sample.candidateCount);
  return {
    sampleCount: samples.length,
    candidateMean: mean(candidates),
    candidateP95: percentile(candidates, 0.95),
    modelInferenceP50Ms: percentile(inference, 0.5),
    modelInferenceP95Ms: percentile(inference, 0.95),
    submitCompletionP50Ms: percentile(samples.map((sample) => sample.submitCompletionMs), 0.5),
    submitCompletionP95Ms: percentile(samples.map((sample) => sample.submitCompletionMs), 0.95),
  };
}

function shuffledOrdinals(count, seed) {
  const values = Array.from({ length: count }, (_, index) => index);
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  for (let index = values.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [values[index], values[other]] = [values[other], values[index]];
  }
  return values;
}

function finiteArray(values, size, name) {
  if (!Array.isArray(values) || values.length !== size || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${name} 格式无效。`);
  }
  return values.map(Number);
}

function makeCamera(pose, workload) {
  const position = finiteArray(pose.position, 3, 'position');
  const forwardValues = finiteArray(pose.forward, 3, 'forward');
  const forward = new Vector3(...forwardValues).normalize();
  const quaternion = new Quaternion().setFromUnitVectors(FORWARD_AXIS, forward);
  const camera = new PerspectiveCamera(
    Number(workload.fovYDeg),
    Number(pose.aspect),
    Number(workload.near),
    Number(workload.far),
  );
  camera.position.fromArray(position);
  camera.quaternion.copy(quaternion);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return camera;
}

function adapterHardwareGate(adapter) {
  const text = [adapter.vendor, adapter.architecture, adapter.device, adapter.description]
    .map((value) => String(value || '').trim())
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
  return {
    hardware: Boolean(text) && !SOFTWARE_ADAPTER_TOKENS.some((token) => text.includes(token)),
    description: text || 'adapter info unavailable',
  };
}

function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(3)} ms` : '-';
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

class BenchmarkPage {
  constructor() {
    this.elements = Object.fromEntries([
      'secure-status', 'device-label', 'device-model', 'session-count', 'start-button',
      'stop-button', 'download-button', 'scene-id', 'runtime-backend',
      'backend-value', 'adapter-value',
      'timing-source-value', 'workload-value', 'init-value', 'upload-value',
      'progress', 'progress-text', 'results-body', 'message',
    ].map((id) => [id, document.getElementById(id)]));
    this.workload = null;
    this.scenes = new Map();
    this.selectedScene = null;
    this.candidateIds = null;
    this.result = null;
    this.runtime = null;
    this.abortRequested = false;
    this.visibilityViolations = 0;
    this.wakeLock = null;
  }

  async init() {
    this.elements['start-button'].addEventListener('click', () => this.start());
    this.elements['stop-button'].addEventListener('click', () => { this.abortRequested = true; });
    this.elements['download-button'].addEventListener('click', () => this.downloadResult());
    this.elements['scene-id'].addEventListener('change', () => this.loadSelectedScene());
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this.runtime) {
        this.visibilityViolations += 1;
        this.abortRequested = true;
      }
    });

    if (!window.isSecureContext) {
      this.setEnvironment(false, '需要 HTTPS 安全环境');
      this.setMessage('请使用本站 HTTPS 地址。', true);
      return;
    }
    this.setEnvironment(true, '安全环境可用');
    try {
      const response = await fetch(SCENE_MANIFEST_URL, { cache: 'no-cache' });
      if (!response.ok) throw new Error(`场景清单 HTTP ${response.status}`);
      const manifest = await response.json();
      if (manifest.schema !== 'pvs-paper-runtime-scenes-v1' || !Array.isArray(manifest.scenes)) {
        throw new Error('场景清单 schema 不正确');
      }
      for (const scene of manifest.scenes) {
        if (!scene?.id || !scene?.workloadUrl) throw new Error('场景清单条目不完整');
        this.scenes.set(String(scene.id), scene);
        const option = document.createElement('option');
        option.value = String(scene.id);
        option.textContent = String(scene.label || scene.id);
        this.elements['scene-id'].appendChild(option);
      }
      if (!this.scenes.size) throw new Error('场景清单为空');
      this.elements['scene-id'].disabled = false;
      await this.loadSelectedScene();
    } catch (error) {
      this.setEnvironment(false, '工作负载加载失败');
      this.setMessage(error.message, true);
    }
  }

  async loadSelectedScene() {
    this.elements['start-button'].disabled = true;
    this.workload = null;
    this.candidateIds = null;
    try {
      const startedAt = nowMs();
      const scene = this.scenes.get(this.elements['scene-id'].value)
        || this.scenes.values().next().value;
      this.selectedScene = scene;
      const response = await fetch(scene.workloadUrl, { cache: 'force-cache' });
      if (!response.ok) throw new Error(`工作负载 HTTP ${response.status}`);
      this.workload = await response.json();
      if (this.workload.schema !== 'pvs-v4-browser-runtime-workload-v1'
          || !Number.isInteger(this.workload.poseCount)
          || this.workload.poseCount <= 0
          || this.workload.poses?.length !== this.workload.poseCount) {
        throw new Error('工作负载 schema 或 pose 列表不正确');
      }
      const candidateResponse = await fetch(
        new URL(this.workload.candidateFile, response.url),
        { cache: 'force-cache' },
      );
      if (!candidateResponse.ok) throw new Error(`候选文件 HTTP ${candidateResponse.status}`);
      this.candidateIds = new Uint32Array(await candidateResponse.arrayBuffer());
      if (this.candidateIds.length !== this.workload.candidateCount) {
        throw new Error('候选文件长度与工作负载不一致');
      }
      this.workloadLoadMs = nowMs() - startedAt;
      this.elements['workload-value'].textContent = `${this.workload.poseCount.toLocaleString()} poses / ${this.candidateIds.length.toLocaleString()} candidates`;
      this.elements['start-button'].disabled = false;
      this.setMessage('工作负载已加载。模型资产会在点击开始后加载，但加载时间不计入推理。');
    } catch (error) {
      this.setEnvironment(false, '工作负载加载失败');
      this.setMessage(error.message, true);
    }
  }

  setEnvironment(ok, text) {
    const element = this.elements['secure-status'];
    element.textContent = text;
    element.classList.toggle('ok', ok);
    element.classList.toggle('error', !ok);
  }

  setMessage(text, error = false) {
    this.elements.message.textContent = text;
    this.elements.message.classList.toggle('error', error);
  }

  async requestWakeLock() {
    try {
      this.wakeLock = await navigator.wakeLock?.request('screen');
    } catch (_) {
      this.wakeLock = null;
    }
  }

  candidatesForPose(pose) {
    const start = Number(pose.candidateOffset);
    const end = start + Number(pose.candidateCount);
    return this.candidateIds.subarray(start, end);
  }

  async initializeRuntime() {
    const startedAt = nowMs();
    const backendPreference = this.elements['runtime-backend'].value;
    if (backendPreference === 'webgpu' && !navigator.gpu) {
      throw new Error('当前浏览器不支持 WebGPU；请选择 WASM SIMD。');
    }
    const runtime = new InstancePVSRuntime(this.workload.modelAssetUrl, {
      candidateBenchmark: true,
      debugLogging: false,
      backendPreference,
      wasmUrl: this.workload.wasmUrl,
    });
    await runtime.init();
    const initMs = nowMs() - startedAt;
    const active = runtime.active;
    const adapter = active?.webgpuInfo?.adapter || {};
    const gate = backendPreference === 'webgpu' ? adapterHardwareGate(adapter) : null;
    if (gate && !gate.hardware) {
      runtime.dispose();
      throw new Error(`WebGPU 硬件门失败：${gate.description}`);
    }
    this.runtime = runtime;
    this.elements['backend-value'].textContent = runtime.backend;
    this.elements['adapter-value'].textContent = gate?.description || 'WASM SIMD / CPU';
    this.elements['timing-source-value'].textContent = active?.timestampQuerySupported
      ? 'WebGPU timestamp-query（主指标）'
      : (backendPreference === 'wasm' ? 'WASM SIMD 函数调用' : 'queue submit-to-completion');
    this.elements['init-value'].textContent = `${formatMs(initMs)}（不计入推理）`;
    return { runtime, initMs, adapter, gate, backendPreference };
  }

  async runPose(pose) {
    const result = await this.runtime.benchmarkCandidates(
      makeCamera(pose, this.workload),
      this.candidatesForPose(pose),
    );
    const primary = result.gpuKernelMs == null ? result.submitCompletionMs : result.gpuKernelMs;
    if (!Number.isFinite(primary) || primary < 0) throw new Error('推理后端返回了无效计时。');
    return {
      poseId: Number(pose.poseId),
      ordinal: Number(pose.ordinal),
      candidateCount: Number(pose.candidateCount),
      modelInferenceMs: primary,
      gpuKernelMs: result.gpuKernelMs,
      submitCompletionMs: result.submitCompletionMs,
      timingSource: result.timingSource,
    };
  }

  async warmup() {
    for (let index = 0; index < this.workload.warmupOrdinals.length; index += 1) {
      if (this.abortRequested) throw new Error('测试已停止。');
      const ordinal = this.workload.warmupOrdinals[index];
      await this.runPose(this.workload.poses[ordinal]);
      this.elements['progress-text'].textContent = `预热 ${index + 1} / ${this.workload.warmupOrdinals.length}`;
      if ((index & 7) === 7) await sleep(0);
    }
  }

  async runSession(sessionIndex, totalSessions) {
    await this.warmup();
    const order = shuffledOrdinals(
      this.workload.poseCount,
      Number(this.workload.sessionOrderSeed) + sessionIndex,
    );
    const samples = [];
    for (let index = 0; index < order.length; index += 1) {
      if (this.abortRequested) throw new Error('测试已停止。');
      samples.push(await this.runPose(this.workload.poses[order[index]]));
      const totalDone = sessionIndex * order.length + index + 1;
      this.elements.progress.value = totalDone;
      this.elements['progress-text'].textContent = `第 ${sessionIndex + 1}/${totalSessions} 轮，${index + 1}/${order.length}`;
      if ((index & 7) === 7) await sleep(0);
    }
    return {
      sessionIndex,
      orderSeed: Number(this.workload.sessionOrderSeed) + sessionIndex,
      warmupCount: this.workload.warmupOrdinals.length,
      samples,
      summary: summarizeSamples(samples),
    };
  }

  renderResults(sessions) {
    this.elements['results-body'].textContent = '';
    for (const session of sessions) {
      const row = document.createElement('tr');
      const values = [
        `第 ${session.sessionIndex + 1} 轮`,
        session.summary.sampleCount,
        session.summary.candidateMean.toFixed(1),
        formatMs(session.summary.modelInferenceP50Ms),
        formatMs(session.summary.modelInferenceP95Ms),
      ];
      for (const value of values) {
        const cell = document.createElement('td');
        cell.textContent = value;
        row.appendChild(cell);
      }
      this.elements['results-body'].appendChild(row);
    }
  }

  async start() {
    const deviceLabel = this.elements['device-label'].value.trim();
    if (!deviceLabel) {
      this.setMessage('请先填写设备名称。', true);
      this.elements['device-label'].focus();
      return;
    }
    const totalSessions = Number(this.elements['session-count'].value);
    this.abortRequested = false;
    this.visibilityViolations = 0;
    this.result = null;
    this.elements['start-button'].disabled = true;
    this.elements['stop-button'].disabled = false;
    this.elements['download-button'].disabled = true;
    this.elements['scene-id'].disabled = true;
    this.elements['runtime-backend'].disabled = true;
    this.elements['upload-value'].textContent = '等待测试完成';
    this.elements.progress.max = totalSessions * this.workload.poseCount;
    this.elements.progress.value = 0;
    this.setMessage('正在加载模型资产并初始化推理后端；该阶段不计入模型推理时间。');
    await this.requestWakeLock();

    try {
      const runtimeInfo = await this.initializeRuntime();
      const sessions = [];
      for (let sessionIndex = 0; sessionIndex < totalSessions; sessionIndex += 1) {
        sessions.push(await this.runSession(sessionIndex, totalSessions));
        this.renderResults(sessions);
        if (sessionIndex + 1 < totalSessions) await sleep(500);
      }
      const allSamples = sessions.flatMap((session) => session.samples);
      this.result = {
        schema: RESULT_SCHEMA,
        experimentName: this.workload.experimentName,
        createdAt: new Date().toISOString(),
        device: {
          label: deviceLabel,
          model: this.elements['device-model'].value.trim(),
          userAgent: navigator.userAgent,
          platform: navigator.userAgentData?.platform || navigator.platform || '',
          hardwareConcurrency: navigator.hardwareConcurrency || null,
          deviceMemoryGiB: navigator.deviceMemory || null,
          viewport: [window.innerWidth, window.innerHeight],
          devicePixelRatio: window.devicePixelRatio,
        },
        environment: {
          secureContext: window.isSecureContext,
          pageUrl: location.href,
          backend: this.runtime.backend,
          adapter: runtimeInfo.adapter,
          hardwareGate: runtimeInfo.gate,
          wasmSimd: runtimeInfo.backendPreference === 'wasm',
          timestampQuery: Boolean(this.runtime.active?.timestampQuerySupported),
          visibilityViolations: this.visibilityViolations,
        },
        model: {
          schema: this.runtime.meta.schema,
          modelSchema: this.runtime.meta.modelSchema,
          threshold: Number(this.runtime.meta.threshold),
          numInstances: Number(this.runtime.meta.numInstances),
          runtimeAssetBytes: Number(this.runtime.lastInitTimings?.assetBytes || 0),
        },
        workload: {
          schema: this.workload.schema,
          scene: this.workload.scene,
          split: this.workload.split,
          poseCount: this.workload.poseCount,
          candidateCount: this.workload.candidateCount,
          fovYDeg: this.workload.fovYDeg,
          modelFovYDeg: this.workload.modelFovYDeg,
          aspects: this.workload.aspects,
        },
        excludedSetupTimings: {
          workloadDownloadMs: this.workloadLoadMs,
          modelAssetAndPipelineInitMs: runtimeInfo.initMs,
        },
        timingDefinition: {
          primary: this.runtime.active?.timestampQuerySupported ? 'gpuKernelMs' : 'submitCompletionMs',
          includes: 'V4 neural visibility forward for pre-supplied candidate IDs',
          excludes: [
            'model and workload download', 'GPU buffer initialization', 'candidate generation',
            'candidate upload', 'threshold filtering', '60-degree render filtering',
            'GLB aggregation', 'result readback', 'Worker messaging',
          ],
        },
        sessions,
        summary: summarizeSamples(allSamples),
      };
      this.elements['download-button'].disabled = false;
      this.setMessage('本机测试完成，正在上传结果。');
      await this.uploadResult();
    } catch (error) {
      this.setMessage(error.message || String(error), true);
      this.elements['upload-value'].textContent = '未上传';
    } finally {
      this.runtime?.dispose();
      this.runtime = null;
      await this.wakeLock?.release().catch(() => {});
      this.wakeLock = null;
      this.elements['start-button'].disabled = false;
      this.elements['stop-button'].disabled = true;
      this.elements['scene-id'].disabled = false;
      this.elements['runtime-backend'].disabled = false;
    }
  }

  async uploadResult() {
    try {
      const sessionResponse = await fetch('/api/pvs-runtime-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      });
      if (!sessionResponse.ok) throw new Error(`无法创建上传会话：HTTP ${sessionResponse.status}`);
      const session = await sessionResponse.json();
      const response = await fetch('/api/pvs-runtime-results', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-PVS-Session': session.token,
        },
        body: JSON.stringify(this.result),
      });
      if (!response.ok) throw new Error(`结果上传失败：HTTP ${response.status}`);
      const receipt = await response.json();
      this.result.uploadReceipt = receipt.receiptId;
      this.elements['upload-value'].textContent = `成功：${receipt.receiptId}`;
      this.setMessage('测试和上传均已完成。');
    } catch (error) {
      this.elements['upload-value'].textContent = '失败，请下载结果文件';
      this.setMessage(`${error.message} 页面已保留本机下载结果。`, true);
    }
  }

  downloadResult() {
    if (!this.result) return;
    const blob = new Blob([`${JSON.stringify(this.result, null, 2)}\n`], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    const label = this.result.device.label.replace(/[^a-zA-Z0-9_-]+/g, '_').slice(0, 48) || 'device';
    const backend = String(this.result.environment.backend || 'runtime').replace(/[^a-z0-9_-]+/gi, '_');
    link.download = `pvs_v4_${backend}_${label}_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    const page = new BenchmarkPage();
    window.__pvsRuntimeBenchmark = page;
    page.init();
  });
}

export { adapterHardwareGate, percentile, shuffledOrdinals, summarizeSamples };
