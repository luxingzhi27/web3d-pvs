import { GeometryShellHZB } from './GeometryShellHZB.js';

const FORWARD_AXIS = [0, 0, -1];

function normalize(value, fallback = [0, 0, -1]) {
  const length = Math.hypot(...value);
  if (!Number.isFinite(length) || length < 1e-8) return fallback.slice();
  return value.map((component) => component / length);
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function multiplyMatrix(a, b) {
  const out = new Array(16).fill(0);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      for (let k = 0; k < 4; k += 1) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
    }
  }
  return out;
}

function cameraFromPose(pose, defaults) {
  const position = Array.from(pose.position || pose.cameraPos || pose.camera_pos, Number);
  const forward = normalize(Array.from(pose.forward || pose.cameraForward || pose.camera_forward, Number));
  const upSeed = Math.abs(forward[1]) > 0.98 ? [0, 0, 1] : [0, 1, 0];
  const right = normalize(cross(forward, upSeed), [1, 0, 0]);
  const up = normalize(cross(right, forward), [0, 1, 0]);
  const fovYDeg = Number(pose.fovYDeg ?? pose.fov_y ?? pose.renderFovYDeg ?? defaults.fovYDeg);
  const aspect = Number(pose.aspect ?? defaults.aspect);
  const near = Number(pose.near ?? defaults.near);
  const far = Number(pose.far ?? defaults.far);
  const tanY = Math.tan((fovYDeg * Math.PI) / 360);
  const projection = [
    1 / (tanY * aspect), 0, 0, 0,
    0, 1 / tanY, 0, 0,
    0, 0, far / (near - far), -1,
    0, 0, (near * far) / (near - far), 0,
  ];
  const view = [
    right[0], up[0], -forward[0], 0,
    right[1], up[1], -forward[1], 0,
    right[2], up[2], -forward[2], 0,
    -dot(right, position), -dot(up, position), dot(forward, position), 1,
  ];
  return {
    position,
    forward,
    up,
    viewMatrix: view,
    viewProjectionMatrix: multiplyMatrix(projection, view),
    fovYDeg,
    aspect,
    near,
    far,
    width: defaults.width,
    height: defaults.height,
  };
}

function getWebGLInfo() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  const extension = gl?.getExtension('WEBGL_debug_renderer_info');
  return {
    vendor: String(gl && extension ? gl.getParameter(extension.UNMASKED_VENDOR_WEBGL) : ''),
    renderer: String(gl && extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : ''),
    version: String(gl ? gl.getParameter(gl.VERSION) : ''),
  };
}

async function readJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`workload HTTP ${response.status}`);
  return response.json();
}

async function readCandidateIds(url, count) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`candidate file HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== count * 4) throw new Error('candidate file byte length mismatch');
  return new Uint32Array(buffer);
}

function summarize(samples) {
  const values = samples.map((sample) => sample.timings.totalMs);
  const sorted = [...values].sort((a, b) => a - b);
  const quantile = (value) => {
    if (!sorted.length) return null;
    const position = (sorted.length - 1) * value;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    return lower === upper ? sorted[lower] : sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
  };
  return {
    sampleCount: samples.length,
    totalP50Ms: quantile(0.5),
    totalP95Ms: quantile(0.95),
    candidateMean: samples.length
      ? samples.reduce((sum, sample) => sum + sample.candidateCount, 0) / samples.length : null,
    visibleMean: samples.length
      ? samples.reduce((sum, sample) => sum + sample.visibleCount, 0) / samples.length : null,
  };
}

async function runBenchmark(config) {
  const workload = await readJson(config.workloadUrl);
  if (workload.schema !== 'geometry-shell-hzb-browser-workload-v1') {
    throw new Error(`unsupported HZB workload schema: ${workload.schema}`);
  }
  const candidateIds = await readCandidateIds(
    new URL(workload.candidateFile, `${config.workloadUrl.replace(/[^/]*$/, '')}`).toString(),
    Number(workload.candidateCount),
  );
  const defaults = {
    width: Number(config.width || workload.width || 512),
    height: Number(config.height || workload.height || 288),
    aspect: Number(config.aspect || workload.aspect || 16 / 9),
    fovYDeg: Number(config.fovYDeg || workload.fovYDeg || 60),
    near: Number(config.near || workload.near || 0.05),
    far: Number(config.far || workload.far || 20000),
  };
  const runtime = new GeometryShellHZB(config.assetBaseUrl, {
    width: defaults.width,
    height: defaults.height,
  });
  await runtime.init();
  const rows = Array.isArray(config.ordinals) && config.ordinals.length
    ? config.ordinals.map((ordinal) => workload.poses[Number(ordinal)])
    : workload.poses;
  if (rows.some((row) => !row)) throw new Error('workload ordinal is out of range');
  const warmupCount = Math.min(Number(config.warmupCount || 0), rows.length);
  for (let index = 0; index < warmupCount; index += 1) {
    const row = rows[index];
    const ids = candidateIds.subarray(Number(row.candidateOffset), Number(row.candidateOffset) + Number(row.candidateCount));
    const camera = cameraFromPose(row, defaults);
    if (config.mode === 'Region66' && Array.isArray(row.subposes)) {
      await runtime.queryRegion66(row.subposes.map((pose) => cameraFromPose(pose, defaults)), ids, config);
    } else {
      await runtime.queryPoint60(camera, ids, config);
    }
  }
  const samples = [];
  for (const row of rows.slice(warmupCount)) {
    const ids = candidateIds.subarray(Number(row.candidateOffset), Number(row.candidateOffset) + Number(row.candidateCount));
    const camera = cameraFromPose(row, defaults);
    const result = config.mode === 'Region66' && Array.isArray(row.subposes)
      ? await runtime.queryRegion66(row.subposes.map((pose) => cameraFromPose(pose, defaults)), ids, config)
      : await runtime.queryPoint60(camera, ids, config);
    samples.push({
      poseId: Number(row.poseId ?? row.pose_id ?? row.ordinal),
      candidateCount: Number(row.candidateCount),
      visibleCount: Number(result.visibleCount),
      uncertainCount: Number(result.uncertainCount || 0),
      passCount: Number(result.passCount || 1),
      perPoseVisibleCounts: Array.isArray(result.perPose)
        ? result.perPose.map((pose) => Number(pose.visibleCount || 0))
        : [Number(result.visibleCount || 0)],
      visibleInstanceIds: Array.from(result.visibleIds),
      modelList: result.modelList,
      timings: result.timings,
    });
  }
  const adapterInfo = runtime.webgpuInfo?.adapter || {};
  const gpuBackend = { api: 'webgpu', ...adapterInfo };
  const webglInfo = getWebGLInfo();
  const result = {
    schema: 'geometry-shell-hzb-browser-result-v1',
    mode: config.mode || 'Point60',
    assetBaseUrl: config.assetBaseUrl,
    workload: {
      schema: workload.schema,
      scene: workload.scene,
      split: workload.split,
      poseCount: rows.length - warmupCount,
      candidateCount: workload.candidateCount,
      width: defaults.width,
      height: defaults.height,
      fovYDeg: defaults.fovYDeg,
    },
    gpuBackend,
    adapterInfo,
    webglInfo,
    gpuValidationErrors: [...(runtime.gpuValidationErrors || [])],
    samples,
    summary: summarize(samples),
  };
  runtime.dispose();
  return result;
}

window.runGeometryShellHZBBenchmark = runBenchmark;
window.__geometryShellHZBReady = true;
