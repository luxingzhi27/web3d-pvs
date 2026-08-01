/**
 * Automated Visibility Data Sampler for NeuralPVS (v2 — Concurrent)
 * 
 * Connects N concurrent WebSocket workers to the rcServer, each processing
 * a batch of camera poses in parallel for ~Nx throughput.
 * 
 * Usage:
 *   node sample_visibility.js [options]
 * 
 * Options:
 *   --rc-server <url>       rcServer WebSocket address (default: wss://smart3d.hkust-gz.edu.cn/5640/)
 *   --grid-step <meters>    XZ grid spacing (default: 200)
 *   --num-orientations <n>  Number of yaw angles per position (default: 8)
 *   --concurrency <n>       Number of parallel WebSocket workers (default: 20)
 *   --pipeline-depth <n>    In-flight requests per worker (default: 8)
 *   --delay <ms>            Delay between requests per worker (default: 20)
 *   --pose-limit <n>        Limit total poses for tuning/benchmarking (default: unlimited)
 *   --timeout-ms <ms>       Request timeout in milliseconds (default: 8000)
 * 
 * FOV Strategy:
 *   Real viewer uses fov=60°. SLM2Loader computes backFov = fov * 1.1 = 66°
 *   and applies CameraFrameUpscale = 1.2 on the viewport dimensions.
 *   For training labels we intentionally keep a small extra FOV margin beyond
 *   the viewer's back camera so the teacher set remains slightly conservative
 *   and avoids false negatives when the neural predictor has small pose error.
 * 
 * Angle Strategy:
 *   - YAW (horizontal): Uniformly spaced 360° ÷ N orientations. This ensures
 *     full coverage in all horizontal directions from each position.
 *   - PITCH (vertical): [-30°, -15°, 0°, +15°, +30°]. These cover downward
 *     looking (common when walking on elevated terrain), level gaze (most
 *     common), and upward looking (viewing tall buildings). The range is
 *     limited to ±30° because extreme vertical angles are rare in real roaming.
 *   - ROLL: Fixed at 0° (users don't roll in FPS controls).
 *   - POSITION: Grid sampling across scene AABB at multiple height levels,
 *     representing ground floor, first floor, rooftop, and aerial views.
 */

const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const THREE = require('three');

// ============ Argument Parser ============
function getArg(name, defaultVal) {
  const idx = process.argv.indexOf('--' + name);
  return idx >= 0 ? process.argv[idx + 1] : defaultVal;
}

// ============ Configuration ============
const CONFIG = {
  rcServer: getArg('rc-server', 'wss://smart3d.hkust-gz.edu.cn/5640/'),
  concurrency: parseInt(getArg('concurrency', '20')),
  pipelineDepth: parseInt(getArg('pipeline-depth', '8')),
  requestDelayMs: parseInt(getArg('delay', '20')),
  requestTimeoutMs: parseInt(getArg('timeout-ms', '8000')),
  debugMode: process.argv.includes('--debug'),
  debugLimit: parseInt(getArg('debug-limit', '12')),
  debugVisiblePrefix: parseInt(getArg('debug-visible-prefix', '20')),
  poseLimit: parseInt(getArg('pose-limit', '0')),

  bounds: {
    center: [-464.49610000000007, 26.123992500000004, 251.4833000000001],
    size: [6784.185600000001, 155.752235, 5579.0466]
  },

  gridStepXZ: parseFloat(getArg('grid-step', '200')),
  heightLevels: [5, 15, 30, 50, 80],
  numOrientations: parseInt(getArg('num-orientations', '8')),
  pitchAngles: [-30, -15, 0, 15, 30],

  // Keep a small safety margin beyond the viewer's back camera request.
  realViewerFov: 60,
  backFovMultiplier: 1.1,
  trainingFovBuffer: 1.05,
  get samplingFov() {
    return this.realViewerFov * this.backFovMultiplier * this.trainingFovBuffer;
  },
  aspect: 16 / 9,
  near: 0.01,
  far: 100000,
  viewportWidth: 1920,
  viewportHeight: 1080,
  cameraFrameUpscale: 1.2,

  outputDir: path.join(__dirname, '..', 'assets', 'visibility_data'),
  batchSize: 1000,
};

// ============ Globals ============
let totalSamples = 0;
let totalFailed = 0;
let totalVisibleSum = 0;
let allSamplesBuffer = [];
let outputFileIndex = 0;
let startTime = Date.now();
let completedPoses = 0;
let totalPoses = 0;
const writeLock = { locked: false };

if (CONFIG.debugMode) {
  CONFIG.concurrency = 1;
  CONFIG.pipelineDepth = 1;
  CONFIG.batchSize = Math.max(1, CONFIG.debugLimit);
  CONFIG.requestDelayMs = Math.max(CONFIG.requestDelayMs, 120);
}

// ============ Utility Functions ============
function degToRad(d) { return d * Math.PI / 180; }

function buildCameraRequest(position, rotation, requestId, previousCameraHash) {
  const cam = new THREE.PerspectiveCamera(CONFIG.samplingFov, CONFIG.aspect, CONFIG.near, CONFIG.far);
  cam.position.set(position.x, position.y, position.z);
  // Viewer FPS controls build the camera quaternion from a YXZ Euler.
  const fpsEuler = new THREE.Euler(rotation.x, rotation.y, rotation.z, 'YXZ');
  cam.quaternion.setFromEuler(fpsEuler);
  cam.updateProjectionMatrix();
  cam.updateMatrixWorld(true);

  const mvpMatrix = new THREE.Matrix4();
  mvpMatrix.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);

  const posHashResolution = 1;
  const rotHashResolution = 5;
  const cameraPosHash =
    `${(cam.position.x * posHashResolution).toFixed(0)}-` +
    `${(cam.position.y * posHashResolution).toFixed(0)}-` +
    `${(cam.position.z * posHashResolution).toFixed(0)}`;
  const cameraRotHash =
    `${(cam.rotation.x * rotHashResolution).toFixed(0)}-` +
    `${(cam.rotation.y * rotHashResolution).toFixed(0)}-` +
    `${(cam.rotation.z * rotHashResolution).toFixed(0)}`;
  const newCameraHash = `${cameraPosHash}:${cameraRotHash}`;

  return {
    cameraData: {
      type: 0,
      id: requestId,
      mvp: Array.from(mvpMatrix.elements),
      width: Math.round(CONFIG.viewportWidth * CONFIG.cameraFrameUpscale),
      height: Math.round(CONFIG.viewportHeight * CONFIG.cameraFrameUpscale),
      cull: 0,
      hash: previousCameraHash == null ? '0' : newCameraHash,
    },
    resolvedRotation: {
      x: cam.rotation.x,
      y: cam.rotation.y,
      z: cam.rotation.z,
    },
    requestFov: cam.fov,
    cameraHash: newCameraHash,
  };
}

function selectDebugPoses(poses, limit) {
  if (poses.length <= limit) return poses;

  const selected = [];
  const used = new Set();
  const step = (poses.length - 1) / Math.max(1, limit - 1);

  for (let i = 0; i < limit; i++) {
    const idx = Math.round(i * step);
    if (!used.has(idx)) {
      used.add(idx);
      selected.push(poses[idx]);
    }
  }

  return selected;
}

function summarizeArray(arr, limit) {
  return arr.slice(0, Math.min(limit, arr.length)).join(', ');
}

function generateAllPoses() {
  const poses = [];
  const { center, size } = CONFIG.bounds;
  const minX = center[0] - size[0] / 2;
  const maxX = center[0] + size[0] / 2;
  const minZ = center[2] - size[2] / 2;
  const maxZ = center[2] + size[2] / 2;
  const groundY = center[1] - size[1] / 2;

  for (let x = minX; x <= maxX; x += CONFIG.gridStepXZ) {
    for (let z = minZ; z <= maxZ; z += CONFIG.gridStepXZ) {
      for (const hoff of CONFIG.heightLevels) {
        const y = groundY + hoff;
        for (let yi = 0; yi < CONFIG.numOrientations; yi++) {
          const yaw = (yi / CONFIG.numOrientations) * 2 * Math.PI;
          for (const pitchDeg of CONFIG.pitchAngles) {
            poses.push({
              pose_index: poses.length,
              position: { x, y, z },
              rotation: { x: degToRad(pitchDeg), y: yaw, z: 0 }
            });
          }
        }
      }
    }
  }
  return poses;
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
function formatDuration(ms) {
  const s = Math.floor(ms / 1000) % 60;
  const m = Math.floor(ms / 60000) % 60;
  const h = Math.floor(ms / 3600000);
  return `${h}h ${m}m ${s}s`;
}

function flushBuffer() {
  if (allSamplesBuffer.length === 0) return;
  if (!fs.existsSync(CONFIG.outputDir)) fs.mkdirSync(CONFIG.outputDir, { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const fn = `vis_${String(outputFileIndex++).padStart(4, '0')}_${ts}.jsonl`;
  const fp = path.join(CONFIG.outputDir, fn);
  fs.writeFileSync(fp, allSamplesBuffer.map(s => JSON.stringify(s)).join('\n') + '\n');
  console.log(`  💾 Wrote ${allSamplesBuffer.length} samples → ${fn}`);
  allSamplesBuffer = [];
}

function pushSample(sample) {
  allSamplesBuffer.push(sample);
  totalSamples++;
  totalVisibleSum += sample.visible_count;
  if (allSamplesBuffer.length >= CONFIG.batchSize) {
    flushBuffer();
  }
}

function printProgress() {
  const elapsed = Date.now() - startTime;
  const speed = totalSamples / (elapsed / 1000);
  const remaining = speed > 0 ? (totalPoses - completedPoses) / speed : 0;
  const avgVis = totalSamples > 0 ? (totalVisibleSum / totalSamples).toFixed(0) : '?';
  process.stdout.write(
    `\r  📊 ${completedPoses.toLocaleString()}/${totalPoses.toLocaleString()} | ` +
    `✅ ${totalSamples.toLocaleString()} ok ❌ ${totalFailed} fail | ` +
    `Avg vis: ${avgVis} | ${speed.toFixed(1)}/s | ` +
    `ETA: ${formatDuration(remaining * 1000)}     `
  );
}

// ============ Worker: One WebSocket Connection ============
async function createWorker(workerId, poseBatch) {
  return new Promise((resolve) => {
    const ws = new WebSocket(CONFIG.rcServer);
    ws.binaryType = 'arraybuffer';
    let reqId = workerId * 1000000;
    let nextPoseIndex = 0;
    let inFlight = 0;
    let lastCameraHash = null;
    let lastDebugSignature = null;
    let settled = false;
    const pendingRequests = new Map();

    function markProgress() {
      completedPoses++;
      if (completedPoses % 500 === 0) printProgress();
    }

    function settleWorker() {
      if (settled) return;
      settled = true;
      resolve();
    }

    function handleFailure(meta, err) {
      totalFailed++;
      if (CONFIG.debugMode) {
        console.error(
          `[Debug][Err ${meta.localIndex + 1}/${poseBatch.length}] pose_index=${meta.pose.pose_index}`,
          err.message || err
        );
      }
    }

    function handleSuccess(meta, ml, wl) {
      pushSample({
        sample_id: totalSamples,
        pose_index: meta.pose.pose_index,
        pos: [
          +meta.pose.position.x.toFixed(2),
          +meta.pose.position.y.toFixed(2),
          +meta.pose.position.z.toFixed(2)
        ],
        rot: [
          +meta.request.resolvedRotation.x.toFixed(4),
          +meta.request.resolvedRotation.y.toFixed(4),
          +meta.request.resolvedRotation.z.toFixed(4)
        ],
        fov: CONFIG.realViewerFov,
        request_fov: +meta.request.requestFov.toFixed(1),
        aspect: CONFIG.aspect,
        camera_hash: meta.request.cameraHash,
        visible_count: ml.length,
        visible_ids: ml,
        weights: wl
      });

      if (CONFIG.debugMode) {
        const signature = `${ml.length}|${ml.slice(0, CONFIG.debugVisiblePrefix).join(',')}`;
        const repeated = lastDebugSignature === signature;
        console.log(
          `[Debug][Res ${meta.localIndex + 1}/${poseBatch.length}] ` +
          `pose_index=${meta.pose.pose_index} visible_count=${ml.length} repeated_signature=${repeated} ` +
          `visible_ids[0:${CONFIG.debugVisiblePrefix}]=[${summarizeArray(ml, CONFIG.debugVisiblePrefix)}] ` +
          `weights[0:${Math.min(10, wl.length)}]=[${summarizeArray(wl, 10)}]`
        );
        if (repeated) {
          console.warn(`[Debug][Warn] Consecutive response signature repeated for pose_index=${meta.pose.pose_index}`);
        }
        lastDebugSignature = signature;
      }
    }

    function maybeFinish() {
      if (!settled && nextPoseIndex >= poseBatch.length && inFlight === 0) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close();
        } else {
          settleWorker();
        }
      }
    }

    function schedulePump() {
      if (settled) return;
      setTimeout(pump, Math.max(0, CONFIG.requestDelayMs));
    }

    function pump() {
      if (settled || ws.readyState !== WebSocket.OPEN) return;

      while (inFlight < CONFIG.pipelineDepth && nextPoseIndex < poseBatch.length) {
        const pose = poseBatch[nextPoseIndex];
        const localIndex = nextPoseIndex;
        nextPoseIndex++;

        const id = reqId++;
        const request = buildCameraRequest(pose.position, pose.rotation, id, lastCameraHash);
        lastCameraHash = request.cameraHash;

        if (CONFIG.debugMode) {
          console.log(
            `[Debug][Req ${localIndex + 1}/${poseBatch.length}] ` +
            `pose_index=${pose.pose_index} id=${id} hash=${request.cameraData.hash} next_hash=${request.cameraHash} ` +
            `pos=${pose.position.x.toFixed(2)},${pose.position.y.toFixed(2)},${pose.position.z.toFixed(2)} ` +
            `rot_in=${pose.rotation.x.toFixed(4)},${pose.rotation.y.toFixed(4)},${pose.rotation.z.toFixed(4)} ` +
            `rot_sent=${request.resolvedRotation.x.toFixed(4)},${request.resolvedRotation.y.toFixed(4)},${request.resolvedRotation.z.toFixed(4)} ` +
            `fov=${request.requestFov.toFixed(1)} viewport=${request.cameraData.width}x${request.cameraData.height}`
          );
        }

        const meta = { id, pose, localIndex, request };
        meta.timeout = setTimeout(() => {
          if (!pendingRequests.has(id)) return;
          pendingRequests.delete(id);
          inFlight--;
          handleFailure(meta, new Error('timeout'));
          markProgress();
          schedulePump();
          maybeFinish();
        }, CONFIG.requestTimeoutMs);

        pendingRequests.set(id, meta);
        inFlight++;
        ws.send(JSON.stringify(request.cameraData));

        if (CONFIG.requestDelayMs > 0) {
          schedulePump();
          return;
        }
      }

      maybeFinish();
    }

    ws.on('open', () => {
      pump();
    });

    ws.on('message', (data) => {
      try {
        const buf = data instanceof ArrayBuffer ? data : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
        const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf)));
        const meta = pendingRequests.get(json.id);
        if (!meta) return;

        clearTimeout(meta.timeout);
        pendingRequests.delete(json.id);
        inFlight--;

        const ml = typeof json.list === 'string' ? JSON.parse(json.list) : json.list;
        const wl = typeof json.weight === 'string' ? JSON.parse(json.weight) : json.weight;
        handleSuccess(meta, ml, wl);
        markProgress();

        if (CONFIG.requestDelayMs > 0) {
          schedulePump();
        } else {
          pump();
        }
        maybeFinish();
      } catch (e) { /* parse error, will timeout */ }
    });

    ws.on('error', (err) => {
      if (CONFIG.debugMode) {
        console.error(`[Debug][Worker ${workerId}] socket error`, err.message || err);
      }
    });

    ws.on('close', () => {
      if (settled) {
        settleWorker();
        return;
      }

      for (const meta of pendingRequests.values()) {
        clearTimeout(meta.timeout);
        inFlight--;
        handleFailure(meta, new Error('socket closed'));
        markProgress();
      }
      pendingRequests.clear();

      while (nextPoseIndex < poseBatch.length) {
        const pose = poseBatch[nextPoseIndex];
        totalFailed++;
        if (CONFIG.debugMode) {
          console.error(
            `[Debug][Err ${nextPoseIndex + 1}/${poseBatch.length}] pose_index=${pose.pose_index}`,
            'socket closed before dispatch'
          );
        }
        nextPoseIndex++;
        markProgress();
      }

      settleWorker();
    });
  });
}

// ============ Main ============
async function main() {
  console.log('═══════════════════════════════════════════════════');
  console.log(' NeuralPVS Visibility Sampler v2 (Concurrent)');
  console.log('═══════════════════════════════════════════════════');
  console.log(`  rcServer:     ${CONFIG.rcServer}`);
  console.log(`  Concurrency:  ${CONFIG.concurrency} parallel WebSocket workers`);
  console.log(`  Pipeline:     ${CONFIG.pipelineDepth} in-flight requests per worker`);
  console.log(`  Grid step:    ${CONFIG.gridStepXZ}m (XZ)`);
  if (!CONFIG.debugMode && CONFIG.poseLimit > 0) {
    console.log(`  Pose limit:   ${CONFIG.poseLimit}`);
  }
  console.log(`  Heights:      ${CONFIG.heightLevels.join(', ')}m`);
  console.log(`  Yaw dirs:     ${CONFIG.numOrientations}`);
  console.log(`  Pitch:        ${CONFIG.pitchAngles.join(', ')}°`);
  console.log(`  Real FOV:     ${CONFIG.realViewerFov}°`);
  console.log(`  Sampling FOV: ${CONFIG.samplingFov.toFixed(1)}° (= ${CONFIG.realViewerFov} × ${CONFIG.backFovMultiplier} × ${CONFIG.trainingFovBuffer})`);
  console.log(`  Viewport:     ${CONFIG.viewportWidth}×${CONFIG.viewportHeight} × ${CONFIG.cameraFrameUpscale} upscale`);
  console.log(`  Output:       ${CONFIG.outputDir}`);
  if (CONFIG.debugMode) {
    console.log(`  Debug mode:   ON (${CONFIG.debugLimit} sampled poses, visible prefix ${CONFIG.debugVisiblePrefix})`);
  }
  console.log('───────────────────────────────────────────────────');

  let poses = generateAllPoses();
  if (CONFIG.debugMode) {
    poses = selectDebugPoses(poses, CONFIG.debugLimit);
  } else if (CONFIG.poseLimit > 0) {
    poses = poses.slice(0, CONFIG.poseLimit);
  }
  totalPoses = poses.length;
  console.log(`  📐 Total poses: ${totalPoses.toLocaleString()}`);

  // Split poses into batches for workers
  const batchSize = Math.ceil(totalPoses / CONFIG.concurrency);
  const batches = [];
  for (let i = 0; i < totalPoses; i += batchSize) {
    batches.push(poses.slice(i, i + batchSize));
  }
  console.log(`  🧵 Dispatching to ${batches.length} workers (${batchSize.toLocaleString()} poses each)`);
  const estSpeed = CONFIG.concurrency * CONFIG.pipelineDepth * (1000 / (CONFIG.requestDelayMs + 30));
  console.log(`  ⏱  Est. speed: ~${estSpeed.toFixed(0)} samples/s → ETA: ${formatDuration(totalPoses / estSpeed * 1000)}`);
  console.log('───────────────────────────────────────────────────');

  // Launch all workers concurrently
  startTime = Date.now();
  const workerPromises = batches.map((batch, i) => createWorker(i, batch));
  await Promise.all(workerPromises);

  // Flush remaining
  flushBuffer();
  printProgress();

  const elapsed = Date.now() - startTime;
  const avgVis = totalSamples > 0 ? (totalVisibleSum / totalSamples).toFixed(0) : 0;

  console.log('\n');
  console.log('═══════════════════════════════════════════════════');
  console.log(' ✅ Sampling Complete!');
  console.log('═══════════════════════════════════════════════════');
  console.log(`  Total samples:    ${totalSamples.toLocaleString()}`);
  console.log(`  Failed requests:  ${totalFailed}`);
  console.log(`  Avg visible IDs:  ${avgVis}`);
  console.log(`  Total time:       ${formatDuration(elapsed)}`);
  console.log(`  Throughput:       ${(totalSamples / (elapsed / 1000)).toFixed(1)} samples/sec`);
  console.log(`  Output files:     ${outputFileIndex} files`);
  console.log(`  Output dir:       ${CONFIG.outputDir}`);
  console.log('═══════════════════════════════════════════════════');

  // Write summary
  const summary = {
    total_samples: totalSamples,
    failed_requests: totalFailed,
    avg_visible_count: parseFloat(avgVis),
    total_scene_components: 18831,
    scene_bounds: CONFIG.bounds,
    sampling_config: {
      grid_step_xz: CONFIG.gridStepXZ,
      height_levels: CONFIG.heightLevels,
      num_orientations: CONFIG.numOrientations,
      pitch_angles: CONFIG.pitchAngles,
      real_viewer_fov: CONFIG.realViewerFov,
      sampling_fov: +CONFIG.samplingFov.toFixed(1),
      fov_formula: `${CONFIG.realViewerFov} × ${CONFIG.backFovMultiplier} (backFov) × ${CONFIG.trainingFovBuffer} (safety buffer) = ${CONFIG.samplingFov.toFixed(1)}°`,
      viewport: `${CONFIG.viewportWidth}×${CONFIG.viewportHeight}`,
      camera_frame_upscale: CONFIG.cameraFrameUpscale,
      concurrency: CONFIG.concurrency,
      pipeline_depth: CONFIG.pipelineDepth,
      request_timeout_ms: CONFIG.requestTimeoutMs,
      pose_limit: CONFIG.poseLimit,
      debug_mode: CONFIG.debugMode,
    },
    duration_ms: elapsed,
    timestamp: new Date().toISOString()
  };
  const sp = path.join(CONFIG.outputDir, 'sampling_summary.json');
  fs.writeFileSync(sp, JSON.stringify(summary, null, 2));
  console.log(`  📋 Summary → sampling_summary.json`);

  process.exit(0);
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
