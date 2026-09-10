import * as THREE from '/node_modules/three/build/three.module.js';
import { GLTFLoader } from '/node_modules/three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from '/node_modules/three/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from '/node_modules/three/examples/jsm/libs/meshopt_decoder.module.js';
import { buildComponentIdsByGlb } from './component_mapping.js';
import { patchColorIdAlphaFragmentShader } from './color_id_alpha.js';

THREE.ColorManagement.enabled = false;

const statusEl = document.getElementById('status');
const ID_BASE = 1;
const WEIGHT_SCALE = 1000000;

function setStatus(message) {
  statusEl.textContent = message;
}

async function loadJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  return response.json();
}

function colorFromComponentId(componentId) {
  const encoded = componentId + ID_BASE;
  return new THREE.Color(
    ((encoded >> 0) & 255) / 255,
    ((encoded >> 8) & 255) / 255,
    ((encoded >> 16) & 255) / 255,
  );
}

function decodeComponentId(r, g, b, a) {
  if (a === 0) return -1;
  const encoded = r | (g << 8) | (b << 16);
  return encoded - ID_BASE;
}

function normalizeVector(values, fallback) {
  const v = new THREE.Vector3(Number(values?.[0]), Number(values?.[1]), Number(values?.[2]));
  if (!Number.isFinite(v.x) || !Number.isFinite(v.y) || !Number.isFinite(v.z) || v.lengthSq() < 1e-12) {
    return fallback.clone();
  }
  return v.normalize();
}

function preserveCutoutAlpha(material, sourceMaterial) {
  if (!sourceMaterial?.map || !(Number(sourceMaterial.alphaTest) > 0)) return material;
  material.map = sourceMaterial.map;
  material.alphaTest = Number(sourceMaterial.alphaTest);
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = patchColorIdAlphaFragmentShader(shader.fragmentShader);
  };
  material.customProgramCacheKey = () => `component-color-id-alpha-v1:${material.alphaTest}`;
  return material;
}

function makeFlatMaterial(componentId, sourceMaterial) {
  return preserveCutoutAlpha(new THREE.MeshBasicMaterial({
    color: colorFromComponentId(componentId),
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
    toneMapped: false,
  }), sourceMaterial);
}

function makeInstanceColorMaterial(sourceMaterial) {
  return preserveCutoutAlpha(new THREE.MeshBasicMaterial({
    vertexColors: true,
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
    toneMapped: false,
  }), sourceMaterial);
}

function replaceMaterials(source, factory) {
  return Array.isArray(source)
    ? source.map((material) => factory(material))
    : factory(source);
}

function forceWhiteVertexColors(geometry) {
  const position = geometry.getAttribute('position');
  if (!position) return;
  const colors = new Float32Array(position.count * 3);
  colors.fill(1);
  // MeshBasicMaterial multiplies the ordinary vertex color by instanceColor.
  // Replacing any source color also prevents material appearance from
  // changing the encoded component ID in the off-screen pass.
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, false));
}

function assignColorIdMaterials(root, componentIds) {
  const meshes = [];
  root.traverse((object) => {
    if (object.isMesh || object.isInstancedMesh) meshes.push(object);
  });
  if (meshes.length === 0 || componentIds.length === 0) return;

  const instancedMeshes = meshes.filter((object) => object.isInstancedMesh);
  if (instancedMeshes.length > 0) {
    let cursor = 0;
    for (const mesh of instancedMeshes) {
      const count = Math.max(0, Number(mesh.count || 0));
      // The GLB stores the complete prototype placement set in the instance
      // matrix buffer. Three.js cannot derive a conservative aggregate bounds
      // from EXT_mesh_gpu_instancing during loading, so its default frustum
      // test can discard the whole prototype before the instance matrices are
      // evaluated. Color-ID sampling must render the full placement set.
      mesh.frustumCulled = false;
      forceWhiteVertexColors(mesh.geometry);
      mesh.material = replaceMaterials(mesh.material, makeInstanceColorMaterial);
      for (let i = 0; i < count; i += 1) {
        // Instanced scene assets map placements into the source component ID
        // space. If a future mapping supplies one ID per placement, consume
        // those IDs in order and repeat the last ID for extra placements.
        const idIndex = Math.min(cursor + i, componentIds.length - 1);
        mesh.setColorAt(i, colorFromComponentId(componentIds[idIndex]));
      }
      mesh.instanceColor.needsUpdate = true;
      cursor += Math.min(count, componentIds.length);
    }
    for (const mesh of meshes) {
      if (!mesh.isInstancedMesh) {
        mesh.material = replaceMaterials(
          mesh.material,
          (sourceMaterial) => makeFlatMaterial(componentIds[0], sourceMaterial),
        );
      }
    }
    return;
  }

  if (componentIds.length === meshes.length) {
    for (let i = 0; i < meshes.length; i += 1) {
      meshes[i].material = replaceMaterials(
        meshes[i].material,
        (sourceMaterial) => makeFlatMaterial(componentIds[i], sourceMaterial),
      );
    }
    return;
  }

  const componentId = componentIds[0];
  for (const mesh of meshes) {
    mesh.material = replaceMaterials(
      mesh.material,
      (sourceMaterial) => makeFlatMaterial(componentId, sourceMaterial),
    );
  }
}

function disposeObject(root) {
  root.traverse((object) => {
    if (!object.isMesh && !object.isInstancedMesh) return;
    if (object.geometry) object.geometry.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (material && typeof material.dispose === 'function') material.dispose();
    }
  });
}

async function loadSceneObjects(options, runtimeMeta, glbIndex) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0, 0, 0);

  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath('/node_modules/three/examples/jsm/libs/draco/gltf/');
  const loader = new GLTFLoader();
  loader.setDRACOLoader(dracoLoader);
  loader.setMeshoptDecoder(MeshoptDecoder);

  const entries = Array.isArray(glbIndex.entries) ? glbIndex.entries : [];
  const componentIdsByGlb = buildComponentIdsByGlb(runtimeMeta, glbIndex);
  const allowedGlbIds = Array.isArray(options.glbIdList)
    ? new Set(options.glbIdList.map((value) => Number(value)).filter(Number.isFinite))
    : null;
  const maxGlbs = Number.isFinite(Number(options.maxGlbs)) && Number(options.maxGlbs) > 0
    ? Math.min(entries.length, Number(options.maxGlbs))
    : entries.length;
  let loaded = 0;
  let skipped = 0;
  const objects = [];

  // Concurrent batched GLB load: serially awaiting ~275k tiny GLBs takes hours, so fetch in
  // parallel batches. Each batch awaits a Promise.all of `loadConcurrency` loader.loadAsync calls.
  const loadConcurrency = Number.isFinite(Number(options.loadConcurrency)) && Number(options.loadConcurrency) > 0
    ? Number(options.loadConcurrency)
    : 64;
  for (let start = 0; start < maxGlbs; start += loadConcurrency) {
    const end = Math.min(maxGlbs, start + loadConcurrency);
    const batch = [];
    for (let i = start; i < end; i += 1) {
      const entry = entries[i];
      if (allowedGlbIds && !allowedGlbIds.has(Number(entry.globalId))) {
        skipped += 1;
        continue;
      }
      const componentIds = componentIdsByGlb.get(Number(entry.globalId));
      if (!componentIds) throw new Error(`globalGlbId ${entry.globalId} has no component mapping`);
      const url = `/assets/${entry.path || `task-${entry.taskId}/glb/LOD0/sub_${entry.baseId}.glb`}`;
      batch.push(
        loader.loadAsync(url).then((gltf) => {
          assignColorIdMaterials(gltf.scene, componentIds);
          scene.add(gltf.scene);
          objects.push(gltf.scene);
          loaded += 1;
        }).catch((error) => {
          console.warn('[sampler] failed to load', url, error);
          skipped += 1;
        })
      );
    }
    await Promise.all(batch);
    if (loaded % 100 < loadConcurrency || start + loadConcurrency >= maxGlbs) {
      setStatus(`loading glb ${end}/${maxGlbs}, loaded=${loaded}, skipped=${skipped}`);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }

  return {
    scene,
    objects,
    loaded,
    skipped,
    totalEntries: entries.length,
    loadedGlbSubset: allowedGlbIds ? allowedGlbIds.size : null,
  };
}

function fallbackPosePlan(options) {
  const yaws = Array.isArray(options.yaws) && options.yaws.length > 0 ? options.yaws : [0, 90, 180, 270];
  const pitches = Array.isArray(options.pitches) && options.pitches.length > 0 ? options.pitches : [0];
  const poses = [];
  let index = 0;
  for (const yaw of yaws) {
    for (const pitch of pitches) {
      const yawRad = Number(yaw) * Math.PI / 180;
      const pitchRad = Number(pitch) * Math.PI / 180;
      poses.push({
        pose_index: index,
        sample_category: 'fallback',
        sample_category_id: 255,
        camera_pos: [0, 0, 0],
        camera_forward: [
          -Math.sin(yawRad) * Math.cos(pitchRad),
          Math.sin(pitchRad),
          -Math.cos(yawRad) * Math.cos(pitchRad),
        ],
      fov_y: Number(options.fovYDeg || 66),
        aspect: Number(options.width || 256) / Math.max(1, Number(options.height || 144)),
        width: Number(options.width || 256),
        height: Number(options.height || 144),
      });
      index += 1;
    }
  }
  return poses;
}

function makeCameraForPose(pose, options) {
  const aspect = Number.isFinite(Number(pose.aspect)) && Number(pose.aspect) > 0
    ? Number(pose.aspect)
    : Number(options.width || pose.width || 256) / Math.max(1, Number(options.height || pose.height || 144));
  const height = Math.max(1, Math.round(Number(options.height || pose.height || 144)));
  const width = Math.max(1, Math.round(height * aspect));
  const fovY = Number.isFinite(Number(pose.fov_y)) ? Number(pose.fov_y) : Number(options.fovYDeg || 66);
  const camera = new THREE.PerspectiveCamera(fovY, aspect, Number(options.near || 0.01), Number(options.far || 1000000));
  camera.position.fromArray((pose.camera_pos || [0, 0, 0]).map(Number));
  const forward = normalizeVector(pose.camera_forward, new THREE.Vector3(0, 0, -1));
  const target = camera.position.clone().add(forward);
  camera.up.set(0, 1, 0);
  if (Math.abs(forward.dot(camera.up)) > 0.98) camera.up.set(0, 0, 1);
  camera.lookAt(target);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();
  return { camera, width, height, aspect, fovY, forward };
}

function countVisiblePixels(buffer) {
  const counts = new Map();
  for (let i = 0; i < buffer.length; i += 4) {
    const id = decodeComponentId(buffer[i], buffer[i + 1], buffer[i + 2], buffer[i + 3]);
    if (id < 0) continue;
    counts.set(id, (counts.get(id) || 0) + 1);
  }
  return counts;
}

function buildRecord(pose, renderInfo, counts) {
  const totalPixels = Math.max(1, renderInfo.width * renderInfo.height);
  const visible = Array.from(counts.entries()).sort((a, b) => a[0] - b[0]);
  const ids = [];
  const weights = [];
  const ratios = [];
  let visiblePixelCount = 0;
  for (const [id, pixels] of visible) {
    visiblePixelCount += pixels;
    ids.push(id);
    ratios.push(pixels / totalPixels);
    weights.push(Math.max(1, Math.round((pixels / totalPixels) * WEIGHT_SCALE)));
  }
  return {
    sampler: 'three_color_id',
    weight_semantics: 'screen_coverage_ppm',
    weight_scale: WEIGHT_SCALE,
    pose_index: Number(pose.pose_index ?? 0),
    viewcell_id: Number.isFinite(Number(pose.viewcell_id)) ? Number(pose.viewcell_id) : undefined,
    subpose_id: Number.isFinite(Number(pose.subpose_id)) ? Number(pose.subpose_id) : undefined,
    viewcell_category: pose.viewcell_category,
    viewcell_center: Array.isArray(pose.viewcell_center) ? pose.viewcell_center.map(Number) : undefined,
    viewcell_shape: pose.viewcell_shape,
    viewcell_half_extent: Array.isArray(pose.viewcell_half_extent) ? pose.viewcell_half_extent.map(Number) : undefined,
    viewcell_radius: Number.isFinite(Number(pose.viewcell_radius)) ? Number(pose.viewcell_radius) : undefined,
    viewcell_forward: Array.isArray(pose.viewcell_forward) ? pose.viewcell_forward.map(Number) : undefined,
    viewcell_yaw_deg: Number.isFinite(Number(pose.viewcell_yaw_deg)) ? Number(pose.viewcell_yaw_deg) : undefined,
    viewcell_pitch_deg: Number.isFinite(Number(pose.viewcell_pitch_deg)) ? Number(pose.viewcell_pitch_deg) : undefined,
    pvs_fov_y: Number.isFinite(Number(pose.pvs_fov_y)) ? Number(pose.pvs_fov_y) : undefined,
    pvs_fov_x: Number.isFinite(Number(pose.pvs_fov_x)) ? Number(pose.pvs_fov_x) : undefined,
    pvs_back_offset: Number.isFinite(Number(pose.pvs_back_offset)) ? Number(pose.pvs_back_offset) : undefined,
    split: pose.split,
    sample_category: pose.sample_category || 'unknown',
    sample_category_id: Number(pose.sample_category_id ?? 255),
    camera_pos: (pose.camera_pos || [0, 0, 0]).map(Number),
    camera_forward: renderInfo.forward.toArray(),
    yaw_deg: Number(pose.yaw_deg ?? 0),
    pitch_deg: Number(pose.pitch_deg ?? 0),
    fov_y: renderInfo.fovY,
    aspect: renderInfo.aspect,
    width: renderInfo.width,
    height: renderInfo.height,
    visible_component_ids: ids,
    component_weights: weights,
    component_screen_ratios: ratios,
    visible_pixel_count: visiblePixelCount,
  };
}

async function runInstanceSampler(options = {}) {
  setStatus('loading runtime metadata...');
  const runtimeMeta = await loadJson('/assets/runtimeVisibilityMeta.json');
  const glbIndex = await loadJson('/assets/glbIndex.json');
  const { scene, objects, loaded, skipped, totalEntries, loadedGlbSubset } = await loadSceneObjects(options, runtimeMeta, glbIndex);

  const renderer = new THREE.WebGLRenderer({
    antialias: false,
    alpha: false,
    preserveDrawingBuffer: false,
    powerPreference: 'high-performance',
  });
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.autoClear = true;
  document.body.appendChild(renderer.domElement);
  const gl = renderer.getContext();
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  const webglVendor = debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
  const webglRenderer = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  console.log(`[sampler] WebGL backend vendor="${webglVendor}" renderer="${webglRenderer}"`);

  const rawPlan = Array.isArray(options.posePlan) && options.posePlan.length > 0
    ? options.posePlan
    : fallbackPosePlan(options);
  const poseStart = Number.isFinite(Number(options.poseStart)) && Number(options.poseStart) > 0
    ? Math.min(rawPlan.length, Math.floor(Number(options.poseStart)))
    : 0;
  const remaining = rawPlan.length - poseStart;
  const requestedCount = Number.isFinite(Number(options.poseCount)) && Number(options.poseCount) > 0
    ? Math.min(remaining, Math.floor(Number(options.poseCount)))
    : remaining;
  const maxPositions = Number.isFinite(Number(options.maxPositions)) && Number(options.maxPositions) > 0
    ? Math.min(requestedCount, Number(options.maxPositions))
    : requestedCount;
  const poses = rawPlan.slice(poseStart, poseStart + maxPositions);
  const startedAt = performance.now();
  let visibleSum = 0;
  let lastWidth = 0;
  let lastHeight = 0;

  for (let i = 0; i < poses.length; i += 1) {
    const pose = poses[i];
    const renderInfo = makeCameraForPose(pose, options);
    if (renderInfo.width !== lastWidth || renderInfo.height !== lastHeight) {
      renderer.setSize(renderInfo.width, renderInfo.height, false);
      lastWidth = renderInfo.width;
      lastHeight = renderInfo.height;
    }
    renderer.render(scene, renderInfo.camera);
    const pixels = new Uint8Array(renderInfo.width * renderInfo.height * 4);
    gl.readPixels(0, 0, renderInfo.width, renderInfo.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const counts = countVisiblePixels(pixels);
    visibleSum += counts.size;
    const record = buildRecord(pose, renderInfo, counts);
    if (typeof window.emitInstanceSample === 'function') await window.emitInstanceSample(record);
    if (typeof window.emitInstanceProgress === 'function') {
      await window.emitInstanceProgress({
        poseIndex: i + 1,
        poseCount: poses.length,
        visibleCount: counts.size,
        elapsedMs: performance.now() - startedAt,
      });
    }
    if ((i + 1) % 25 === 0 || i + 1 === poses.length) {
      setStatus(`sampling ${i + 1}/${poses.length}, lastVisible=${counts.size}, loadedGlbs=${loaded}/${totalEntries}`);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }

  renderer.dispose();
  for (const object of objects) disposeObject(object);
  return {
    sampled: poses.length,
    loadedGlbs: loaded,
    skippedGlbs: skipped,
    totalGlbs: totalEntries,
    loadedGlbSubset,
    avgVisible: visibleSum / Math.max(1, poses.length),
    elapsedMs: performance.now() - startedAt,
    gpuBackend: {
      api: 'WebGL',
      vendor: webglVendor,
      renderer: webglRenderer,
      version: gl.getParameter(gl.VERSION),
    },
  };
}

window.runInstanceSampler = runInstanceSampler;
setStatus('Color-ID sampler ready.');
