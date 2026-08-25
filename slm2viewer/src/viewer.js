import {
  AmbientLight,
  AnimationMixer,
  AxesHelper,
  Box3,
  Cache,
  DirectionalLight,
  GridHelper,
  HemisphereLight,
  LinearEncoding,
  LoaderUtils,
  LoadingManager,
  PMREMGenerator,
  PerspectiveCamera,
  OrthographicCamera,
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
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  FloatType,
  LinearFilter,
  RepeatWrapping
} from 'three';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { GUI } from 'dat.gui';
import { environments } from '../assets/environment/index.js';
import { createBackground } from '../lib/three-vignette.js';
import { SLM2Loader } from '../slm2/SLM2Loader';
import { keyboardMgr } from './keyboardMgr.js';
import { TrajectoryCollector } from './TrajectoryCollector.js';
import { TouchMoveController } from './TouchMoveController.js';
import { startupLog } from './startupTimeline.js';
import { FRONTEND_RENDER_FOV_Y_DEG } from './neuralPvsFovProtocol.js';

//import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass.js';
import { N8AOPostPass  } from 'n8ao';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

import { EffectComposer, RenderPass, EffectPass, SMAAEffect, SMAAPreset } from "postprocessing";
import { Camera } from 'three';

import { HDRJPGLoader } from '@monogrid/gainmap-js'

const MANAGER = new LoadingManager();
const THREE_PATH = `https://unpkg.com/three@0.${REVISION}.x`
const DRACO_LOADER = new DRACOLoader( MANAGER ).setDecoderPath( `${THREE_PATH}/examples/js/libs/draco/gltf/` );
const KTX2_LOADER = new KTX2Loader( MANAGER ).setTranscoderPath( `${THREE_PATH}/examples/js/libs/basis/` );

import { Loader3DTiles } from 'three-loader-3dtiles';

const MAP_NAMES = [
  'map',
  'aoMap',
  'emissiveMap',
  'glossinessMap',
  'metalnessMap',
  'normalMap',
  'roughnessMap',
  'specularMap',
];

Cache.enabled = true;
const VIEWER_CONFIG_CACHE_VERSION = 'pvs-v4-gpu-hkust-liteweb3d-20260825';
const WHITE = new Color(0xffffff);
const FRONTEND_RUNTIME_ASSET_ESTIMATE = {
  label: 'PVS V4 固定实例特征与查询网络，约 5.48 MB，不含按需 GLB',
};

export class Viewer 
{
  constructor (el, options) 
  {
    startupLog('viewer:constructor:start');
    this.el = el;
    this.options = options;

    this.lights = [];
    this.content = null;
    this.gui = null;

    this.DebugMode = false;

    this.state = 
    {
      environment: environments[1].name,
      background: true,
      wireframe: false,
      doubleSide: true,
      nearPlane: 10,
      farPlane: 100000,
      grid: false,

      // Lights
      addLights: true,
      textureEncoding: 'sRGB',
      directIntensity: 2.3,
      directColor: 0xffffff,
      ambientIntensity: 2.3,
      ambientColor: 0xffffff,
      bgColor1: '#ffffff',
      bgColor2: '#353588',
      envMapIntensity: 1.1,
      lightMapIntensity: 1.0,
      vertexColor: false,

      effectController: 
      {
        aoSamples: 16.0,
        denoiseSamples: 8.0,
        denoiseRadius: 12.0,
        aoRadius: 5.0,
        distanceFalloff: 7.0,
        screenSpaceRadius: false,
        halfRes: false,
        depthAwareUpsampling: true,
        intensity: 5.0,
        renderMode: "Combined",
        color: [0, 0, 0],
        colorMultiply: true
      }
    };

    var parseUrlParams = function()
    {
      var urlParams = window.location.href;
      var vars = {};
      var parts = urlParams.replace(/[?&]+([^=&]+)=([^&]*)/gi,
        function (m, key, value) {
          vars[key] = decodeURIComponent(value);
        });
        
      return vars;
    }
    this.paramJson = parseUrlParams();

    this.materials = [];

    this.prevTime = 0;

    this.stats = new Stats();
    this.stats.dom.height = '48px';
    [].forEach.call(this.stats.dom.children, (child) => (child.style.display = ''));
    startupLog('viewer:stats-ready');

    this.scene = new Scene();
    this.sceneGround = null;

    const fov = FRONTEND_RENDER_FOV_Y_DEG;
    this.activeCamera = new PerspectiveCamera(fov, el.clientWidth / el.clientHeight, this.state.nearPlane, this.state.farPlane);
    this.scene.add(this.activeCamera);

    this.debugLoadingMode = false;

    this.renderer = window.renderer = new WebGLRenderer();//{antialias: (this.DebugMode ? false: true)});
    this.renderer.physicallyCorrectLights = true;
    this.renderer.outputEncoding = sRGBEncoding;
    this.renderer.setClearColor( 0xdddddd );
    this.renderer.setPixelRatio( window.devicePixelRatio );
    this.renderer.setSize( el.clientWidth, el.clientHeight );
    //this.renderer.autoClear = false;
    startupLog('viewer:renderer-ready', {
      width: el.clientWidth,
      height: el.clientHeight,
      pixelRatio: window.devicePixelRatio,
    });

    this.pmremGenerator = new PMREMGenerator( this.renderer );
    this.pmremGenerator.compileEquirectangularShader();
    startupLog('viewer:pmrem-compiled');

    this.controls = new OrbitControls( this.activeCamera, this.renderer.domElement );
    this.controls.autoRotate = false;
    this.controls.autoRotateSpeed = -10;
    this.controls.screenSpacePanning = true;

    this.vignette = createBackground({
      aspect: this.activeCamera.aspect,
      grainScale: 0.001,
      colors: [this.state.bgColor1, this.state.bgColor2]
    });
    this.vignette.name = 'Vignette';
    this.vignette.renderOrder = -1;

    this.el.appendChild(this.renderer.domElement);
    startupLog('viewer:canvas-attached');

    this.cameraCtrl = null;
    this.cameraFolder = null;
    this.showGUI = true;

    if (this.paramJson['showGUI'])
    {
      this.showGUI = !(this.paramJson['showGUI'] == 'false');
    }

    this.hasLightMap = false;
    if (this.paramJson['lightmap'])
    {
      this.hasLightMap = !(this.paramJson['lightmap'] == 'false');
    }

    if (this.hasLightMap)
    {
      this.state.addLights = false;
      this.state.effectController.aoRadius = 2.0;
      this.state.effectController.intensity = 1.0;
    }

    this.animate = this.animate.bind(this);
    requestAnimationFrame( this.animate );
    window.addEventListener('resize', this.resize.bind(this), false);

    window.addEventListener('keydown', this.keydown.bind(this), false);
    window.addEventListener('keyup', this.keyup.bind(this), false);

    this.slm2Loader = new SLM2Loader();
    this.keyboardMgr = new keyboardMgr(this);
    this.touchMoveController = new TouchMoveController(this);
    this.trajectoryCollector = new TrajectoryCollector(this);
    startupLog('viewer:runtime-helpers-ready');
    this.runtimeDebugState = {
      cullingMode: this.slm2Loader.getCullingMode(),
      neuralRenderPolicy: this.slm2Loader.getNeuralRenderPolicy(),
      neuralDownloadPlanMode: this.slm2Loader.getNeuralDownloadPlanMode(),
      frontendAssetEstimate: FRONTEND_RUNTIME_ASSET_ESTIMATE.label,
      pvsModelVersion: '-',
      pvsModelSchema: '-',
      pvsModelWorkpoint: '-',
      pvsCullingSource: '未初始化',
      pvsBackend: '-',
      pvsReady: '否',
      pvsFallbackReason: '-',
      pvsPredictMs: '-',
      pvsInferenceMs: '-',
      pvsPostMs: '-',
      pvsPredictionAgeMs: '-',
      pvsCandidateSelection: '-',
      pvsGate: '-',
      pvsRawInstances: '0',
      pvsRawGlbs: '0',
      pvsImmediateGlbs: '0',
      pvsPrefetchGlbs: '0',
      pvsRenderInstances: '0',
      pvsRenderGlbs: '0',
      pvsActualRender: '-',
      pvsDownloadQueue: '0',
      pvsPrefetchQueue: '0',
      pvsActiveLoads: '0',
      pvsInflightLoads: '0',
      pvsPendingIntegrate: '0',
      pvsHttpMbps: '-',
      pvsHttpConcurrency: '-',
      pvsCacheSummary: '-',
      pvsRenderSummary: '-',
      pvsPredictionDebugSummary: '-',
    };
    this.cullingModeController = null;
    this.neuralRenderPolicyController = null;
    this.neuralDownloadPlanModeController = null;
    this.runtimeDebugControllers = [];
    this.runtimeDebugValueEls = {};
    this.runtimeDebugGuiLastUpdate = 0;
    this.runtimeDebugGuiIntervalMs = 500;
    this.queueDebugEnabled = this.paramJson['queueDebug'] === 'true' || this.paramJson['queueDebug'] === '1';
    this.queueDebugPanel = null;
    this.queueDebugContent = null;
    this.queueDebugLastUpdate = 0;
    this.queueDebugIntervalMs = 500;
    this.predictionDebugEnabled = this.paramJson['predictionDebug'] === 'true' || this.paramJson['predictionDebug'] === '1';
    this.predictionDebugForceShow = true;
    this.predictionDebugMaterial = new MeshBasicMaterial({
      color: 0xff2020,
      transparent: true,
      opacity: 0.82,
      depthTest: true,
      depthWrite: false,
      side: DoubleSide,
    });
    this.predictionDebugMaterial.name = 'NeuralPredictionDebugRed';
    this.predictionDebugTouched = new Map();
    this.predictionDebugFrozenMode = false;
    this.predictionDebugFrozenComponentIds = [];
    this.predictionDebugFrozenGlbIds = [];
    this.predictionDebugLastStats = {
      enabled: this.predictionDebugEnabled,
      predicted: 0,
      loaded: 0,
      marked: 0,
      missing: 0,
      attached: 0,
      restored: 0,
      frozen: false,
      snapshot: null,
    };
    this.predictionDebugToggleButton = null;
    this.predictionDebugFreezeButton = null;
    if (this.queueDebugEnabled)
    {
      this.createQueueDebugPanel();
    }

    this.usePostEffect = true;

    if (this.usePostEffect)
    {
      this.composer = new EffectComposer(this.renderer);
      this.composer.addPass(new RenderPass(this.scene, this.activeCamera));
      this.n8aopass = new N8AOPostPass (this.scene, this.activeCamera, el.clientWidth, el.clientHeight);
      this.composer.addPass(this.n8aopass);
      this.composer.addPass(new EffectPass(this.activeCamera, new SMAAEffect({preset: SMAAPreset.ULTRA})));
      startupLog('viewer:post-effects-ready');
  
      //const gammaCorrectionPass = new ShaderPass( GammaCorrectionShader )
      //this.composer.addPass(gammaCorrectionPass);
    }

    startupLog('viewer:constructor:end');
    //this.setup3DTiles();
  }

  setup3DTiles()
  {
    var scope = this;
    scope.tilesRuntime = null;

    new Promise( ( resolve, reject ) => {
      Loader3DTiles.load(
        {
        url: 'https://tile.googleapis.com/v1/3dtiles/root.json',
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight,
          devicePixelRatio: window.devicePixelRatio
        },
        options: 
        {
          googleApiKey: 'AIzaSyBUVr4yky9VLM-M4FJgD5xQvvDjux2WZvU',
          dracoDecoderPath: 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/draco',
          basisTranscoderPath: 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/basis',
          maximumScreenSpaceError: 48
        }
      }
    ).then(( result ) => {
      
      if (scope.DebugMode) console.log(result);

      const {model, runtime} = result
      scope.tilesRuntime = runtime
      scope.scene.add(model);

      // To HKUST(GZ)
      scope.tilesRuntime.orientToGeocoord({
        lat: Number(22.8912), 
        long: Number(113.4772), 
        height: Number(100)
      });

      scope.activeCamera.translateY(1000);
      scope.controls.update();
    });
    });
  }

  keydown(keyEvent) 
  {
    //console.log(keyEvent);
    // if (keyEvent.key == 'e')
    // {
      
    // }
    // else if (keyEvent.key == ' ')
    // {
    //   // this.clearLoadedForDebug();

    //   // this.sceneCulling();
    // }
    // else if (keyEvent.key == 't')
    // {
    //   this.loadingTimescale = (this.loadingTimescale == 1 ? 200 : 1);
    // }
    // else 
    if (keyEvent.key == '1')
    {
      console.log(
        '\"cameraPostion\": [' + this.activeCamera.position.x.toFixed(2) + ',' + this.activeCamera.position.y.toFixed(2) + ',' + this.activeCamera.position.z.toFixed(2) + '],\n' + 
        '\"cameraTarget\": [' + this.controls.target.x.toFixed(2) + ',' + this.controls.target.y.toFixed(2) + ',' + this.controls.target.z.toFixed(2) + '],');
    }
  }

  keyup(keyEvent) 
  {

  }

  animate(time) 
  {
    requestAnimationFrame( this.animate );

    const dt = (time - this.prevTime);
    this.prevTime = time;

    this.controls.update();
    this.stats.update();

    if (this.tilesRuntime) {
      this.tilesRuntime.update(dt, this.activeCamera);
    }

    this.updatePredictionDebugOverlay();

    // Reconcile downloads, instance visibility, and render-scene membership
    // before drawing this frame. Running it after render leaves the previous
    // frame's resident objects in the GPU submission for one extra frame.
    this.slm2Loader.update(dt, this.prevTime);

    this.render();

    this.keyboardMgr.update(dt);
    if (this.touchMoveController) {
      this.touchMoveController.update(dt);
    }

    if (this.trajectoryCollector) {
      this.trajectoryCollector.update(dt / 1000.0);
    }

    this.updateQueueDebugPanel(time);
    this.updateRuntimeDebugGui(time);
  }

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
    const actualRender = neural.actualRender || {};
    const scheduler = neural.priorityScheduler || {};
    const candidateSelection = neural.predictTimings?.candidateSelection || scheduler.candidateSelection || {};
    const resourceWS = neural.resourceWS || {};
    const frozenInspect = neural.frozenInspect || {};
    const httpAdaptive = load.httpAdaptive || {};
    const initTimings = neural.initTimings || {};
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
      `neural=${neural.enabled} transport=${neural.resourceTransport || 'http'} ws=${neural.resourcesWSConfigured ? (neural.resourcesWSConnected ? 'connected' : 'pending') : 'none'} renderPolicy=${neural.renderPolicy} cpu=${neural.cpuPerfMode}`,
      `wsPool ready=${resourceWS.readyConnections || 0}/${resourceWS.configuredConnections || 0} batch=${resourceWS.inFlightBatches || 0} models=${resourceWS.inFlightModels || 0} parse=${resourceWS.pendingParseCount || 0}/${resourceWS.activeParseCount || 0} parseMB=${fmt(resourceWS.pendingParseMB, 1)}`,
      `wsPool avg=${fmt(resourceWS.avgBatchLatencyMs, 1)}ms last=${fmt(resourceWS.lastBatchLatencyMs, 1)}ms mbps=${fmt(resourceWS.throughputMBps, 2)} retry=${resourceWS.retryCount || 0} fallback=${resourceWS.fallbackHttpCount || 0} timeout=${resourceWS.timeoutCount || resourceWS.timedOutBatches || 0} stale=${resourceWS.staleDropCount || 0}`,
      `httpAdaptive enabled=${httpAdaptive.enabled !== false} now=${httpAdaptive.current || '-'} prefetch=${httpAdaptive.prefetchCurrent || '-'} mbps=${fmt(httpAdaptive.ewmaMbps, 1)} downlink=${fmt(httpAdaptive.downlinkMbps, 1)} load=${fmt(httpAdaptive.ewmaLoadMs, 0)}ms reason=${httpAdaptive.lastReason || '-'}`,
      `startup sceneWeb=${fmt(startup.sceneWebMs, 0)}ms glbIndex=${fmt(startup.glbIndexMs, 0)} runtimeMeta=${fmt(startup.runtimeVisibilityMetaMs, 0)} material=${fmt(startup.materialConfigMs, 0)}ms`,
      `neuralInit total=${fmt(initTimings.totalMs, 0)}ms fetch=${fmt(initTimings.fetchMs, 0)} decode=${fmt(initTimings.decodeMs, 0)} parse=${fmt(initTimings.parseMs, 0)} gpu=${fmt(initTimings.gpuInitMs, 0)}ms`,
      `gate pos=${fmt(gate.positionDelta, 1)} angle=${fmt(gate.angleDeltaDeg, 1)}deg min=${fmt(gate.minIntervalMs, 0)}ms forceNext=${gate.forceNext}`,
      `prediction serial=${visibility.serial || '-'} age=${predictionAge == null ? '-' : fmt(predictionAge, 0) + 'ms'} latency=${fmt(visibility.latencyMs)}ms`,
      `candidate source=${candidateSelection.source || '-'} count=${candidateSelection.candidateCount ?? '-'} cells=${candidateSelection.queryCellCount ?? '-'} indexed=${candidateSelection.indexedInstanceCount ?? '-'} overflow=${candidateSelection.overflowInstanceCount ?? '-'}`,
      `modelBackGlb=${visibility.rawGlbCount || 0} loadNow=${notes.loadNowGlbCount != null ? notes.loadNowGlbCount : '-'} currentFrustumGlb=${notes.renderCandidateGlbCount != null ? notes.renderCandidateGlbCount : '-'} residentVisible=${notes.renderResidentCount != null ? notes.renderResidentCount : (visibility.visibleGlbCount != null ? visibility.visibleGlbCount : '-')} modelBackInst=${visibility.rawInstanceCount || 0}`,
      `download queue=${load.queueLength || 0} wanted=${load.wantedHashCount || 0} inflight=${load.inflightCount || 0} pendingParse=${load.pendingHashCount || 0} pendingIntegrate=${load.pendingSceneInsertions || 0}`,
      `prefetch=${load.prefetchQueueLength || 0} activeLoads=${load.activeDirectLoadCount || 0} integrated+=${load.lastIntegratedCount || 0} batch=${fmt(load.lastBatchMs)}ms tex=${fmt(load.lastTextureTaskMs)}ms`,
      `cache total=${cache.totalNums || 0} visible=${cache.visibleNums || 0} invisible=${cache.invisibleNums || 0} inSceneHidden=${cache.inSceneNums || 0} mem=${fmt((cache.memoryUsedKB || 0) / 1024, 0)}MB`,
      `render working=${render.workingSetSize || 0} activeEval=${render.activeEvaluationSize || 0} evaluated=${render.evaluatedCount || 0} visible=${render.visibleCount || 0} skipped=${render.skippedByGate}`,
      `actual visibleMesh=${actualRender.visibleMeshCount || 0}/${actualRender.meshCount || 0} instancedMesh=${actualRender.visibleInstancedMeshCount || 0}/${actualRender.instancedMeshCount || 0} drawnInst=${actualRender.drawnInstanceCount || 0} tri≈${fmt(actualRender.visibleTriangleEstimate, 0)}`,
      `render delta=+${render.addedCount || 0}/-${render.removedCount || 0} evaluated=${render.evaluatedCount || 0} attach=${render.attachedCount || 0} detach=${render.detachedCount || 0} renderMs=${fmt(render.durationMs, 3)}`,
      `scheduler total=${scheduler.total || 0} now=${scheduler.visibleNow || 0} prefetch=${scheduler.prefetch || 0} skipped=${scheduler.skipped || 0} tested=${scheduler.testedComponents || 0}`,
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

    if (backend.indexOf('aabb') >= 0 || backend.indexOf('fallback') >= 0 || fallbackReason)
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
    this.runtimeDebugState.pvsBackend = visibility.backend || neural.backend || neural.neuralBackend || '-';
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
      predictTimings.immediateGlbCount != null ? predictTimings.immediateGlbCount : notes.loadNowGlbCount
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

  setPredictionDebugEnabled(enabled)
  {
    this.predictionDebugEnabled = Boolean(enabled);
    if (this.predictionDebugToggleButton)
    {
      this.predictionDebugToggleButton.textContent = this.predictionDebugEnabled ? 'red on' : 'red off';
    }

    if (!this.predictionDebugEnabled)
    {
      this.clearPredictionDebugOverlay();
    }
    else
    {
      this.updatePredictionDebugOverlay(true);
    }
  }

  _getPredictionDebugComponentIds()
  {
    if (this.predictionDebugFrozenMode && Array.isArray(this.predictionDebugFrozenComponentIds))
    {
      return this.predictionDebugFrozenComponentIds.slice();
    }

    return this._getFinalPredictionDebugSnapshot().componentIds;
  }

  _getFinalPredictionDebugSnapshot()
  {
    if (!this.slm2Loader || typeof this.slm2Loader.getBenchmarkVisibilityIds !== 'function')
    {
      return {
        available: false,
        componentIds: [],
        glbIds: [],
      };
    }

    const ids = this.slm2Loader.getBenchmarkVisibilityIds();
    const currentMode = typeof this.slm2Loader.getCullingMode === 'function'
      ? this.slm2Loader.getCullingMode()
      : null;
    return {
      available: Boolean(ids && Number(ids.serial || 0) > 0 && ids.mode === 'neural' && currentMode === 'neural'),
      componentIds: ids && Array.isArray(ids.renderComponentIds)
        ? ids.renderComponentIds.slice()
        : [],
      glbIds: ids && Array.isArray(ids.renderGlbIds)
        ? ids.renderGlbIds.slice()
        : [],
    };
  }

  _getPredictionDebugInstancedSlotMap(hash)
  {
    const states = this.slm2Loader && this.slm2Loader.loadedInstancedVisibilityStatesByHash
      ? this.slm2Loader.loadedInstancedVisibilityStatesByHash
      : {};
    const state = states[hash];
    if (!state || state.disabled || !Array.isArray(state.activeIndices))
    {
      return null;
    }

    const slotMap = new Map();
    const activeIndices = state.activeIndices;
    for (let slot = 0; slot < activeIndices.length; ++slot)
    {
      const originalIndex = Number(activeIndices[slot]);
      if (Number.isInteger(originalIndex) && originalIndex >= 0)
      {
        slotMap.set(originalIndex, slot);
      }
    }
    return slotMap;
  }

  _decodePredictionDebugHash(globalGlbId)
  {
    if (!this.slm2Loader || typeof this.slm2Loader.decodeGlobalGlbInfo !== 'function')
    {
      return null;
    }

    const decoded = this.slm2Loader.decodeGlobalGlbInfo({
      id: globalGlbId,
      weight: 1,
      idMode: 'global-glb-priority',
    });
    return decoded ? decoded.hash : null;
  }

  _getPredictionDebugTargets()
  {
    const componentIds = this._getPredictionDebugComponentIds();
    const componentRecords = this.slm2Loader && Array.isArray(this.slm2Loader.componentVisibilityRecords)
      ? this.slm2Loader.componentVisibilityRecords
      : [];
    const instancedBindings = this.slm2Loader && this.slm2Loader.instancedVisibilityBindingByComponentId
      ? this.slm2Loader.instancedVisibilityBindingByComponentId
      : {};

    const instancedByHash = new Map();
    const regularHashes = new Set();

    for (let i = 0; i < componentIds.length; ++i)
    {
      const componentId = Number(componentIds[i]);
      if (!Number.isFinite(componentId))
      {
        continue;
      }

      const instancedBinding = instancedBindings[componentId];
      if (instancedBinding && instancedBinding.hash)
      {
        let entries = instancedByHash.get(instancedBinding.hash);
        if (!entries)
        {
          entries = [];
          instancedByHash.set(instancedBinding.hash, entries);
        }
        entries.push({
          componentId,
          instanceIndex: Number(instancedBinding.instanceIndex),
        });
        continue;
      }

      const record = componentRecords[componentId];
      if (record && record.glbHash)
      {
        regularHashes.add(record.glbHash);
      }
    }

    return {
      componentIds,
      instancedByHash,
      regularHashes,
    };
  }

  _restorePredictionDebugObject(hash, state)
  {
    if (!state)
    {
      return false;
    }

    if (state.type === 'instanced')
    {
      const meshStates = Array.isArray(state.meshStates) ? state.meshStates : [];
      for (const meshState of meshStates)
      {
        const mesh = meshState.mesh;
        if (!mesh) continue;

        if (meshState.hadInstanceColor)
        {
          for (let index = 0; index < meshState.originalColors.length; ++index)
          {
            mesh.setColorAt(index, meshState.originalColors[index]);
          }
        }
        else
        {
          mesh.instanceColor = null;
        }

        if (mesh.instanceColor)
        {
          mesh.instanceColor.needsUpdate = true;
        }
      }

      this.predictionDebugTouched.delete(hash);
      return true;
    }

    const object = state.object;
    if (object && typeof object.traverse === 'function')
    {
      object.traverse((node) => {
        if (!node || !node.isMesh || !node.userData)
        {
          return;
        }

        if (node.userData.__predictionDebugOriginalMaterial !== undefined)
        {
          node.material = node.userData.__predictionDebugOriginalMaterial;
          delete node.userData.__predictionDebugOriginalMaterial;
        }
      });

      object.visible = state.wasVisible;
      if (state.wasInScene === false && object.parent)
      {
        object.parent.remove(object);
      }
    }

    this.predictionDebugTouched.delete(hash);
    return true;
  }

  clearPredictionDebugOverlay()
  {
    let restored = 0;
    for (const [hash, state] of this.predictionDebugTouched.entries())
    {
      if (this._restorePredictionDebugObject(hash, state))
      {
        restored++;
      }
    }

    this.predictionDebugLastStats = Object.assign({}, this.predictionDebugLastStats, {
      enabled: this.predictionDebugEnabled,
      predicted: 0,
      loaded: 0,
      marked: 0,
      missing: 0,
      attached: 0,
      restored: restored,
      frozen: this.predictionDebugFrozenMode,
    });
  }

  updatePredictionDebugOverlay(force = false)
  {
    if (!this.predictionDebugEnabled && !force)
    {
      return;
    }

    if (!this.predictionDebugEnabled)
    {
      this.clearPredictionDebugOverlay();
      return;
    }

    const pool = this.slm2Loader && this.slm2Loader.modelCacheMgr
      ? this.slm2Loader.modelCacheMgr.objectsPool
      : null;
    if (!pool)
    {
      return;
    }

    const targets = this._getPredictionDebugTargets();
    const predictedComponentIds = new Set((targets.componentIds || []).map((id) => Number(id)).filter((id) => Number.isFinite(id)));
    const predictedRegularHashes = targets.regularHashes || new Set();
    const predictedInstancedByHash = targets.instancedByHash || new Map();
    const activeKeys = new Set();
    for (const hash of predictedRegularHashes)
    {
      activeKeys.add(`hash:${hash}`);
    }
    for (const hash of predictedInstancedByHash.keys())
    {
      activeKeys.add(`instanced:${hash}`);
    }

    let restored = 0;
    for (const [key, state] of Array.from(this.predictionDebugTouched.entries()))
    {
      const hash = key.startsWith('instanced:') ? key.slice(10) : key.startsWith('hash:') ? key.slice(5) : key;
      if (!activeKeys.has(key) || !pool[hash] || !pool[hash].meshObject)
      {
        if (this._restorePredictionDebugObject(key, state))
        {
          restored++;
        }
      }
    }

    let loaded = 0;
    let marked = 0;
    let attached = 0;

    for (const [hash, entries] of predictedInstancedByHash.entries())
    {
      const item = pool[hash];
      if (!item || !item.meshObject)
      {
        continue;
      }

      loaded++;
      const touchKey = `instanced:${hash}`;
      let state = this.predictionDebugTouched.get(touchKey);
      if (!state)
      {
        const meshStates = [];
        item.meshObject.traverse((node) => {
          if (!node || !node.isInstancedMesh)
          {
            return;
          }
          const hadInstanceColor = Boolean(node.instanceColor);
          const originalColors = [];
          if (hadInstanceColor)
          {
            const color = new Color();
            for (let index = 0; index < node.count; ++index)
            {
              node.getColorAt(index, color);
              originalColors.push(color.clone());
            }
          }
          meshStates.push({
            mesh: node,
            hadInstanceColor,
            originalColors,
          });
        });

        state = {
          type: 'instanced',
          meshStates,
        };
        this.predictionDebugTouched.set(touchKey, state);
      }

      const activeCount = entries.length;
      const slotMap = this._getPredictionDebugInstancedSlotMap(hash);
      for (const meshState of state.meshStates)
      {
        const mesh = meshState.mesh;
        if (!mesh)
        {
          continue;
        }

        if (meshState.hadInstanceColor)
        {
          for (let index = 0; index < meshState.originalColors.length; ++index)
          {
            mesh.setColorAt(index, meshState.originalColors[index]);
          }
        }
        else
        {
          for (let index = 0; index < mesh.count; ++index)
          {
            mesh.setColorAt(index, WHITE);
          }
        }

        for (let entryIndex = 0; entryIndex < activeCount; ++entryIndex)
        {
          const instanceIndex = Number(entries[entryIndex] && entries[entryIndex].instanceIndex);
          const slotIndex = slotMap ? slotMap.get(instanceIndex) : instanceIndex;
          if (!Number.isInteger(slotIndex) || slotIndex < 0 || slotIndex >= mesh.count)
          {
            continue;
          }
          mesh.setColorAt(slotIndex, this.predictionDebugMaterial.color);
          marked++;
        }
        if (mesh.instanceColor)
        {
          mesh.instanceColor.needsUpdate = true;
        }
      }
    }

    for (const hash of predictedRegularHashes)
    {
      const item = pool[hash];
      if (!item || !item.meshObject)
      {
        continue;
      }

      loaded++;
      const touchKey = `hash:${hash}`;
      let state = this.predictionDebugTouched.get(touchKey);
      if (!state)
      {
        state = {
          type: 'object',
          object: item.meshObject,
          wasVisible: item.meshObject.visible,
          wasInScene: Boolean(item.isInScene),
        };
        this.predictionDebugTouched.set(touchKey, state);

        item.meshObject.traverse((node) => {
          if (!node || !node.isMesh || !node.userData)
          {
            return;
          }
          if (node.userData.__predictionDebugOriginalMaterial === undefined)
          {
            node.userData.__predictionDebugOriginalMaterial = node.material;
          }
          node.material = this.predictionDebugMaterial;
          marked++;
        });
      }
      else
      {
        item.meshObject.traverse((node) => {
          if (node && node.isMesh)
          {
            marked++;
          }
        });
      }

      if (this.predictionDebugForceShow)
      {
        if (!item.isInScene && this.slm2Loader.rootScene)
        {
          this.slm2Loader.rootScene.add(item.meshObject);
          attached++;
        }
        item.meshObject.visible = true;
      }
    }

    this.predictionDebugLastStats = {
      enabled: this.predictionDebugEnabled,
      frozen: this.predictionDebugFrozenMode,
      predicted: predictedComponentIds.size,
      loaded: loaded,
      marked: marked,
      missing: Math.max(0, predictedComponentIds.size - marked),
      attached: attached,
      restored: restored,
      snapshot: this.predictionDebugLastStats ? this.predictionDebugLastStats.snapshot : null,
    };
  }

  _getTopDownBounds()
  {
    if (this.sceneBounds && this.sceneBounds.center && this.sceneBounds.size)
    {
      const center = new Vector3(
        this.sceneBounds.center[0],
        this.sceneBounds.center[1],
        this.sceneBounds.center[2]
      );
      const size = new Vector3(
        Math.max(1, this.sceneBounds.size[0]),
        Math.max(1, this.sceneBounds.size[1]),
        Math.max(1, this.sceneBounds.size[2])
      );
      return { center, size };
    }

    const box = new Box3().setFromObject(this.scene);
    if (!Number.isFinite(box.min.x) || !Number.isFinite(box.max.x))
    {
      return {
        center: new Vector3(0, 0, 0),
        size: new Vector3(1000, 1000, 1000),
      };
    }

    const center = new Vector3();
    const size = new Vector3();
    box.getCenter(center);
    box.getSize(size);
    return { center, size };
  }

  capturePredictionDebugTopDown()
  {
    const wasEnabled = this.predictionDebugEnabled;
    if (!wasEnabled)
    {
      this.setPredictionDebugEnabled(true);
    }
    else
    {
      this.updatePredictionDebugOverlay(true);
    }

    const bounds = this._getTopDownBounds();
    const aspect = this.renderer.domElement.width / Math.max(1, this.renderer.domElement.height);
    const horizontal = Math.max(bounds.size.x, bounds.size.z, 1) * 0.62;
    const vertical = horizontal / Math.max(0.01, aspect);
    const height = Math.max(bounds.size.y * 4.0, Math.max(bounds.size.x, bounds.size.z) * 1.2, 1000);
    const topCamera = new OrthographicCamera(
      -horizontal,
      horizontal,
      vertical,
      -vertical,
      1,
      height * 2.5
    );
    topCamera.position.set(bounds.center.x, bounds.center.y + height, bounds.center.z);
    topCamera.up.set(0, 0, -1);
    topCamera.lookAt(bounds.center);
    topCamera.updateMatrixWorld();
    topCamera.updateProjectionMatrix();

    const previousAutoClear = this.renderer.autoClear;
    this.renderer.autoClear = true;
    this.renderer.clear();
    this.renderer.render(this.scene, topCamera);

    let snapshot = null;
    try
    {
      const dataUrl = this.renderer.domElement.toDataURL('image/png');
      const link = document.createElement('a');
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      link.download = `neural_predicted_topdown_${stamp}.png`;
      link.href = dataUrl;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      snapshot = `saved ${stamp}`;
    }
    catch (err)
    {
      console.warn('[Viewer] Top-down prediction snapshot failed.', err);
      snapshot = 'failed';
    }
    finally
    {
      this.renderer.autoClear = previousAutoClear;
      this.render();
      if (!wasEnabled)
      {
        this.setPredictionDebugEnabled(false);
      }
    }

    this.predictionDebugLastStats = Object.assign({}, this.predictionDebugLastStats, {
      snapshot: snapshot,
    });
    this.updateQueueDebugPanel(performance.now() + this.queueDebugIntervalMs);
  }

  toggleFrozenPredictionInspectMode()
  {
    if (this.predictionDebugFrozenMode)
    {
      this.predictionDebugFrozenMode = false;
      this.predictionDebugFrozenComponentIds = [];
      this.predictionDebugFrozenGlbIds = [];
      if (this.predictionDebugFreezeButton)
      {
        this.predictionDebugFreezeButton.textContent = 'freeze view';
      }
      if (this.slm2Loader && typeof this.slm2Loader.stopFrozenPredictionInspectSession === 'function')
      {
        this.slm2Loader.stopFrozenPredictionInspectSession();
      }
      this.updatePredictionDebugOverlay(true);
      return;
    }

    const snapshot = this._getFinalPredictionDebugSnapshot();
    if (!snapshot.available)
    {
      console.warn('[Viewer] Cannot freeze the current render-frustum result before the first visibility result is available.');
      return;
    }

    this.predictionDebugFrozenMode = true;
    this.predictionDebugFrozenComponentIds = snapshot.componentIds.slice();
    this.predictionDebugFrozenGlbIds = snapshot.glbIds.slice();
    if (this.predictionDebugFreezeButton)
    {
      this.predictionDebugFreezeButton.textContent = 'unfreeze';
    }
    this.setPredictionDebugEnabled(true);
    if (this.slm2Loader && typeof this.slm2Loader.startFrozenPredictionInspectSession === 'function')
    {
      this.slm2Loader.startFrozenPredictionInspectSession(
        this.predictionDebugFrozenComponentIds,
        this.predictionDebugFrozenGlbIds
      );
    }
    this.updatePredictionDebugOverlay(true);
  }

  render()
  {
    

    if (this.usePostEffect)
    {
      this.composer.render();
    }
    else
    {
      this.renderer.clear();
      this.renderer.render( this.scene, this.activeCamera );
    }
  }

  resize() 
  {
    const {clientHeight, clientWidth} = this.el.parentElement;

    this.activeCamera.aspect = clientWidth / clientHeight;
    this.activeCamera.updateProjectionMatrix();

    this.renderer.setSize(clientWidth, clientHeight);

    this.slm2Loader.setSize(clientWidth, clientHeight);

    this.composer.setSize(clientWidth, clientHeight);

    if (this.DebugMode) console.log(clientWidth, clientHeight);
  }

  load() 
  {
    startupLog('viewer:load:start');
    var scope = this;

    var fileLoader = new FileLoader();
    startupLog('viewer:config-request');
		fileLoader.load("assets/config.json?v=" + encodeURIComponent(VIEWER_CONFIG_CACHE_VERSION), function(data) 
    {
			var config = JSON.parse(data);
      startupLog('viewer:config-loaded', {
        sceneCount: config && config.scenes ? Object.keys(config.scenes).length : 0,
      });

      scope.sceneConfigs = config.scenes;

      var defaultSceneName = config.defaultScene || 'hkust-v3';
      var activeSceneName = scope.paramJson['scene'] ? scope.paramJson['scene'] : defaultSceneName;

      document.title = activeSceneName.toUpperCase();

      scope.activeScene = scope.sceneConfigs[activeSceneName];
      if (scope.activeScene == undefined)
      {
        activeSceneName = defaultSceneName;
        scope.activeScene = scope.sceneConfigs[activeSceneName] || scope.sceneConfigs['default_config'];
      }

      scope.addConfiguredViewpoints();

      scope.setContent(new Object3D());
      startupLog('viewer:setContent-done');

      var _materialLoadedCallback = scope.updateEnvAndLightMap.bind(scope);

      var sceneLoadMethod = null;
      var sceneLoadConfig = null;

      var staticMode = scope.paramJson['static'];

      if (staticMode == 'true')
      {
        sceneLoadMethod = 'loadStatic';
        sceneLoadConfig = "http://127.0.0.1:8080/sceneWeb.json";
      }
      else
      {
        sceneLoadMethod = 'load';
        sceneLoadConfig = {
          name: activeSceneName,
          lbServer: config.lbs,
          loader: scope.activeScene.loaderConfig,
        };
      }

      startupLog('viewer:slm2Loader-load-call', { method: sceneLoadMethod });
      var rootScene = scope.slm2Loader[sceneLoadMethod](sceneLoadConfig, scope.renderer, scope.activeCamera, 
      {
        materialLoadedCallback: _materialLoadedCallback,
        paramJson: scope.paramJson,
      }, function(config)
      {
        startupLog('viewer:slm2Loader-callback:start');
        if (config['hasLightmap'])
        {
          scope.hasLightMap = config['hasLightmap'];
        }
    
        if (scope.hasLightMap)
        {
          scope.state.addLights = false;
          scope.state.effectController.aoRadius = 2.0;
          scope.state.effectController.intensity = 1.0;

          scope.updateLights();
          scope.updateSSAO();
        }

        if (config.bounds && config.bounds.center && config.bounds.size) {
            scope.sceneBounds = config.bounds;
            scope.updateSceneGround(config.bounds);
        }

        if (scope.paramJson['autoCamera'] && config.bounds && config.bounds.center && config.bounds.size)
        {
          var cameraOffset = [0, config.bounds.size[1] * 0.1, config.bounds.size[2] * 0.5];

          var cameraPosition = new Vector3(config.bounds.center[0], config.bounds.center[1], config.bounds.center[2] + config.bounds.size[2] * 0.5 + cameraOffset[2]);
          var cameraTarget = new Vector3(config.bounds.center[0], config.bounds.center[1], config.bounds.center[2]);
          
          scope.setCamera(cameraPosition, cameraTarget, Math.abs(cameraTarget.z - cameraPosition.z));
        }

        scope.addGUI();
        startupLog('viewer:slm2Loader-callback:end');
      });

      scope.scene.add(rootScene);
      startupLog('viewer:rootScene-attached');
		});
  }

  setContent(object)
  {
    this.clear();

    this.setCamera();

    this.scene.add(object);

    this.updateLights();
    this.updateEnvironment();
    this.updateTextureEncoding();
    this.updateDisplay();

    this.updateEnvAndLightMap();
  }

  setCamera(_cameraPosition, _cameraTarget, _cameraOffset)
  {
    var cameraPosition = _cameraPosition ? _cameraPosition : new Vector3(this.activeScene.cameraPostion[0], this.activeScene.cameraPostion[1], this.activeScene.cameraPostion[2]);
    var cameraTarget = _cameraTarget ? _cameraTarget : new Vector3(this.activeScene.cameraTarget[0], this.activeScene.cameraTarget[1], this.activeScene.cameraTarget[2]);
    var shouldUpdateProjection = false;

    if (this.activeScene && this.activeScene.cameraNearPlane != null)
    {
      var nearPlane = Number(this.activeScene.cameraNearPlane);
      if (Number.isFinite(nearPlane) && nearPlane > 0)
      {
        this.state.nearPlane = nearPlane;
        shouldUpdateProjection = true;
      }
    }

    if (this.activeScene && this.activeScene.cameraFarPlane != null)
    {
      var farPlane = Number(this.activeScene.cameraFarPlane);
      if (Number.isFinite(farPlane) && farPlane > 0)
      {
        this.state.farPlane = farPlane;
        shouldUpdateProjection = true;
      }
    }

    if (_cameraOffset)
    {
      this.state.farPlane = Math.max(500, Math.abs(_cameraOffset) * 2.0);
      this.state.nearPlane = Math.max(0.01, this.state.farPlane * 0.00001);

      if (this.DebugMode) console.log(this.state);

      this.updateCamera();
    }
    else if (shouldUpdateProjection)
    {
      this.updateCamera();
    }

    this.activeCamera.position.copy(cameraPosition);
    this.activeCamera.lookAt(cameraTarget);
    this.controls.target.copy(cameraTarget);
    this.controls.update();
  }

  addConfiguredViewpoints()
  {
    const viewpoints = Array.isArray(this.activeScene && this.activeScene.viewpoints)
      ? this.activeScene.viewpoints
      : [];
    if (viewpoints.length === 0)
    {
      return;
    }

    if (this.configuredViewpointPanel)
    {
      this.configuredViewpointPanel.remove();
    }

    const panel = document.createElement('nav');
    panel.className = 'configured-viewpoint-panel';
    panel.setAttribute('aria-label', '场景观察点');

    viewpoints.forEach((viewpoint, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      const label = String(viewpoint.label || viewpoint.name || `视点 ${index + 1}`);
      button.textContent = label;
      button.title = `切换到${label}`;
      button.setAttribute('aria-label', `切换到${label}`);
      button.addEventListener('click', () => {
        const position = viewpoint.position || viewpoint.cameraPosition;
        const target = viewpoint.target || viewpoint.cameraTarget;
        if (!Array.isArray(position) || position.length < 3 || !Array.isArray(target) || target.length < 3)
        {
          return;
        }
        this.setCamera(
          new Vector3(Number(position[0]), Number(position[1]), Number(position[2])),
          new Vector3(Number(target[0]), Number(target[1]), Number(target[2])),
        );
        const groupId = viewpoint.neuralAssetGroupId != null
          ? viewpoint.neuralAssetGroupId
          : (viewpoint.assetGroupId != null ? viewpoint.assetGroupId : null);
        const forceCulling = () =>
        {
          if (this.slm2Loader && typeof this.slm2Loader.forceSceneCullingNow === 'function')
          {
            this.slm2Loader.forceSceneCullingNow({
              forceNeural: this.slm2Loader.getCullingMode() === 'neural',
            });
          }
        };
        if (groupId != null && this.slm2Loader && typeof this.slm2Loader.switchNeuralAssetGroup === 'function')
        {
          this.slm2Loader.switchNeuralAssetGroup(groupId).then(forceCulling);
        }
        else
        {
          forceCulling();
        }
      });
      panel.appendChild(button);
    });

    this.el.appendChild(panel);
    this.configuredViewpointPanel = panel;
  }

  getSceneGroundConfig()
  {
    if (!this.activeScene)
    {
      return null;
    }

    var ground = this.activeScene.groundPlane;
    if (ground && ground.enabled)
    {
      return ground;
    }

    return null;
  }

  removeSceneGround()
  {
    if (!this.sceneGround)
    {
      return;
    }

    this.scene.remove(this.sceneGround);
    this.sceneGround.traverse((node) =>
    {
      if (node.geometry && typeof node.geometry.dispose === 'function')
      {
        node.geometry.dispose();
      }
      if (node.material)
      {
        var materials = Array.isArray(node.material) ? node.material : [node.material];
        materials.forEach((material) =>
        {
          if (material && typeof material.dispose === 'function')
          {
            material.dispose();
          }
        });
      }
    });
    this.sceneGround = null;
  }

  updateSceneGround(bounds)
  {
    this.removeSceneGround();

    var groundConfig = this.getSceneGroundConfig();
    if (!groundConfig || !bounds || !bounds.center || !bounds.size)
    {
      return;
    }

    var center = bounds.center;
    var size = bounds.size;
    var extentX = Math.max(1, Number(size[0]) || 1);
    var extentZ = Math.max(1, Number(size[2]) || 1);
    var sizeMultiplier = Number.isFinite(Number(groundConfig.sizeMultiplier)) ? Number(groundConfig.sizeMultiplier) : 1.2;
    var groundSize = Math.max(extentX, extentZ) * sizeMultiplier;
    var bottomY = Number(center[1]) - (Number(size[1]) * 0.5);
    var yOffset = Number.isFinite(Number(groundConfig.yOffset)) ? Number(groundConfig.yOffset) : -0.03;
    var groundY = bottomY + yOffset;

    var root = new Object3D();
    root.name = 'SceneGround';

    var material = new MeshBasicMaterial({
      color: new Color(groundConfig.color || '#6f756d'),
      transparent: true,
      opacity: Number.isFinite(Number(groundConfig.opacity)) ? Number(groundConfig.opacity) : 0.72,
      side: DoubleSide,
      depthWrite: true,
    });
    var plane = new Mesh(new PlaneGeometry(groundSize, groundSize), material);
    plane.name = 'SceneGroundPlane';
    plane.rotation.x = -Math.PI * 0.5;
    plane.position.set(Number(center[0]) || 0, groundY, Number(center[2]) || 0);
    plane.renderOrder = -2;
    root.add(plane);

    var divisions = Math.max(2, Math.floor(Number(groundConfig.gridDivisions) || 64));
    var grid = new GridHelper(
      groundSize,
      divisions,
      new Color(groundConfig.gridCenterColor || '#d4dccd'),
      new Color(groundConfig.gridColor || '#aeb5a8')
    );
    grid.name = 'SceneGroundGrid';
    grid.position.set(Number(center[0]) || 0, groundY + 0.01, Number(center[2]) || 0);
    var gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
    gridMaterials.forEach((gridMaterial) =>
    {
      if (!gridMaterial) return;
      gridMaterial.transparent = true;
      gridMaterial.opacity = 0.45;
    });
    grid.renderOrder = -1;
    root.add(grid);

    this.sceneGround = root;
    this.scene.add(root);
  }

  setCullingMode(value)
  {
    if (this.predictionDebugFrozenMode)
    {
      this.toggleFrozenPredictionInspectMode();
    }
    var applied = this.slm2Loader
      ? this.slm2Loader.setCullingMode(value)
      : (String(value).toLowerCase() === 'frustum' ? 'frustum' : 'neural');
    this.runtimeDebugState.cullingMode = applied;

    if (this.cullingModeController)
    {
      this.cullingModeController.updateDisplay();
    }
    this.updateRuntimeDebugGui(performance.now(), true);
    return applied;
  }

  setNeuralRenderPolicy(value)
  {
    var applied = this.slm2Loader ? this.slm2Loader.setNeuralRenderPolicy(value) : value;
    this.runtimeDebugState.neuralRenderPolicy = applied;

    if (this.slm2Loader && applied === 'culled' && typeof this.slm2Loader.forceSceneCullingNow === 'function')
    {
      this.slm2Loader.forceSceneCullingNow();
    }

    if (this.neuralRenderPolicyController)
    {
      this.neuralRenderPolicyController.updateDisplay();
    }

    return applied;
  }

  setNeuralDownloadPlanMode(value)
  {
    var applied = this.slm2Loader ? this.slm2Loader.setNeuralDownloadPlanMode(value) : (value === 'raw-visible' ? 'raw-visible' : 'viewcell-priority');
    this.runtimeDebugState.neuralDownloadPlanMode = applied;

    if (this.neuralDownloadPlanModeController)
    {
      this.neuralDownloadPlanModeController.updateDisplay();
    }

    if (this.slm2Loader && typeof this.slm2Loader.forceSceneCullingNow === 'function')
    {
      this.slm2Loader.forceSceneCullingNow();
    }

    return applied;
  }

  updateTextureEncoding () 
  {
    const encoding = this.state.textureEncoding === 'sRGB'
      ? sRGBEncoding
      : LinearEncoding;
    traverseMaterials(this.content, (material) => {
      if (material.map) material.map.encoding = encoding;
      if (material.emissiveMap) material.emissiveMap.encoding = encoding;
      if (material.map || material.emissiveMap) material.needsUpdate = true;
    });
  }

  updateEnvAndLightMap()
  {
    if (this.slm2Loader)
    {
      var scope = this;

      var testLM = false;

      if (testLM)
      {
        var lightmapUrl = 'assets/Lightmap-0_comp_light.hdr';
        new HDRLoader().setDataType(FloatType).load(
            lightmapUrl,
            function(lmTexture) 
            {
              // 注意：
              // 1. 必须设置贴图所对应的UV通道，不然默认会从uv0来进行采样；
              // 2. 需要对外部加载的lightmap贴图进行flipY为false的设置；
              lmTexture.channel = 1;
              lmTexture.flipY = false;
              
              lmTexture.type = FloatType;
              lmTexture.minFilter = LinearFilter;
              lmTexture.magFilter = LinearFilter;
              lmTexture.wrapS = RepeatWrapping;
              lmTexture.wrapT = RepeatWrapping;
              lmTexture.needsUpdate = true;

              var mtls = scope.slm2Loader.getMaterials();

              for (var i = 0; i < mtls.length; ++i)
              {
                mtls[i].envMapIntensity = 0;
                mtls[i].lightMap = lmTexture;
                mtls[i].lightMapIntensity = 3.0;
                mtls[i].aoMap = null;
              }
            }
          );
      }
      else
      {
        var mtls = this.slm2Loader.getMaterials();

        for (var i = 0; i < mtls.length; ++i)
        {
          mtls[i].envMapIntensity = this.state.envMapIntensity;
          mtls[i].envMap = this.scene.environment;

          mtls[i].lightMapIntensity = this.state.lightMapIntensity;
        }
      }
      
    }
  }

  updateLights () {
    const state = this.state;
    const lights = this.lights;

    if (state.addLights && !lights.length) 
    {
      this.addLights();
    }
    else if (!state.addLights && lights.length) 
    {
      this.removeLights();
    }

    if (lights.length >= 1) 
    {
      lights[0].intensity = state.directIntensity;
      lights[0].color.setHex(state.directColor);

      lights[1].intensity = state.ambientIntensity;
      lights[1].color.setHex(state.ambientColor);
    }
  }

  updateMaterials()
  {
    this.content.traverse((node) => {
      if (node.isMesh) {
        if (node.material)
        {
          node.material.vertexColors = this.state.vertexColor;
        }
      }
    });
  }

  addLights () 
  {
    const state = this.state;

    const light1  = new DirectionalLight(state.directColor, state.directIntensity);
    light1.position.set(0.5, 1, 0.866); // ~60º
    light1.name = 'main_light';
    this.scene.add(light1);
    this.lights.push(light1);

    const light2  = new AmbientLight(state.ambientColor, state.ambientIntensity);
    light2.position.set(0, 0, 0);
    light2.name = 'ambient_light';
    this.scene.add(light2);
    this.lights.push(light2);
  }

  removeLights () 
  {
    this.lights.forEach((light) => light.removeFromParent());
    this.lights.length = 0;
  }

  updateEnvironment () 
  {
    const environment = environments.filter((entry) => entry.name === this.state.environment)[0];

    this.getCubeMapTexture( environment ).then(( { envMap } ) => {
      this.scene.environment = envMap;
      this.scene.background = this.state.background ? envMap : null;
    });
  }

  getCubeMapTexture ( environment ) 
  {
    var scope = this;
    const { path } = environment;

    // no envmap
    if ( ! path ) return Promise.resolve( { envMap: null } );

    return new Promise( ( resolve, reject ) => {

      if (path.endsWith('.hdr'))
      {
        new HDRLoader().load( path, ( texture ) => {

          const envMap = this.pmremGenerator.fromEquirectangular( texture ).texture;
          this.pmremGenerator.dispose();

          resolve( { envMap } );

        }, undefined, reject );
      }
      else
      {
        // HDR encode to JPG
        //https://github.com/MONOGRID/gainmap-js
        new HDRJPGLoader(scope.renderer).load(path, (result) =>
        {
          const envMap = this.pmremGenerator.fromEquirectangular(result.renderTarget.texture).texture;
          this.pmremGenerator.dispose();

          resolve( { envMap } );
        } )
      }
    });
  }

  updateCamera()
  {
    this.activeCamera.near = this.state.nearPlane;
    this.activeCamera.far = this.state.farPlane;

    this.activeCamera.updateProjectionMatrix();
  }

  updateDisplay () 
  {
    if (this.slm2Loader)
    {
      var mtls = this.slm2Loader.getMaterials();

      for (var i = 0; i < mtls.length; ++i)
      {
        mtls[i].wireframe = this.state.wireframe;

        mtls[i].side = this.state.doubleSide ? DoubleSide: FrontSide;
      }
    }
  }

  updateBackground () 
  {
    this.vignette.style({colors: [this.state.bgColor1, this.state.bgColor2]});
  }

  updateSSAO()
  {
    if (this.n8aopass)
    {
      this.n8aopass.configuration.aoRadius = this.state.effectController.aoRadius;
      this.n8aopass.configuration.distanceFalloff = this.state.effectController.distanceFalloff;
      this.n8aopass.configuration.intensity = this.state.effectController.intensity;
      this.n8aopass.configuration.aoSamples = this.state.effectController.aoSamples;
      this.n8aopass.configuration.denoiseRadius = this.state.effectController.denoiseRadius;
      this.n8aopass.configuration.denoiseSamples = this.state.effectController.denoiseSamples;
      this.n8aopass.configuration.renderMode = ["Combined", "AO", "No AO", "Split", "Split AO"].indexOf(this.state.effectController.renderMode);
      this.n8aopass.configuration.color = new Color(this.state.effectController.color[0], this.state.effectController.color[1], this.state.effectController.color[2]);
      this.n8aopass.configuration.screenSpaceRadius = this.state.effectController.screenSpaceRadius;
      this.n8aopass.configuration.halfRes = this.state.effectController.halfRes;
      this.n8aopass.configuration.depthAwareUpsampling = this.state.effectController.depthAwareUpsampling;
      this.n8aopass.configuration.colorMultiply = this.state.effectController.colorMultiply;
    }
  }

  generateLoadList()
  {
    if (this.slm2Loader)
    {
      if (this.slm2Loader.schedulingStrategy == 'static')
      {
        window.alert("静态模式下不能生成加载列表");

        return;
      }
      
      this.slm2Loader.fetchCameraVisibilityList(function(result)
      {
        var saveLoadList = function(txtString, fileName) 
        {
          var link = document.createElement('a');
          link.style.display = 'none';
          document.body.appendChild(link); // Firefox workaround, see #6594

          function save(blob, filename){
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.click();
          }

          function saveString(text, filename){ save( new Blob([text], {type: 'text/plain'}), filename); }

          saveString( txtString, fileName);
        }

        saveLoadList(JSON.stringify(result), "initial.json");
      });
      
    }
  }

  addGUI () 
  {
    if (this.showGUI == false)
    {
      this.updateSSAO();
      
      return;
    }

    const gui = this.gui = new GUI({autoPlace: false, width: 430, hideable: true});

    // Display controls.
    const dispFolder = gui.addFolder('显示');
    const envBackgroundCtrl = dispFolder.add(this.state, 'background');
    envBackgroundCtrl.onChange(() => this.updateEnvironment());
    const wireframeCtrl = dispFolder.add(this.state, 'wireframe');
    wireframeCtrl.onChange(() => this.updateDisplay());

    const doublesideCtrl = dispFolder.add(this.state, 'doubleSide');
    doublesideCtrl.onChange(() => this.updateDisplay());

    const nearPlaneCtrl = dispFolder.add(this.state, 'nearPlane', 0.1, 20.0);
    const farPlaneCtrl = dispFolder.add(this.state, 'farPlane', 100, 100000);
    nearPlaneCtrl.onChange(() => this.updateCamera());
    farPlaneCtrl.onChange(() => this.updateCamera());

    dispFolder.add(this.controls, 'autoRotate');
    //dispFolder.add(this.controls, 'screenSpacePanning');
    const bgColor1Ctrl = dispFolder.addColor(this.state, 'bgColor1');
    const bgColor2Ctrl = dispFolder.addColor(this.state, 'bgColor2');
    bgColor1Ctrl.onChange(() => this.updateBackground());
    bgColor2Ctrl.onChange(() => this.updateBackground());

    // Lighting controls.
    const lightFolder = gui.addFolder('光照');
    const envMapCtrl = lightFolder.add(this.state, 'environment', environments.map((env) => env.name));
    envMapCtrl.onChange(() => this.updateEnvironment());
    [
      lightFolder.add(this.state, 'addLights').listen(),
      lightFolder.add(this.state, 'directIntensity', 0, 50),
      lightFolder.addColor(this.state, 'directColor'),
      lightFolder.add(this.state, 'ambientIntensity', 0, 50),
      lightFolder.addColor(this.state, 'ambientColor'),
    ].forEach((ctrl) => ctrl.onChange(() => this.updateLights()));

    [
      lightFolder.add(this.state, 'envMapIntensity', 0, 10),
    ].forEach((ctrl) => ctrl.onChange(() => this.updateEnvAndLightMap()));

    [
      lightFolder.add(this.state, 'lightMapIntensity', 0, 10),
    ].forEach((ctrl) => ctrl.onChange(() => this.updateEnvAndLightMap()));

    [
    lightFolder.add(this.state, 'vertexColor').listen()
    ].forEach((ctrl) => ctrl.onChange(() => this.updateMaterials()));

    // SSAO
    const effectFolder = gui.addFolder('AO');

    var simpleAOCtrl = !(this.paramJson['ssao'] != undefined && this.paramJson['ssao'] == 'true');

    if (simpleAOCtrl == false)
    {
      effectFolder.add(this.state.effectController, "aoSamples", 1.0, 64.0, 1.0).onChange(()=>this.updateSSAO());
      effectFolder.add(this.state.effectController, "denoiseSamples", 1.0, 64.0, 1.0).onChange(()=>this.updateSSAO());
      effectFolder.add(this.state.effectController, "denoiseRadius", 0.0, 24.0, 0.01).onChange(()=>this.updateSSAO());
      const aor = effectFolder.add(this.state.effectController, "aoRadius", 1.0, 10.0, 0.01).onChange(()=>this.updateSSAO());
      const df = effectFolder.add(this.state.effectController, "distanceFalloff", 0.0, 10.0, 0.01).onChange(()=>this.updateSSAO());
      effectFolder.add(this.state.effectController, "screenSpaceRadius").onChange((value) => {
          if (value) {
            this.state.effectController.aoRadius = 48.0;
            this.state.effectController.distanceFalloff = 0.2;
              aor._min = 0;
              aor._max = 64;
              df._min = 0;
              df._max = 1;
          } else {
            this.state.effectController.aoRadius = 5.0;
            this.state.effectController.distanceFalloff = 1.0;
              aor._min = 1;
              aor._max = 10;
              df._min = 0;
              df._max = 10;
          }
          aor.updateDisplay();
          df.updateDisplay();
  
          this.updateSSAO();
      });
      effectFolder.add(this.state.effectController, "halfRes").onChange(()=>this.updateSSAO());
      effectFolder.add(this.state.effectController, "depthAwareUpsampling").onChange(()=>this.updateSSAO());
      effectFolder.add(this.state.effectController, "intensity", 0.0, 10.0, 0.01).onChange(()=>this.updateSSAO());
      effectFolder.addColor(this.state.effectController, "color").onChange(()=>this.updateSSAO());
      effectFolder.add(this.state.effectController, "colorMultiply").onChange(()=>this.updateSSAO());
    }
    
    effectFolder.add(this.state.effectController, "renderMode", ["Combined", "AO", "No AO", "Split", "Split AO"]).onChange(()=>this.updateSSAO());

    this.updateSSAO();

    // Stats.
    const perfFolder = gui.addFolder('性能');
    this.runtimeDebugState.cullingMode = this.slm2Loader.getCullingMode();
    this.cullingModeController = perfFolder
      .add(this.runtimeDebugState, 'cullingMode', {
        '神经剔除': 'neural',
        '视锥剔除': 'frustum',
      })
      .name('剔除模式');
    this.cullingModeController.onChange((value) => this.setCullingMode(value));
    this.runtimeDebugState.neuralRenderPolicy = this.slm2Loader.getNeuralRenderPolicy();
    this.runtimeDebugState.neuralDownloadPlanMode = this.slm2Loader.getNeuralDownloadPlanMode();
    this.neuralRenderPolicyController = perfFolder
      .add(this.runtimeDebugState, 'neuralRenderPolicy', ['culled', 'resident'])
      .name('神经渲染策略');
    this.neuralRenderPolicyController.onChange((value) => this.setNeuralRenderPolicy(value));
    this.neuralDownloadPlanModeController = perfFolder
      .add(this.runtimeDebugState, 'neuralDownloadPlanMode', {
        '当前相机优先调度': 'viewcell-priority',
        '直接下载模型可见集': 'raw-visible',
      })
      .name('下载调度模式');
    this.neuralDownloadPlanModeController.onChange((value) => this.setNeuralDownloadPlanMode(value));
    const pvsDebugFolder = gui.addFolder('PVS调试');
    const addRuntimeStatus = (property, label) => {
      const row = document.createElement('li');
      row.className = 'cr string pvs-runtime-row';

      const name = document.createElement('span');
      name.className = 'property-name';
      name.textContent = label;

      const valueWrap = document.createElement('div');
      valueWrap.className = 'c';

      const value = document.createElement('div');
      value.className = 'pvs-runtime-value';
      value.textContent = this.runtimeDebugState[property] != null ? String(this.runtimeDebugState[property]) : '-';
      value.title = value.textContent;

      valueWrap.appendChild(value);
      row.appendChild(name);
      row.appendChild(valueWrap);
      pvsDebugFolder.__ul.appendChild(row);
      this.runtimeDebugValueEls[property] = value;
      return value;
    };
    addRuntimeStatus('frontendAssetEstimate', '前端资产估算');
    addRuntimeStatus('pvsModelVersion', '模型版本');
    addRuntimeStatus('pvsModelSchema', '模型Schema/阈值');
    addRuntimeStatus('pvsModelWorkpoint', 'Calibration安全工作点');
    addRuntimeStatus('pvsCullingSource', '当前剔除来源');
    addRuntimeStatus('pvsBackend', 'Worker后端');
    addRuntimeStatus('pvsReady', '模型就绪');
    addRuntimeStatus('pvsFallbackReason', '降级原因');
    addRuntimeStatus('pvsPredictMs', '预测总耗时');
    addRuntimeStatus('pvsInferenceMs', '模型推理耗时');
    addRuntimeStatus('pvsPostMs', '后处理耗时');
    addRuntimeStatus('pvsPredictionAgeMs', '预测结果年龄');
    addRuntimeStatus('pvsCandidateSelection', '后退视锥候选索引');
    addRuntimeStatus('pvsGate', '触发门槛');
    addRuntimeStatus('pvsRawInstances', '后退视锥模型构件');
    addRuntimeStatus('pvsRawGlbs', '后退视锥模型GLB');
    addRuntimeStatus('pvsImmediateGlbs', '立即下载GLB');
    addRuntimeStatus('pvsPrefetchGlbs', '预取GLB');
    addRuntimeStatus('pvsRenderInstances', '当前视锥显示构件');
    addRuntimeStatus('pvsRenderGlbs', '当前视锥显示GLB');
    addRuntimeStatus('pvsActualRender', '实际可见Mesh/实例');
    addRuntimeStatus('pvsDownloadQueue', '下载队列');
    addRuntimeStatus('pvsPrefetchQueue', '预取队列');
    addRuntimeStatus('pvsActiveLoads', '主动下载中');
    addRuntimeStatus('pvsInflightLoads', 'HTTP进行中');
    addRuntimeStatus('pvsPendingIntegrate', '待集成GLB');
    addRuntimeStatus('pvsHttpMbps', '带宽状态');
    addRuntimeStatus('pvsHttpConcurrency', '当前并发');
    addRuntimeStatus('pvsCacheSummary', '缓存状态');
    addRuntimeStatus('pvsRenderSummary', '显示过滤');
    addRuntimeStatus('pvsPredictionDebugSummary', '红色调试');

    const pvsDebugActions = {
      refreshPrediction: () => {
        if (this.slm2Loader && typeof this.slm2Loader.forceSceneCullingNow === 'function')
        {
          this.slm2Loader.forceSceneCullingNow({
            forceNeural: this.slm2Loader.getCullingMode() === 'neural',
          });
        }
        this.updateRuntimeDebugGui(performance.now(), true);
      },
      togglePredictionRed: () => {
        this.setPredictionDebugEnabled(!this.predictionDebugEnabled);
        this.updateRuntimeDebugGui(performance.now(), true);
      },
      freezePrediction: () => {
        this.toggleFrozenPredictionInspectMode();
        this.updateRuntimeDebugGui(performance.now(), true);
      },
      captureTopDown: () => {
        this.capturePredictionDebugTopDown();
        this.updateRuntimeDebugGui(performance.now(), true);
      },
    };
    pvsDebugFolder.add(pvsDebugActions, 'refreshPrediction').name('立即刷新预测');
    pvsDebugFolder.add(pvsDebugActions, 'togglePredictionRed').name('切换红色高亮');
    pvsDebugFolder.add(pvsDebugActions, 'freezePrediction').name('冻结/解冻预测');
    pvsDebugFolder.add(pvsDebugActions, 'captureTopDown').name('保存俯视调试图');
    pvsDebugFolder.close();
    this.updateRuntimeDebugGui(performance.now(), true);

    const perfLi = document.createElement('li');
    this.stats.dom.style.position = 'static';
    perfLi.appendChild(this.stats.dom);
    perfLi.classList.add('gui-stats');
    perfFolder.__ul.appendChild( perfLi );

    const opFolder = gui.addFolder('操作');
    var scope = this;
    var obj = { 生成加载列表 : function()
      {
        scope.generateLoadList();
      }
    };
    opFolder.add(obj,'生成加载列表');

    const guiWrap = document.createElement('div');
    this.el.appendChild( guiWrap );
    guiWrap.classList.add('gui-wrap');
    guiWrap.appendChild(gui.domElement);
    gui.close();
  }

  clear()
  {
    this.removeSceneGround();

    if(!this.content) return;

    this.scene.remove( this.content );

    // dispose geometry
    this.content.traverse((node) => 
    {
      if ( !node.isMesh ) return;
      node.geometry.dispose();
    } );

    // dispose textures
    traverseMaterials( this.content, (material) => 
    {
      MAP_NAMES.forEach( (map) => {
        if (material[ map ]) material[ map ].dispose();
      });
    });
  }
};

function traverseMaterials (object, callback) {
  // object.traverse((node) => {
  //   if (!node.isMesh) return;
  //   const materials = Array.isArray(node.material)
  //     ? node.material
  //     : [node.material];
  //   materials.forEach(callback);
  // });
}

function getGometrySize (obj) {
  let size = 0;
  obj.traverse((node) => {
    if (node.isMesh) {
      const geometry = node.geometry;
      if (geometry) {
        size += BufferGeometryUtils.estimateBytesUsed(geometry); //加上几何体大小
      }
    }
  });
  return size;
}
