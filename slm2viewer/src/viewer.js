import {
  Cache,
  PMREMGenerator,
  PerspectiveCamera,
  Scene,
  WebGLRenderer,
  sRGBEncoding,
  DoubleSide,
  MeshBasicMaterial,
} from 'three';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { environments } from '../assets/environment/index.js';
import { createBackground } from '../lib/three-vignette.js';
import { SLM2Loader } from '../slm2/SLM2Loader';
import { keyboardMgr } from './keyboardMgr.js';
import { TrajectoryCollector } from './TrajectoryCollector.js';
import { TouchMoveController } from './TouchMoveController.js';
import { startupLog } from './startupTimeline.js';
import { FRONTEND_RENDER_FOV_Y_DEG } from './neuralPvsFovProtocol.js';
import { resolveRenderSurface } from './RenderSurfacePolicy.js';
import { FRONTEND_RUNTIME_ASSET_ESTIMATE } from './ViewerQueueDiagnostics.js';
import { ViewerSceneController } from './ViewerSceneController.js';

import { N8AOPostPass  } from 'n8ao';

import { EffectComposer, RenderPass, EffectPass, SMAAEffect, SMAAPreset } from "postprocessing";

Cache.enabled = true;

export class Viewer extends ViewerSceneController
{
  constructor (el, options) 
  {
    super();
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
    this.renderSurface = resolveRenderSurface({
      width: el.clientWidth,
      height: el.clientHeight,
      devicePixelRatio: window.devicePixelRatio,
    });
    this.resizeDebounceMs = 150;
    this.resizeTimer = null;

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

    this.renderer = window.renderer = new WebGLRenderer();//{antialias: (this.DebugMode ? false: true)});
    this.renderer.physicallyCorrectLights = true;
    this.renderer.outputEncoding = sRGBEncoding;
    this.renderer.setClearColor( 0xdddddd );
    this.renderer.setPixelRatio(this.renderSurface.pixelRatio);
    this.renderer.setSize( el.clientWidth, el.clientHeight );
    //this.renderer.autoClear = false;
    startupLog('viewer:renderer-ready', {
      width: el.clientWidth,
      height: el.clientHeight,
      pixelRatio: this.renderSurface.pixelRatio,
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
    this.animationFrameId = null;
    this.maintenanceTimer = null;
    this.renderRequested = false;
    this.orbitInteractionActive = false;
    this.scheduleResize = this.scheduleResize.bind(this);
    window.addEventListener('resize', this.scheduleResize, false);

    window.addEventListener('keydown', this.keydown.bind(this), false);
    this.slm2Loader = new SLM2Loader();
    this.keyboardMgr = new keyboardMgr(this);
    this.touchMoveController = new TouchMoveController(this);
    this.trajectoryCollector = new TrajectoryCollector(this);
    this.controls.addEventListener('start', () => {
      this.orbitInteractionActive = true;
      this.requestRender('orbit-start');
    });
    this.controls.addEventListener('change', () => {
      if (this.slm2Loader) this.slm2Loader.notifyCameraChanged();
      this.requestRender('orbit-change');
    });
    this.controls.addEventListener('end', () => {
      this.orbitInteractionActive = false;
      this.requestRender('orbit-end');
    });
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

    this.requestRender('viewer-startup');

    this.usePostEffect = true;

    if (this.usePostEffect)
    {
      this.composer = new EffectComposer(this.renderer);
      this.renderPass = new RenderPass(this.scene, this.activeCamera);
      this.composer.addPass(this.renderPass);
      this.n8aopass = new N8AOPostPass (this.scene, this.activeCamera, el.clientWidth, el.clientHeight);
      this.n8aopass.enabled = true;
      this.composer.addPass(this.n8aopass);
      this.smaaPass = new EffectPass(this.activeCamera, new SMAAEffect({preset: SMAAPreset.ULTRA}));
      this.smaaPass.enabled = true;
      this.composer.addPass(this.smaaPass);
      startupLog('viewer:post-effects-ready');
  
      //const gammaCorrectionPass = new ShaderPass( GammaCorrectionShader )
      //this.composer.addPass(gammaCorrectionPass);
    }

    startupLog('viewer:constructor:end');
  }

};
