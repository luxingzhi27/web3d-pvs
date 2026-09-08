import { AmbientLight, Color, DirectionalLight, DoubleSide, FloatType, LinearEncoding, LinearFilter, Mesh, RepeatWrapping, sRGBEncoding } from 'three';
import { GUI } from 'dat.gui';
import { HDRJPGLoader } from '@monogrid/gainmap-js';
import { environments } from '../assets/environment/index.js';
import { traverseMaterials } from './ViewerMaterialTraversal.js';
import { ViewerRenderLoop } from './ViewerRenderLoop.js';

const MAP_NAMES = ['map', 'aoMap', 'emissiveMap', 'glossinessMap', 'metalnessMap', 'normalMap', 'roughnessMap', 'specularMap'];

export class ViewerPresentation extends ViewerRenderLoop {
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
      this.n8aopass.enabled = true;
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
    addRuntimeStatus('pvsImmediateGlbs', '计划立即下载GLB');
    addRuntimeStatus('pvsPrefetchGlbs', '计划预取GLB总数');
    addRuntimeStatus('pvsRenderInstances', '当前视锥显示构件');
    addRuntimeStatus('pvsRenderGlbs', '当前视锥显示GLB');
    addRuntimeStatus('pvsActualRender', '实际可见Mesh/实例');
    addRuntimeStatus('pvsDownloadQueue', '剩余立即队列');
    addRuntimeStatus('pvsPrefetchQueue', '剩余预取队列');
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

    const guiWrap = document.createElement('div');
    this.el.appendChild( guiWrap );
    guiWrap.classList.add('gui-wrap');
    guiWrap.appendChild(gui.domElement);
    gui.domElement.addEventListener('input', () => this.requestRender('gui-input'));
    gui.domElement.addEventListener('change', () => this.requestRender('gui-change'));
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

    this.content.traverse((node) => {
      if (!node.isMesh || !node.material) return;
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      materials.forEach((material) => {
        MAP_NAMES.forEach((map) => {
          if (material[map]) material[map].dispose();
        });
      });
    });
  }
}

