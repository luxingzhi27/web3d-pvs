import { Color, DoubleSide, FileLoader, GridHelper, Mesh, MeshBasicMaterial, Object3D, PlaneGeometry, Vector3 } from 'three';
import { startupLog } from './startupTimeline.js';
import { ViewerPresentation } from './ViewerPresentation.js';

const VIEWER_CONFIG_CACHE_VERSION = 'pvs-v4-hkust-native-dpr-20260826';

export class ViewerSceneController extends ViewerPresentation {
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

      var sceneLoadConfig = {
        name: activeSceneName,
        loader: scope.activeScene.loaderConfig,
      };

      startupLog('viewer:slm2Loader-load-call', { method: 'load' });
      var rootScene = scope.slm2Loader.load(sceneLoadConfig, scope.renderer, scope.activeCamera,
      {
        materialLoadedCallback: _materialLoadedCallback,
        paramJson: scope.paramJson,
        requestRender: scope.requestRender.bind(scope),
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
}

