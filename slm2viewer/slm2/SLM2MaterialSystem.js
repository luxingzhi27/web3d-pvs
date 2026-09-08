import { CanvasTexture, DoubleSide, FileLoader, FloatType, LinearFilter, RepeatWrapping } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { HDRJPGLoader } from '@monogrid/gainmap-js';
import { startupLog } from '../src/startupTimeline.js';
import { DRACO_LOADER, joinUrlPath } from './SLM2RuntimeAssets.js';
import { SLM2SceneCatalog } from './SLM2SceneCatalog.js';

const sequentialPromiseMap = require('sequential-promise-map');

export class SLM2MaterialSystem extends SLM2SceneCatalog {
  loadMaterialConfig(callback)
  {
    startupLog('slm2:materialConfig:start');
    var scope = this;
    var materialConfigStartedAt = performance.now();
    if (this.DebugMode) console.log(this.sceneConfig);

    var loadMtlProxy = function (taskData, index/*, array*/) 
    {
      return new Promise(function(resolve)
      {
        var taskStartedAt = performance.now();
        var proxyModelUrl = scope.resourcesBaseUrl + '/' + taskData.proxyGlb;
        startupLog('slm2:materialConfig:proxy-request', {
          index: index,
          proxyGlb: taskData.proxyGlb,
        });

        var gltfLoader = new GLTFLoader()
          .setCrossOrigin('anonymous')
          .setDRACOLoader(DRACO_LOADER)
          .setMeshoptDecoder(MeshoptDecoder);
        gltfLoader.load(proxyModelUrl, (gltf) => 
        {
          var gltfLoadedAt = performance.now();
          startupLog('slm2:materialConfig:proxy-loaded', {
            index: index,
            ms: gltfLoadedAt - taskStartedAt,
          });
          // Load image lod config
          var fileLoader = new FileLoader();
          startupLog('slm2:materialConfig:image-config-request', {
            index: index,
            imageConfig: taskData.imageConfig,
          });
          var imageConfigUrl = joinUrlPath(
            scope.glbResourcesBaseUrl || scope.resourcesBaseUrl,
            taskData.imageConfig
          );
          fileLoader.load(imageConfigUrl, function(data)
          {
            var imageConfig = null;
            try
            {
              imageConfig = JSON.parse(data);
            }
            catch (error)
            {
              console.warn('[SLM2Loader] Invalid material image config response:', imageConfigUrl, error);
              resolve(null);
              return;
            }
            startupLog('slm2:materialConfig:image-config-loaded', {
              index: index,
              ms: performance.now() - gltfLoadedAt,
            });

            resolve(
              {
                gltf: gltf,
                imageConfig: imageConfig,
                taskIndex: index,
                proxyGlb: taskData.proxyGlb,
                taskStartedAt: taskStartedAt,
              });
          }, null, function(err)
          {
            console.warn('[SLM2Loader] Failed to load material image config:', imageConfigUrl, err);
            resolve(null);
          });
        }, null, (err) =>{
          console.warn('[SLM2Loader] Failed to load material proxy:', proxyModelUrl, err);
          startupLog('slm2:materialConfig:proxy-failed', {
            index: index,
            proxyGlb: taskData.proxyGlb,
          });
          resolve(null);
        });
      });
    };

    var collectMaterialTexture = function(mtl)
    {
      var texs = [];

      for (var key in mtl)
      {
        var item = mtl[key];
        if (item != null && item.isTexture)
        {
          var texChannel = key;

          if (key == 'aoMap')
          {
            texChannel = 'lightMap'; 
          }

          texs.push(
          {
            name: item.name,
            channel: texChannel,
            level: 0,
            data: 
            {
              "0": item
            }
          });
        }
      };

      if (scope.DebugMode) console.log(mtl, texs);

      return texs;
    }

    var mtlConfigTask = [];

    for (var i = 0; i < this.sceneConfig.materials.proxy.length; ++i)
    {
      mtlConfigTask.push(
        {
          proxyGlb: this.sceneConfig.materials.proxy[i],
          imageConfig: 'task-' + i + '/' + 'images/image_lod.json',
        });
    }

    var loadMaterialProxyList = function(tasks)
    {
      return new Promise(function(resolve)
      {
        var results = new Array(tasks.length);
        var runNext = function(index)
        {
          if (index >= tasks.length)
          {
            startupLog('slm2:materialConfig:all-proxies-loaded', {
              count: results.length,
            });
            resolve(results);
            return;
          }

          var loadCurrent = function()
          {
            loadMtlProxy(tasks[index], index).then(function(result)
            {
              results[index] = result;
              setTimeout(function()
              {
                runNext(index + 1);
              }, 0);
            });
          };

          if (typeof requestIdleCallback === 'function')
          {
            requestIdleCallback(loadCurrent, { timeout: 1000 });
          }
          else
          {
            setTimeout(loadCurrent, 0);
          }
        };

        runNext(0);
      });
    };

    loadMaterialProxyList(mtlConfigTask).then(results => 
    {
      this.sceneConfig.materials.data = new Array(this.sceneConfig.materials.proxy.length);
      this.sceneConfig.materials.config = new Array(this.sceneConfig.materials.proxy.length);

      for (var i = 0; i < results.length; ++i)
      {
        if (!results[i])
        {
          continue;
        }
        var mtls = {};
        results[i].gltf.scene.traverse((node) => 
        {
          if (!node.isMesh) return;
          if (node.material) 
          {
            node.material.flatShading = false;

            var texs = collectMaterialTexture(node.material);

            var preloadMaterial = node.material;
            preloadMaterial.side = DoubleSide;

            mtls[preloadMaterial.name] = 
            {
              material: preloadMaterial,
              textures: texs
            }
          }
        });

        this.sceneConfig.materials.config[results[i].taskIndex] = results[i].imageConfig;
        this.sceneConfig.materials.data[results[i].taskIndex] = mtls;
      }

      if (this.options.materialLoadedCallback)
      {
        this.options.materialLoadedCallback();
      }
      this.isMaterialConfigReady = true;
      this.lastMaterialRebindCount = this._rebindResidentMaterialsToCache();
      this.staticSceneOptimizer.optimizeAllResidents();
      this.requestRender('material-config-ready');
      startupLog('slm2:materialConfig:ready', {
        totalMs: performance.now() - materialConfigStartedAt,
        reboundMeshes: this.lastMaterialRebindCount,
      });
      //console.log(this.sceneConfig.materials.data);

      if (callback)
      {
        scope.startupMetrics.materialConfigMs = performance.now() - materialConfigStartedAt;
        callback();
      }
    }).catch(()=>{
      this.isMaterialConfigReady = false;
      this.lastMaterialRebindCount = 0;
      startupLog('slm2:materialConfig:failed');
      if (callback)
      {
        scope.startupMetrics.materialConfigMs = performance.now() - materialConfigStartedAt;
        callback();
      }
    });
  }

  getMaterials()
  {
    var mtls = [];

    if (this.sceneConfig && this.sceneConfig.materials)
    {
      for (var key in this.sceneConfig.materials.data)
      {
        var item = this.sceneConfig.materials.data[key];

        for (var _key in item)
        {
          var _item = item[_key];

          mtls.push(_item.material);
        };
      };
    }

    return mtls;
  }

  fetchCachedMaterial(srcMtl, gltfDesc)
  {
    var matchedGroupId = parseInt(gltfDesc.hashCode.split('-')[0]);
    if (!this.sceneConfig ||
      !this.sceneConfig.materials ||
      !Array.isArray(this.sceneConfig.materials.data) ||
      this.sceneConfig.materials.data[matchedGroupId] == undefined)
    {
      return srcMtl;
    }
    var cachedMtlObj = this.sceneConfig.materials.data[matchedGroupId][srcMtl.name];

    // 添加新的贴图加载任务
    var newImageTask = 
    {
      groupId: matchedGroupId,
      mtlName: srcMtl.name,
      gltfHash: gltfDesc.hashCode,
      weightNormalized: this.modelCacheMgr.objectsPool[gltfDesc.hashCode].weight * this.screenPixelReciprocal
    };

    if (this.materialImageLoadingTasks == undefined)
    {
      this.materialImageLoadingTasks = {};
      this.isMaterialImageLoading = false;
    }

    if (this.materialImageLoadingTasks[srcMtl.name] == undefined || this.materialImageLoadingTasks[srcMtl.name].weight < newImageTask.weight)
    {
      this.materialImageLoadingTasks[srcMtl.name] = newImageTask;
    }

    if (cachedMtlObj && cachedMtlObj.material)
    {
      return cachedMtlObj.material;
    }
    else
    {
      console.log('failed to find material: ' + srcMtl.name);

      return srcMtl;
    }
  }

  _rebindResidentMaterialsToCache()
  {
    if (!this.modelCacheMgr || !this.modelCacheMgr.objectsPool)
    {
      return 0;
    }

    if (!this.sceneConfig ||
      !this.sceneConfig.materials ||
      !Array.isArray(this.sceneConfig.materials.data) ||
      this.sceneConfig.materials.data.length === 0)
    {
      return 0;
    }

    var reboundMeshes = 0;
    var pool = this.modelCacheMgr.objectsPool;
    for (var hash in pool)
    {
      var item = pool[hash];
      if (!item || !item.meshObject)
      {
        continue;
      }

      var extras = item.meshObject.asset && item.meshObject.asset.extras
        ? item.meshObject.asset.extras
        : { hashCode: hash };
      item.meshObject.traverse((node) =>
      {
        if (!node.isMesh || !node.material)
        {
          return;
        }
        node.material = this.fetchCachedMaterial(node.material, extras);
        reboundMeshes++;
      });
    }

    return reboundMeshes;
  }

  refreshMaterialTexture(groupId, materialName, textureId, textureData, isFromCache, imageMeta)
  {
    var material = this.sceneConfig.materials.data[groupId][materialName].material;
    var texture = this.sceneConfig.materials.data[groupId][materialName].textures[textureId];

    if (material != undefined && texture != undefined)
    {
      texture.level++;

      // 若贴图为空，可能是加载失败，直接跳过当前的细节度
      if (textureData != null)
      {
        if (texture.channel == 'aoMap')
        {
          if (isFromCache == false)
          {
            textureData.wrapS = material[texture.channel].wrapS;
            textureData.wrapT = material[texture.channel].wrapT;
            textureData.offset = material[texture.channel].offset;
            textureData.repeat  = material[texture.channel].repeat;
            textureData.rotation = material[texture.channel].rotation;
            textureData.center = material[texture.channel].center;
            textureData.colorSpace = material[texture.channel].colorSpace;
            textureData.flipY = material[texture.channel].flipY;
            textureData.generateMipmaps = true;
          }
          
          material[texture.channel] = textureData;
          material.needsUpdate = true;
        }
        else if (texture.channel == 'lightMap')
        {
          if (isFromCache == false)
          {
            textureData.channel = 1;

            if (imageMeta.type == 'jpgr')
            {

            }
            else
            {
              textureData.flipY = (imageMeta.raw == 'exr' ? true: false);
              textureData.type = FloatType;
              textureData.minFilter = LinearFilter;
              textureData.magFilter = LinearFilter;
              textureData.wrapS = RepeatWrapping;
              textureData.wrapT = RepeatWrapping;
            }
            
            textureData.needsUpdate = true;
          }
          
          material[texture.channel] = textureData;
          material.needsUpdate = true;
        }
        else
        {
          if (isFromCache == false)
          {
            textureData.wrapS = material[texture.channel].wrapS;
            textureData.wrapT = material[texture.channel].wrapT;
            textureData.offset = material[texture.channel].offset;
            textureData.repeat  = material[texture.channel].repeat;
            textureData.rotation = material[texture.channel].rotation;
            textureData.center = material[texture.channel].center;
            textureData.colorSpace = material[texture.channel].colorSpace;
            textureData.flipY = material[texture.channel].flipY;
            textureData.generateMipmaps = true;
          }
    
          material[texture.channel] = textureData;
          material[texture.channel].needsUpdate = true;
        }

        texture.data[texture.level] = textureData;
      }
    }
  }

  processImageTask(dt = 0)
  {
    var nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    var minIntervalMs = this.cpuPerfMode === 'mobile' ? 500 : 250;
    if (this.lastTextureTaskAt > 0 && nowMs - this.lastTextureTaskAt < minIntervalMs)
    {
      return;
    }

    if (this._getPendingSceneInsertionCount() > 0 || Number(dt || 0) > 25)
    {
      return;
    }

    this.lastTextureTaskAt = nowMs;
    var textureTaskStart = nowMs;
    if (this.materialImageLoadingTasks == undefined)
    {
      this.lastTextureTaskMs = 0;
      return;
    }
    var scope = this;
    if (this.isMaterialImageLoading == false)
    {
      var totalTasks = [];

      for (var key in this.materialImageLoadingTasks)
      {
        var item = this.materialImageLoadingTasks[key];
        var materialGroup = scope.sceneConfig.materials.data[item.groupId];
        var imageConfigGroup = scope.sceneConfig.materials.config[item.groupId];
        if (!materialGroup || !imageConfigGroup)
        {
          continue;
        }
        var mtlConfig = materialGroup[key];

        if (mtlConfig != undefined && mtlConfig.textures != undefined)
        {
          for (var i = 0; i < mtlConfig.textures.length; ++i)
          {
            var texName = mtlConfig.textures[i].name;
            var imageConfig = imageConfigGroup[texName];
            if (!imageConfig || !Array.isArray(imageConfig.lod))
            {
              continue;
            }
            var maxLodLevel = imageConfig.lod.length;
            if (mtlConfig.textures[i].level < maxLodLevel - 1)
            {
              var cachedInfo = scope.modelCacheMgr.objectsPool[item.gltfHash];

              if (cachedInfo.isVisible)
              {
                totalTasks.push(
                  {
                    groupId: item.groupId,
                    mtlName: key,
                    lodLevel: mtlConfig.textures[i].level,
                    texId: i,
                    imageUrl: joinUrlPath(
                      scope.glbResourcesBaseUrl || scope.resourcesBaseUrl,
                      'task-' + item.groupId + '/images/LOD' + (mtlConfig.textures[i].level + 1) + '/' + encodeURIComponent(imageConfig.uri)
                    ),
                    weightNormalized: 1.0 - item.weightNormalized,
                    texName: texName,
                    isLightMap: mtlConfig.textures[i].channel == 'lightMap',
                  });
                //注意：这里一定需要使用encodeURIComponent来对图片的名称进行编码，否则会出现特殊符号（如#）无法编码，导致加载失败
              }
            }
          }
        }
      };

      var fetchCachedTexture = function(taskData, loadCallback, errorCallback)
      {
        var cacheKey = taskData.groupId + '-' + taskData.texName + "-" + taskData.lodLevel;

        if (scope.loadedTextures[cacheKey])
        {
          if (scope.DebugMode) console.log('texture fectched from cache: ' + taskData.texName + ", " + cacheKey);
          if (loadCallback) loadCallback(scope.loadedTextures[cacheKey], true);
        }
        else
        {
          var loader = scope.textureLoader;

          var imageMeta = 
          {
            'raw' : 'png',
            'type': 'png',
          }
          
          if (taskData.imageUrl.endsWith('.hdr') || taskData.imageUrl.endsWith('.exr'))
          {
            if (taskData.imageUrl.endsWith('.hdr'))
            {
              imageMeta.raw = 'hdr';
              imageMeta.type = 'hdr';
            }
            else if (taskData.imageUrl.endsWith('.exr'))
            {
              imageMeta.raw = 'exr';
              imageMeta.type = 'exr';
            }

            if (scope.sceneConfig.materials.useHdrJpg)
            {
              if (scope.hdrjpgLoader == null)
              {
                scope.hdrjpgLoader = new HDRJPGLoader(scope.renderer);
              }
              
              loader = scope.hdrjpgLoader;

              imageMeta.type = 'jpgr';
            }
            else
            {
              if (taskData.imageUrl.endsWith('.hdr'))
              {
                loader = scope.hdrLoader;
              }
              else if (taskData.imageUrl.endsWith('.exr'))
              {
                loader = scope.exrLoader;
              }
            }
          }
          
          loader.load(taskData.imageUrl, function(_textureData)
          {
            var textureData = null;
            if (imageMeta.type == 'png')
            {
              textureData = new CanvasTexture( _textureData );
            }
            else if (imageMeta.type == 'jpgr')
            {
              textureData = _textureData.renderTarget.texture;
            }
            else
            {
              textureData = _textureData;
            }

            scope.loadedTextures[cacheKey] = textureData;

            if (loadCallback) loadCallback(textureData, false, imageMeta);
          }, null, function(err)
          {
            if (errorCallback) errorCallback(err);
          });
        }
      }

      var loadImage = function (taskData/*, index, array*/) 
      {
        return new Promise(resolve => 
        {
          fetchCachedTexture(taskData,
            function(textureData, isFromCache, imageType)
            {
              scope.refreshMaterialTexture(taskData.groupId, taskData.mtlName, taskData.texId, textureData, isFromCache, imageType);

              if (isFromCache == false)
              {
                // 加入延时，避免占用太多模型加载的资源
                setTimeout(function(){
                  resolve();
                }, 250);
              }
              else
              {
                resolve();
              }
            }, function(err)
            {
              console.log(err);

              scope.refreshMaterialTexture(taskData.groupId, taskData.mtlName, taskData.texId, null);

              resolve();
            }
          ); 
        });
      };

      if (totalTasks.length > 0)
      {
        function compareTexturePriority(keyA, keyB) 
        {
          function getPriority(item)
          {
            var priority = (item.isLightMap? -200 : 0) + item.weightNormalized * -10 + (item.texId + 1) * 100 + (item.lodLevel + 1) * 200;
            return priority;
          }
          
          return getPriority(keyA) - getPriority(keyB);
        }

        totalTasks.sort(compareTexturePriority);

        var AsyncTextureTaskCount = 3;

        var subTasks = totalTasks.slice(0, Math.min(AsyncTextureTaskCount, totalTasks.length)); // 分片加载，防止阻塞

        {
          this.isMaterialImageLoading = true;

          sequentialPromiseMap(subTasks, loadImage).then(results =>
          {
            this.isMaterialImageLoading = false;
            setTimeout(() => this.requestRender('texture-task-complete'), minIntervalMs);
          });
        }
        
      }
    }
    this.lastTextureTaskMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - textureTaskStart;
  }

  update(dt, time)
  {
    if (this.isSceneInitialized == false)
    {
      return;
    }

    this.lastFrameDt = Number(dt || 0);
    if (this.cameraUpdatePending)
    {
      this.syncCamera();
    }
    this.updateLoading(dt);
    this.processPendingSceneInsertions();

    this.modelCacheMgr.update(time);

    // This only reconciles newly resident GLBs against the latest exact GPU
    // render set and never starts neural prediction or performs culling.
    if (this.renderVisibilitySystem && typeof this.renderVisibilitySystem.update === 'function')
    {
      this.renderVisibilitySystem.update();
    }

    this.staticSceneOptimizer.processPending(performance.now(), this._isResourcePipelineIdle());

    this.processImageTask(this.lastFrameDt);
  }

  hasImmediateFrameWork()
  {
    var pipelineIdle = this._isResourcePipelineIdle();
    return this._getPendingSceneInsertionCount() > 0
      || Boolean(this.renderVisibilitySystem && this.renderVisibilitySystem.dirtyHashes.size > 0)
      || this.staticSceneOptimizer.hasReadyWork(performance.now(), pipelineIdle);
  }

  _isResourcePipelineIdle()
  {
    var scheduledWork = this.useNeuralPVS
      ? this.glbResourceScheduler.hasPendingWork()
      : this.modelToLoadList.length > 0;
    return !scheduledWork
      && this.activeDirectLoadCount === 0
      && this._getPendingGlbParseCount() === 0
      && this.activeGlbParseCount === 0
      && this._getPendingSceneInsertionCount() === 0;
  }

  getMaintenanceDelayMs()
  {
    if (this.cameraUpdatePending)
    {
      return Math.max(0, 80 - Number(this.cullingUpdateDelta || 0));
    }
    if (this._getPendingSceneInsertionCount() > 0)
    {
      return 0;
    }
    if (this._getPendingGlbParseCount() > 0 && this.activeGlbParseCount < this.glbParseConcurrency)
    {
      return 0;
    }
    if (!this.useNeuralPVS && this.modelToLoadList.length > 0 && this.activeDirectLoadCount === 0)
    {
      return 0;
    }
    if (this.useNeuralPVS && this.activeDirectLoadCount === 0)
    {
      var schedulerDelay = this.glbResourceScheduler.nextWakeDelay();
      if (schedulerDelay != null)
      {
        return schedulerDelay;
      }
    }
    return this.staticSceneOptimizer.nextWakeDelay(performance.now(), this._isResourcePipelineIdle());
  }
}
