export class CacheMgr 
{
  constructor (options) 
  {
    this.options = options;

    this.lastUpdateTime = -1;

    this.timeSinceStartup = 0;

    this.objectsPool = {};

    this.MaxMemoryUsageInKB = 1024 * 1900; // 100 MB

    this.baseItem =
    {
      meshObject: null, // 3D Object
      isVisible: false, // visible or no 
      isInScene: false, // is curent in scene
      lastUpdateTime: 0, // 
      lastVisibleAt: 0,
      missedUpdates: 0,
      sizeKB: 0, // file size in KB
    }

    this.statsDetail = 
    {
      memoryUsed: 0,// (usedMemroy / 1024).toFixed(0) + '/' + (this.MaxMemoryUsageInKB / 1024).toFixed(0) + ' MB',
      visibleNums: 0,
      invisibleNums: 0, 
      releasedNums: 0,
      inSceneNums: 0,
      totalNums: 0,
    }
  }

  traverseObj(jsonObj, callback)
  {
      for (var key in jsonObj) 
      {
          if (callback)
          {
              callback(key, jsonObj[key]);
          }
      }
  }

  applyRenderState(item)
  {
    if (!item || !item.meshObject)
    {
      return;
    }

    // An invalid instance-to-matrix binding must never fall back to showing
    // the GLB root.  SLM2Loader marks this flag after validating the loaded
    // InstancedMesh against the runtime component metadata.
    var targetVisible = Boolean(item.isVisible && item.isInScene && !item.instancedBindingInvalid);
    var sceneMgr = this.options ? this.options.sceneMgr : null;
    if (item.staticBatchHash && sceneMgr && sceneMgr.staticSceneOptimizer)
    {
      sceneMgr.staticSceneOptimizer.setHashVisible(item.staticBatchHash, targetVisible);
      item.meshObject.visible = false;
      return;
    }
    if (item.meshObject.visible !== targetVisible)
    {
      item.meshObject.visible = targetVisible;
    }
  }

  detachFromScene(item, hashCode)
  {
    if (!item || !item.meshObject || !item.isInScene)
    {
      return false;
    }

    if (item.meshObject.parent)
    {
      item.meshObject.removeFromParent();
    }
    item.isInScene = false;
    this.applyRenderState(item);
    this.notifyResidentChanged(hashCode);
    return true;
  }

  notifyResidentChanged(hashCode)
  {
    var sceneMgr = this.options ? this.options.sceneMgr : null;
    if (sceneMgr && sceneMgr.renderVisibilitySystem && typeof sceneMgr.renderVisibilitySystem.markResidentChanged === 'function')
    {
      sceneMgr.renderVisibilitySystem.markResidentChanged(hashCode);
    }
  }

  getNowMs()
  {
    if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function')
    {
      return performance.now();
    }

    return Date.now();
  }

  releaseObject(meshObject) 
  {
    meshObject.removeFromParent();

    meshObject.traverse((node) => 
    {
      if (!node.isMesh) return;
      node.geometry.dispose();
    });
  }

  releaseItem(item, hashCode)
  {
    var sceneMgr = this.options ? this.options.sceneMgr : null;
    if (item && sceneMgr && sceneMgr.staticSceneOptimizer
        && sceneMgr.staticSceneOptimizer.releaseHash(hashCode || item.staticBatchHash))
    {
      sceneMgr.staticSceneOptimizer.releaseHash(hashCode || item.staticBatchHash);
      if (item.meshObject && item.meshObject.parent)
      {
        item.meshObject.removeFromParent();
      }
      return;
    }
    if (item && item.meshObject)
    {
      this.releaseObject(item.meshObject);
    }
  }

  setSchedulingStrategy(ssMode)
  {
    this.schedulingStrategy = ssMode;
  }

  update(timeSinceStartup)
  {
    // Sync time with engine
    this.timeSinceStartup = timeSinceStartup;

    if (this.timeSinceStartup - this.lastUpdateTime > 1500)
    {
      // Strategy 2: In manual mode (full load), skip all eviction logic
      if (this.schedulingStrategy === 'manual') {
        this.lastUpdateTime = this.timeSinceStartup;
        return;
      }
      this.getStatsDetail();
      // console.table(this.statsDetail);
      
      var usedMemroy = this.statsDetail.memoryUsed;
      //console.log('Memory used: ' + (usedMemroy / 1024).toFixed(0) + '/' + (this.MaxMemoryUsageInKB / 1024).toFixed(0) + ' MB');

      //if (usedMemroy > this.MaxMemoryUsageInKB)
      {
        var scope = this;

        var objsNotInScene = [];

        this.traverseObj(this.objectsPool, function(key, item)
        {
          if (item.isVisible)
          {
            item.lastUpdateTime = scope.timeSinceStartup;
          }

          //if (item.meshObject && item.isInScene == false && item.isVisible == false) // 安全的策略
          if (item.meshObject && item.isVisible == false) // 激进的策略
          {
            objsNotInScene.push(key);
          }
        });

        // 如果内存超出限制，则进行释放
        if (usedMemroy > scope.MaxMemoryUsageInKB && (this.schedulingStrategy == 'auto'))
        {
          function compareLastUpdatetime(keyA, keyB) 
          {
            return scope.objectsPool[keyA].lastUpdateTime - scope.objectsPool[keyB].lastUpdateTime;
          }

          function compareFrequency(keyA, keyB) 
          {
            return scope.objectsPool[keyA].frequency - scope.objectsPool[keyB].frequency;
          }

          // 基于频率、最后更新时间双属性进行排序。注意：需要确保排序算法为稳定的
          //objsNotInScene.sort(compareFrequency);
          objsNotInScene.sort(compareLastUpdatetime);

          for (var i = 0; i < objsNotInScene.length; ++i)
          {
            var item = scope.objectsPool[objsNotInScene[i]];

            if (usedMemroy > scope.MaxMemoryUsageInKB)
            {
              if (item.isInScene)
              {
                item.meshObject.removeFromParent();
              }

              scope.releaseItem(item, objsNotInScene[i]);

              usedMemroy -= item.sizeKB;

              //console.log('free object to reduce memory: ' + item.sizeKB);

              scope.notifyResidentChanged(objsNotInScene[i]);
              delete scope.objectsPool[objsNotInScene[i]];

              scope.statsDetail.releasedNums++;
            }
          }
        }
      }

      this.lastUpdateTime = this.timeSinceStartup;

    }
  }

  getStatsDetail()
  {
    var scope = this;
    this.statsDetail.memoryUsed = 0;
    this.statsDetail.invisibleNums = 0;
    this.statsDetail.visibleNums = 0;
    this.statsDetail.inSceneNums = 0;
    this.statsDetail.totalNums = 0;

    this.traverseObj(this.objectsPool, function(key, item)
    {
      if (item.meshObject != null)
      {
        scope.statsDetail.memoryUsed += item.sizeKB;

        scope.statsDetail.visibleNums += (item.isVisible ? 1 : 0);
        scope.statsDetail.invisibleNums += (item.isVisible == false ? 1 : 0);
        scope.statsDetail.inSceneNums += ((item.isVisible == false && item.isInScene) ? 1 : 0);

        scope.statsDetail.totalNums++;
      }
    });
  }

  addObject(objDesc, meshObject, options = {})
  {
    var newObject =
    {
      meshObject: meshObject, // 3D Object
      isVisible: options.isVisible !== undefined ? options.isVisible : true, // visible or no 
      isInScene: options.isInScene !== undefined ? options.isInScene : true, // is curent in scene
      lastUpdateTime: this.timeSinceStartup, // 
      lastVisibleAt: (options.isVisible !== undefined ? options.isVisible : true) ? this.getNowMs() : 0,
      missedUpdates: 0,
      instancedBindingInvalid: false,
      staticBatchHash: null,
      staticBatchState: null,
      frequency: 0, // 访问频率,实际上等同于漫游期间占用画面的帧数
      weight: 500, // 默认设置比较高的权重
      sizeKB: parseInt(objDesc.sizeKB), // file size in KB
    }

    this.objectsPool[objDesc.hashCode] = newObject;
    this.applyRenderState(newObject);
    this.notifyResidentChanged(objDesc.hashCode);
  }

  tryCacheHit(hashCode, renderRoot)
  {
    if (this.objectsPool[hashCode])
    {
      if (this.objectsPool[hashCode].isInScene == false)
      {
        renderRoot.add(this.objectsPool[hashCode].meshObject);

        this.objectsPool[hashCode].isInScene = true;

        //console.log('add object from cache: ' + hashCode);
      }

      this.objectsPool[hashCode].isVisible = true;
      this.objectsPool[hashCode].lastUpdateTime = this.timeSinceStartup;
      this.objectsPool[hashCode].lastVisibleAt = this.getNowMs();
      this.objectsPool[hashCode].missedUpdates = 0;
      this.objectsPool[hashCode].frequency++;
      this.applyRenderState(this.objectsPool[hashCode]);
      this.notifyResidentChanged(hashCode);

      return true;
    }

    return false;
  }

  hideAll()
  {
    var scope = this;
    this.traverseObj(this.objectsPool, function(key, item)
    {
      item.isVisible = false;
      item.weight = 0;
      scope.applyRenderState(item);
    });
  }

  refreshVisible(modelList, options = {})
  {
    var desiredByHash = new Map();
    var nowMs = this.getNowMs();
    var retainVisibleMs = Number(options.retainVisibleMs !== undefined ? options.retainVisibleMs : 0);
    var missTolerance = Number(options.missTolerance !== undefined ? options.missTolerance : 0);
    var attachVisibleToScene = Boolean(options.attachVisibleToScene);
    var detachHiddenFromScene = Boolean(options.detachHiddenFromScene);
    var renderRoot = options.renderRoot || (this.options && this.options.sceneMgr ? this.options.sceneMgr.rootScene : null);
    var fallbackVisibleHashes = options.fallbackVisibleHashes instanceof Set
      ? options.fallbackVisibleHashes
      : new Set(options.fallbackVisibleHashes || []);
    var stats = {
      desiredCount: 0,
      fallbackVisibleCount: 0,
      visibleCount: 0,
      hiddenCount: 0,
      residentCount: 0,
      detachedCount: 0,
    };

    for (var i = 0; i < modelList.length; ++i)
    {
      var modelDesc = this.options.sceneMgr.decodeModelInfo(modelList[i]);
      
      if (modelDesc != null)
      {
        desiredByHash.set(modelDesc.hash, {
          shouldShow: modelDesc.url != null,
          weight: modelList[i].weight,
        });
      }
    }
    stats.desiredCount = desiredByHash.size;

    var scope = this;
    this.traverseObj(this.objectsPool, function(key, item)
    {
      stats.residentCount++;
      var desired = desiredByHash.get(key);
      var fallbackVisible = desired == null && fallbackVisibleHashes.has(key);
      if (fallbackVisible)
      {
        desired = {
          shouldShow: true,
          weight: item.weight,
        };
      }

      if (desired)
      {
        if (desired.shouldShow && item.meshObject && item.isInScene == false && attachVisibleToScene && renderRoot)
        {
          renderRoot.add(item.meshObject);
          item.isInScene = true;
        }
        item.isVisible = desired.shouldShow;
        item.weight = desired.weight;
        item.missedUpdates = 0;
        if (item.isVisible)
        {
          item.lastVisibleAt = nowMs;
          stats.visibleCount++;
          if (fallbackVisible) stats.fallbackVisibleCount++;
        }
        else
        {
          stats.hiddenCount++;
          if (detachHiddenFromScene && item.isInScene && item.meshObject)
          {
            if (scope.detachFromScene(item, key))
            {
              stats.detachedCount++;
            }
            return;
          }
        }
        scope.applyRenderState(item);
        return;
      }

      var wasVisible = Boolean(item.isVisible && item.isInScene && item.meshObject);
      item.weight = 0;
      item.missedUpdates = Number(item.missedUpdates || 0) + 1;

      var withinRetainWindow = retainVisibleMs > 0 && (nowMs - Number(item.lastVisibleAt || 0)) <= retainVisibleMs;
      var withinMissTolerance = missTolerance > 0 && item.missedUpdates <= missTolerance;
      item.isVisible = wasVisible && (withinRetainWindow || withinMissTolerance);
      if (item.isVisible) stats.visibleCount++;
      else
      {
        stats.hiddenCount++;
        if (detachHiddenFromScene && item.isInScene && item.meshObject)
        {
          if (scope.detachFromScene(item, key))
          {
            stats.detachedCount++;
          }
          return;
        }
      }
      scope.applyRenderState(item);
    });

    return stats;
  }

  reset()
  {
    this.objectsPool = {};
  }

  clearAll()
  {
    var scope = this;
    this.traverseObj(this.objectsPool, function(key, item)
    {
      if (!item || !item.meshObject) return;
      if (item.meshObject.parent)
      {
        item.meshObject.removeFromParent();
      }
      scope.releaseItem(item, key);
      scope.notifyResidentChanged(key);
    });
    this.objectsPool = {};
  }

  pruneToHashes(allowedHashes)
  {
    var scope = this;
    var keepSet = allowedHashes instanceof Set ? allowedHashes : new Set(allowedHashes || []);
    this.traverseObj(this.objectsPool, function(key, item)
    {
      if (keepSet.has(key))
      {
        return;
      }

      if (!item || !item.meshObject)
      {
        scope.notifyResidentChanged(key);
        delete scope.objectsPool[key];
        return;
      }

      if (item.meshObject.parent)
      {
        item.meshObject.removeFromParent();
      }

      scope.releaseItem(item, key);
      scope.notifyResidentChanged(key);
      delete scope.objectsPool[key];
    });
  }

  getSnapshot()
  {
    this.getStatsDetail();
    return {
      memoryUsedKB: this.statsDetail.memoryUsed,
      visibleNums: this.statsDetail.visibleNums,
      invisibleNums: this.statsDetail.invisibleNums,
      releasedNums: this.statsDetail.releasedNums,
      inSceneNums: this.statsDetail.inSceneNums,
      totalNums: this.statsDetail.totalNums,
    };
  }
}
