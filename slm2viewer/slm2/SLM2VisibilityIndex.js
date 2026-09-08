import { joinUrlPath } from './SLM2RuntimeAssets.js';

export class SLM2VisibilityIndex {
  _updateRuntimeVisibilityFrustum(cameraOverride = null)
  {
    var filterCamera = cameraOverride || (this.runtimeFrustumFilterSource === 'active-camera' ? this.activeCamera : this.backCamera);
    if (filterCamera == null)
    {
      return null;
    }

    filterCamera.updateMatrixWorld();
    this._runtimeFrustumMatrix.multiplyMatrices(filterCamera.projectionMatrix, filterCamera.matrixWorldInverse);
    this._runtimeFrustum.setFromProjectionMatrix(this._runtimeFrustumMatrix);
    return this._runtimeFrustum;
  }

  _intersectsFrustumWithCenterSize(bounds)
  {
    if (bounds == null || bounds.center == null || bounds.size == null)
    {
      return true;
    }

    this._runtimeTempCenter.set(bounds.center[0], bounds.center[1], bounds.center[2]);
    this._runtimeTempSize.set(bounds.size[0], bounds.size[1], bounds.size[2]);
    this._runtimeTempBox.setFromCenterAndSize(this._runtimeTempCenter, this._runtimeTempSize);
    return this._runtimeFrustum.intersectsBox(this._runtimeTempBox);
  }

  _intersectsFrustumWithMinMax(aabb)
  {
    if (aabb == null || aabb.min == null || aabb.max == null)
    {
      return true;
    }

    this._runtimeTempBox.min.set(aabb.min[0], aabb.min[1], aabb.min[2]);
    this._runtimeTempBox.max.set(aabb.max[0], aabb.max[1], aabb.max[2]);
    return this._runtimeFrustum.intersectsBox(this._runtimeTempBox);
  }

  _globalGlbIntersectsCurrentFrustum(globalGlbId)
  {
    var glbRecord = this.globalGlbVisibilityRecords[globalGlbId];

    if (glbRecord == null)
    {
      return true;
    }

    if (!this._intersectsFrustumWithMinMax(glbRecord.aabb))
    {
      return false;
    }

    var componentIds = this.globalGlbToComponentIds[globalGlbId] || [];
    if (componentIds.length === 0)
    {
      return true;
    }

    for (var c = 0; c < componentIds.length; ++c)
    {
      var componentRecord = this.componentVisibilityRecords[componentIds[c]];
      if (componentRecord != null && this._intersectsFrustumWithCenterSize(componentRecord.bounds))
      {
        return true;
      }
    }

    return false;
  }

  _filterVisibilityByCameraFrustum(modelList, weightList, idMode)
  {
    var sourceWeights = Array.isArray(weightList) ? weightList : [];
    var sourceIdMode = idMode || this.defaultVisibilityIdMode;
    if (this.useNeuralPVS && sourceIdMode === 'global-glb')
    {
      return {
        modelList: modelList,
        weightList: sourceWeights,
      };
    }

    if (!this.runtimeFrustumFilterEnabled || !Array.isArray(modelList) || modelList.length === 0 || this.runtimeVisibilityMeta == null)
    {
      return {
        modelList: modelList,
        weightList: sourceWeights,
      };
    }

    this._updateRuntimeVisibilityFrustum();

    var filteredIds = [];
    var filteredWeights = [];
    var aabbRejectedCount = 0;
    var componentRejectedCount = 0;
    var keptByMissingMeta = 0;
    var rejectedPreview = [];

    if (sourceIdMode === 'global-glb')
    {
      for (var i = 0; i < modelList.length; ++i)
      {
        var globalGlbId = modelList[i];
        var glbRecord = this.globalGlbVisibilityRecords[globalGlbId];

        if (glbRecord == null)
        {
          filteredIds.push(globalGlbId);
          filteredWeights.push(sourceWeights[i]);
          keptByMissingMeta++;
          continue;
        }

        if (!this._intersectsFrustumWithMinMax(glbRecord.aabb))
        {
          aabbRejectedCount++;
          if (rejectedPreview.length < 12) rejectedPreview.push({ id: globalGlbId, reason: 'glb-aabb' });
          continue;
        }

        var componentIds = this.globalGlbToComponentIds[globalGlbId] || [];
        var isVisibleInFrustum = componentIds.length === 0;
        for (var c = 0; c < componentIds.length && !isVisibleInFrustum; ++c)
        {
          var componentRecord = this.componentVisibilityRecords[componentIds[c]];
          if (componentRecord != null && this._intersectsFrustumWithCenterSize(componentRecord.bounds))
          {
            isVisibleInFrustum = true;
          }
        }

        if (isVisibleInFrustum)
        {
          filteredIds.push(globalGlbId);
          filteredWeights.push(sourceWeights[i]);
        }
        else
        {
          componentRejectedCount++;
          if (rejectedPreview.length < 12) rejectedPreview.push({ id: globalGlbId, reason: 'component-aabb' });
        }
      }
    }
    else if (sourceIdMode === 'component')
    {
      for (var j = 0; j < modelList.length; ++j)
      {
        var componentId = modelList[j];
        var componentInfo = this.componentVisibilityRecords[componentId];
        if (componentInfo == null || this._intersectsFrustumWithCenterSize(componentInfo.bounds))
        {
          filteredIds.push(componentId);
          filteredWeights.push(sourceWeights[j]);
        }
      }
    }
    else
    {
      return {
        modelList: modelList,
        weightList: sourceWeights,
      };
    }

    if (this.runtimeFrustumFilterDebug)
    {
      console.log('[SLM2Loader] Frustum post-filter idMode=' + sourceIdMode +
        ' before=' + modelList.length + ' after=' + filteredIds.length +
        ' camera=' + this.runtimeFrustumFilterSource, {
          aabbRejectedCount: aabbRejectedCount,
          componentRejectedCount: componentRejectedCount,
          keptByMissingMeta: keptByMissingMeta,
          rejectedPreview: rejectedPreview,
        });
    }

    return {
      modelList: filteredIds,
      weightList: filteredWeights,
    };
  }

  _getMaxModelId(idMode)
  {
    if (idMode === 'global-glb' || idMode === 'global-glb-priority')
    {
      return this.globalGlbEntries.length > 0 ? this.globalGlbEntries.length - 1 : -1;
    }

    return this.sceneConfig.groups[this.sceneConfig.groups.length - 1].idRange[1];
  }

  decodeGlobalGlbInfo(modelInfo)
  {
    if (modelInfo.id == null || this.globalGlbEntries.length == 0)
    {
      return null;
    }

    var entry = this.globalGlbEntries[modelInfo.id];
    if (entry == undefined)
    {
      return null;
    }

    var lodPath = this.MeshLodLevel >= 0 ? ('LOD' + this.MeshLodLevel) : 'raw';
    var glbBaseUrl = this.glbResourcesBaseUrl || this.resourcesBaseUrl;
    // Grouped indexes carry a path rooted at the scene GLB directory. Keep the
    // old task/base fallback for legacy indexes that do not provide it.
    var indexedPath = entry.path || ('task-' + entry.taskId + '/glb/' + lodPath + '/sub_' + entry.baseId + '.glb');
    var modelUrl = joinUrlPath(glbBaseUrl, indexedPath);
    var isToLoad = modelInfo.weight == null ? true : modelInfo.weight > 0;

    return {
      hash: entry.hash,
      url: isToLoad ? modelUrl : null,
      weight: modelInfo.weight,
      group: entry.taskId,
      globalGlbId: entry.globalId,
      baseId: entry.baseId,
      prefetch: Boolean(modelInfo.prefetch),
      deferredVisible: Boolean(modelInfo.deferredVisible),
      predictionEpoch: modelInfo.predictionEpoch != null ? Number(modelInfo.predictionEpoch) : null,
      priorityInfo: modelInfo.priorityInfo || null,
    };
  }
  decodeModelInfo(modelInfo)
  {
    var idMode = modelInfo.idMode || this.defaultVisibilityIdMode;

    if (idMode === 'global-glb' || idMode === 'global-glb-priority')
    {
      return this.decodeGlobalGlbInfo(modelInfo);
    }

    if (modelInfo.id == null)
    {
      return null;
    }

    var onlyShowInstanced = false;

    var matchedGroup = null;
    var matchedGroupId = 0;

    for (var i = 0; i < this.sceneConfig.groups.length; ++i)
    {
      if (modelInfo.id >= this.sceneConfig.groups[i].idRange[0] && 
        modelInfo.id <= this.sceneConfig.groups[i].idRange[1])
      {
        matchedGroup = this.sceneConfig.groups[i];
        matchedGroupId = i;
        break;
      }
    }

    var isToLoad = true;

    if (matchedGroup != null)
    {
      var baseId = modelInfo.id - matchedGroup.idRange[0];

      if (matchedGroup.instances[baseId] != undefined) // 检查是否有实例化对象
      {
        baseId = matchedGroup.instances[baseId];
      }
      else
      {
        if (onlyShowInstanced)
        {
          isToLoad = false;
        }
      }

      var hashCode = matchedGroupId + '-' + baseId;

      var lodPath = this.MeshLodLevel >= 0 ? ('LOD' + this.MeshLodLevel) : 'raw';

      var glbBaseUrl = this.glbResourcesBaseUrl || this.resourcesBaseUrl;
      var modelUrl = joinUrlPath(glbBaseUrl, "task-" + matchedGroupId + "/glb/" + lodPath + "/sub_" + baseId + ".glb");

      if (modelInfo.weight < 10)
      {
        isToLoad = false;
      }

      var decoded =
      {
        hash: hashCode,
        url: isToLoad ? modelUrl: null,
        weight: modelInfo.weight,
        group: matchedGroupId,
        prefetch: Boolean(modelInfo.prefetch),
        deferredVisible: Boolean(modelInfo.deferredVisible),
        predictionEpoch: modelInfo.predictionEpoch != null ? Number(modelInfo.predictionEpoch) : null,
      };

      return decoded;
    }

    return null;
  }
}
