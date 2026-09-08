import { Box3, Color, OrthographicCamera, Vector3 } from 'three';
import { ViewerQueueDiagnostics } from './ViewerQueueDiagnostics.js';

const WHITE = new Color(0xffffff);

export class ViewerDiagnostics extends ViewerQueueDiagnostics {
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
    this.requestRender('prediction-debug-toggle');
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
    if (!state || state.disabled || !Array.isArray(state.activeSourceIndices))
    {
      return null;
    }

    const slotMap = new Map();
    const activeIndices = state.activeSourceIndices;
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

    if (state.type === 'static-batch')
    {
      const optimizer = this.slm2Loader?.staticSceneOptimizer;
      optimizer?.setHashHighlight(state.hash, null);
      optimizer?.setHashVisible(state.hash, state.wasVisible && state.wasInScene);
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
      if (item.staticBatchHash)
      {
        if (!state)
        {
          state = {
            type: 'static-batch',
            hash,
            wasVisible: Boolean(item.isVisible),
            wasInScene: Boolean(item.isInScene),
          };
          this.predictionDebugTouched.set(touchKey, state);
        }
        if (this.slm2Loader.staticSceneOptimizer.setHashHighlight(
          hash,
          this.predictionDebugMaterial.color,
        ))
        {
          marked++;
        }
        if (this.predictionDebugForceShow)
        {
          this.slm2Loader.staticSceneOptimizer.setHashVisible(hash, true);
        }
        continue;
      }
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
}
