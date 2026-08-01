import { Box3, Quaternion, Vector3 } from 'three';
import { FRONTEND_RENDER_FOV_Y_DEG } from './neuralPvsFovProtocol.js';

const _cameraPos = new Vector3();
const _center = new Vector3();
const _size = new Vector3();
const _cornerWorld = new Vector3();
const _cornerCamera = new Vector3();
const _projectedCenter = new Vector3();
const _box = new Box3();

const DEFAULTS = {
  enterProjectedAreaThreshold: 0.00012,
  exitProjectedAreaThreshold: 0.000035,
  desktopMinUpdateMs: 50,
  mobileMinUpdateMs: 100,
  cameraPositionThreshold: 1.5,
  cameraAngleThresholdRad: Math.PI / 180,
  flatLargeAspectRatio: 20,
  flatLargeExtent: 500,
};

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

export class RenderVisibilitySystem {
  constructor(loader, options = {}) {
    this.loader = loader;
    this.options = { ...DEFAULTS, ...options };
    this.currentIdMode = null;
    this.currentEpoch = 0;
    this.workingSetHashes = new Set();
    this.workingSetWeights = new Map();
    this.activeEvaluationHashes = new Set();
    this.visibleHashes = new Set();
    this.retainUntilByHash = new Map();
    this.dirtyHashes = new Set();
    this.lastUpdateAt = 0;
    this.lastCameraPosition = new Vector3(Number.NaN, Number.NaN, Number.NaN);
    this.lastCameraQuaternion = new Quaternion();
    this.hasLastCamera = false;
    this.lastStats = null;
  }

  configure(options = {}) {
    this.options = { ...this.options, ...options };
  }

  clear() {
    this.currentIdMode = null;
    this.currentEpoch = 0;
    this.workingSetHashes = new Set();
    this.workingSetWeights = new Map();
    this.activeEvaluationHashes = new Set();
    this.visibleHashes = new Set();
    this.retainUntilByHash = new Map();
    this.dirtyHashes = new Set();
    this.lastUpdateAt = 0;
    this.hasLastCamera = false;
  }

  setWorkingSet(modelInfos, idMode, epoch = 0) {
    this.currentIdMode = idMode || null;
    this.currentEpoch = Number(epoch || 0);
    this.workingSetHashes = new Set();
    this.workingSetWeights = new Map();

    for (const modelInfo of modelInfos || []) {
      const decoded = this.loader.decodeModelInfo(modelInfo);
      if (!decoded || !decoded.hash) continue;
      this.workingSetHashes.add(decoded.hash);
      this.workingSetWeights.set(decoded.hash, Number(modelInfo.weight || 0));
      this.dirtyHashes.add(decoded.hash);
    }
  }

  markResidentChanged(hash) {
    if (hash == null) return;
    this.dirtyHashes.add(hash);
  }

  syncVisibleResidentsFromCache() {
    const pool = this.loader && this.loader.modelCacheMgr ? this.loader.modelCacheMgr.objectsPool : null;
    if (!pool) return;

    for (const hash in pool) {
      const item = pool[hash];
      if (item && item.meshObject && item.isVisible && item.isInScene) {
        this.visibleHashes.add(hash);
        this.dirtyHashes.add(hash);
      }
    }
  }

  _isNeuralGlobalMode() {
    return this.loader.useNeuralPVS && (this.currentIdMode === 'global-glb' || this.currentIdMode === 'global-glb-priority');
  }

  _isMobilePerfMode() {
    return this.loader && this.loader.cpuPerfMode === 'mobile';
  }

  _getMinUpdateIntervalMs() {
    return this._isMobilePerfMode()
      ? this.options.mobileMinUpdateMs
      : this.options.desktopMinUpdateMs;
  }

  _hasCameraMovedEnough(camera) {
    if (!this.hasLastCamera) {
      return true;
    }

    const positionDelta = camera.position.distanceTo(this.lastCameraPosition);
    const angleDelta = camera.quaternion.angleTo(this.lastCameraQuaternion);
    return positionDelta >= this.options.cameraPositionThreshold ||
      angleDelta >= this.options.cameraAngleThresholdRad;
  }

  _rememberCamera(camera) {
    this.lastCameraPosition.copy(camera.position);
    this.lastCameraQuaternion.copy(camera.quaternion);
    this.hasLastCamera = true;
  }

  _hasExpiredRetain(nowMs) {
    for (const retainUntil of this.retainUntilByHash.values()) {
      if (retainUntil <= nowMs) return true;
    }
    return false;
  }

  _buildEvaluationSet(nowMs) {
    const hashes = new Set();

    for (const hash of this.workingSetHashes) hashes.add(hash);
    for (const hash of this.visibleHashes) hashes.add(hash);
    for (const hash of this.dirtyHashes) hashes.add(hash);
    for (const hash of this.retainUntilByHash.keys()) hashes.add(hash);

    this.activeEvaluationHashes = hashes;
    return hashes;
  }

  _setVisible(item, hash, visible, weight, nowMs) {
    item.isVisible = visible;
    if (weight !== undefined) {
      item.weight = weight;
    }

    if (visible) {
      item.lastVisibleAt = nowMs;
      this.visibleHashes.add(hash);
    } else {
      this.visibleHashes.delete(hash);
    }

    // Hidden residents must leave the render scene. Toggling Object3D.visible
    // alone still keeps their geometry in the scene traversal and can retain
    // the previous frame's draw/update cost.
    if (!visible && item.isInScene && item.meshObject &&
        this.loader.modelCacheMgr &&
        typeof this.loader.modelCacheMgr.detachFromScene === 'function') {
      const detached = this.loader.modelCacheMgr.detachFromScene(item, hash);
      if (detached) {
        this.detachedCountThisUpdate = Number(this.detachedCountThisUpdate || 0) + 1;
        return;
      }
    }

    this.loader.modelCacheMgr.applyRenderState(item);
  }

  _getGlbRecord(hash) {
    const globalGlbId = this.loader.globalGlbHashToId[hash];
    if (globalGlbId == null) {
      return null;
    }

    return this.loader.globalGlbVisibilityRecords[globalGlbId] || null;
  }

  _boxFromGlbRecord(glbRecord) {
    if (!glbRecord || !glbRecord.aabb || !glbRecord.aabb.min || !glbRecord.aabb.max) {
      return null;
    }

    _box.min.set(glbRecord.aabb.min[0], glbRecord.aabb.min[1], glbRecord.aabb.min[2]);
    _box.max.set(glbRecord.aabb.max[0], glbRecord.aabb.max[1], glbRecord.aabb.max[2]);
    return _box;
  }

  _isFlatLargeBox(box) {
    box.getSize(_size);
    const axes = [_size.x, _size.y, _size.z].filter((value) => value > 1e-3).sort((a, b) => a - b);
    if (axes.length < 2) return false;
    const minAxis = axes[0];
    const maxAxis = axes[axes.length - 1];
    return maxAxis >= this.options.flatLargeExtent &&
      (maxAxis / Math.max(minAxis, 1e-3)) >= this.options.flatLargeAspectRatio;
  }

  _fastProjectedAreaForBox(box, camera) {
    box.getCenter(_center);
    box.getSize(_size);
    const radius = Math.max(_size.length() * 0.5, 1e-6);
    _cornerCamera.copy(_center);
    camera.worldToLocal(_cornerCamera);

    if (_cornerCamera.z >= 0) {
      return 0;
    }

    const near = Math.max(Number(camera.near || 0.1), 1e-3);
    const forwardDistance = Math.max(-_cornerCamera.z, near);
    const fovRad = (Number(camera.fov || FRONTEND_RENDER_FOV_Y_DEG) * Math.PI) / 180;
    const tanHalfFov = Math.max(Math.tan(fovRad * 0.5), 1e-6);
    const aspect = Math.max(Number(camera.aspect || 1), 1e-6);
    const radiusY = radius / (forwardDistance * tanHalfFov);
    const radiusX = radiusY / aspect;
    const projectedCenter = _projectedCenter.copy(_center).project(camera);
    const centerInViewport = projectedCenter.x >= -1 && projectedCenter.x <= 1 &&
      projectedCenter.y >= -1 && projectedCenter.y <= 1 &&
      projectedCenter.z >= -1 && projectedCenter.z <= 1;
    const centerVisibility = centerInViewport ? 1 : 0.25;
    return clamp01(Math.PI * radiusX * radiusY * 0.25 * centerVisibility);
  }

  _needsPreciseProjection(box, camera) {
    camera.getWorldPosition(_cameraPos);
    if (box.containsPoint(_cameraPos)) {
      return true;
    }

    if (this._isFlatLargeBox(box)) {
      return true;
    }

    box.getCenter(_center);
    box.getSize(_size);
    _cornerCamera.copy(_center);
    camera.worldToLocal(_cornerCamera);
    const radius = Math.max(_size.length() * 0.5, 1e-6);
    const near = Math.max(Number(camera.near || 0.1), 1e-3);
    return (-_cornerCamera.z - radius) <= near;
  }

  _preciseProjectedAreaForBox(box, camera) {
    camera.getWorldPosition(_cameraPos);
    if (box.containsPoint(_cameraPos)) {
      return 1;
    }

    const near = Math.max(Number(camera.near || 0.1), 1e-3);
    const fovRad = (Number(camera.fov || FRONTEND_RENDER_FOV_Y_DEG) * Math.PI) / 180;
    const tanHalfFov = Math.max(Math.tan(fovRad * 0.5), 1e-6);
    const aspect = Math.max(Number(camera.aspect || 1), 1e-6);

    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    for (let ix = 0; ix < 2; ix += 1) {
      for (let iy = 0; iy < 2; iy += 1) {
        for (let iz = 0; iz < 2; iz += 1) {
          _cornerWorld.set(
            ix === 0 ? box.min.x : box.max.x,
            iy === 0 ? box.min.y : box.max.y,
            iz === 0 ? box.min.z : box.max.z,
          );
          _cornerCamera.copy(_cornerWorld);
          camera.worldToLocal(_cornerCamera);

          const clampedZ = Math.min(_cornerCamera.z, -near);
          const forwardDistance = Math.max(-clampedZ, near);
          const ndcX = (_cornerCamera.x / forwardDistance) / (tanHalfFov * aspect);
          const ndcY = (_cornerCamera.y / forwardDistance) / tanHalfFov;

          if (ndcX < minX) minX = ndcX;
          if (ndcX > maxX) maxX = ndcX;
          if (ndcY < minY) minY = ndcY;
          if (ndcY > maxY) maxY = ndcY;
        }
      }
    }

    if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) {
      return 0;
    }

    const clippedMinX = Math.max(-1, minX);
    const clippedMaxX = Math.min(1, maxX);
    const clippedMinY = Math.max(-1, minY);
    const clippedMaxY = Math.min(1, maxY);

    if (clippedMaxX <= clippedMinX || clippedMaxY <= clippedMinY) {
      return 0;
    }

    const ndcArea = (clippedMaxX - clippedMinX) * (clippedMaxY - clippedMinY);
    return Math.max(0, Math.min(1, ndcArea * 0.25));
  }

  _projectedAreaForHash(hash, camera) {
    const glbRecord = this._getGlbRecord(hash);
    const box = this._boxFromGlbRecord(glbRecord);
    if (!box) {
      return { projectedArea: 1, precise: false };
    }

    const precise = this._needsPreciseProjection(box, camera);
    return {
      projectedArea: precise
        ? this._preciseProjectedAreaForBox(box, camera)
        : this._fastProjectedAreaForBox(box, camera),
      precise,
    };
  }

  _intersectsGlbAabb(hash) {
    const glbRecord = this._getGlbRecord(hash);
    if (!glbRecord) {
      return true;
    }
    return this.loader._intersectsFrustumWithMinMax(glbRecord.aabb);
  }

  update(options = {}) {
    if (!this._isNeuralGlobalMode()) {
      return null;
    }

    // Resident mode is an explicit debugging/inspection policy. It must not
    // be undone by the normal culling reconciler on the next animation frame.
    if (this.loader.getNeuralRenderPolicy && this.loader.getNeuralRenderPolicy() === 'resident') {
      this.lastStats = {
        ...(this.lastStats || {}),
        skippedByPolicy: true,
        durationMs: 0,
      };
      return this.lastStats;
    }

    const camera = this.loader.activeCamera;
    if (!camera) {
      return null;
    }

    const nowMs = this.loader.modelCacheMgr.getNowMs();
    const startMs = typeof performance !== 'undefined' ? performance.now() : nowMs;
    const force = Boolean(options.force);
    const minIntervalMs = this._getMinUpdateIntervalMs();
    const elapsedMs = this.lastUpdateAt > 0 ? nowMs - this.lastUpdateAt : Number.POSITIVE_INFINITY;
    const cameraMoved = this._hasCameraMovedEnough(camera);
    const hasDirty = this.dirtyHashes.size > 0;
    const hasExpiredRetain = this._hasExpiredRetain(nowMs);

    if (!force && !hasDirty && !hasExpiredRetain && elapsedMs < minIntervalMs) {
      this.lastStats = {
        ...(this.lastStats || {}),
        skippedByGate: true,
        durationMs: 0,
      };
      return this.lastStats;
    }

    if (!force && !hasDirty && !hasExpiredRetain && !cameraMoved) {
      this.lastStats = {
        ...(this.lastStats || {}),
        skippedByGate: true,
        durationMs: 0,
      };
      return this.lastStats;
    }

    this.loader._updateRuntimeVisibilityFrustum(camera);

    const pool = this.loader.modelCacheMgr && this.loader.modelCacheMgr.objectsPool
      ? this.loader.modelCacheMgr.objectsPool
      : {};
    const retainVisibleMs = Math.max(0, Number(this.loader.getNeuralRenderRetainMs ? this.loader.getNeuralRenderRetainMs() : 0));
    const evaluationHashes = this._buildEvaluationSet(nowMs);
    this.detachedCountThisUpdate = 0;

    let visibleCount = 0;
    let frustumRejected = 0;
    let areaRejected = 0;
    let hiddenOutsideWorkingSet = 0;
    let keptVisibleOutsideWorkingSet = 0;
    let retainedByDelay = 0;
    let evaluatedCount = 0;
    let preciseProjectionCount = 0;
    let fastProjectionCount = 0;
    let missingResidentCount = 0;

    for (const hash of evaluationHashes) {
      const item = pool[hash];
      if (!item || !item.meshObject) {
        this.visibleHashes.delete(hash);
        this.retainUntilByHash.delete(hash);
        this.dirtyHashes.delete(hash);
        missingResidentCount++;
        continue;
      }

      evaluatedCount++;
      const wasVisible = Boolean(item.isVisible && item.isInScene);
      const isInWorkingSet = this.workingSetHashes.has(hash);

      if (!isInWorkingSet) {
        let retainUntil = this.retainUntilByHash.get(hash) || 0;

        if (wasVisible && retainVisibleMs > 0 && retainUntil <= nowMs) {
          retainUntil = nowMs + retainVisibleMs;
          this.retainUntilByHash.set(hash, retainUntil);
        }

        if (wasVisible && retainUntil > nowMs) {
          const inFrustum = this._intersectsGlbAabb(hash);
          if (inFrustum) {
            const areaMetric = this._projectedAreaForHash(hash, camera);
            if (areaMetric.precise) {
              preciseProjectionCount++;
            } else {
              fastProjectionCount++;
            }

            if (areaMetric.projectedArea >= this.options.exitProjectedAreaThreshold) {
              this._setVisible(item, hash, true, Math.max(item.weight || 0, areaMetric.projectedArea), nowMs);
              retainedByDelay++;
              visibleCount++;
              this.dirtyHashes.delete(hash);
              continue;
            }
          }
        }
        if (wasVisible) {
          hiddenOutsideWorkingSet++;
        }
        this.retainUntilByHash.delete(hash);
        this._setVisible(item, hash, false, 0, nowMs);
        this.dirtyHashes.delete(hash);
        continue;
      }

      this.retainUntilByHash.delete(hash);

      const inFrustum = this._intersectsGlbAabb(hash);
      if (!inFrustum) {
        if (wasVisible && retainVisibleMs > 0) {
          this.retainUntilByHash.set(hash, nowMs + retainVisibleMs);
          this._setVisible(item, hash, true, item.weight, nowMs);
          retainedByDelay++;
          visibleCount++;
        } else {
          frustumRejected++;
          this._setVisible(item, hash, false, this.workingSetWeights.get(hash) || 0, nowMs);
        }
        this.dirtyHashes.delete(hash);
        continue;
      }

      const areaMetric = this._projectedAreaForHash(hash, camera);
      if (areaMetric.precise) {
        preciseProjectionCount++;
      } else {
        fastProjectionCount++;
      }
      const projectedArea = areaMetric.projectedArea;
      const threshold = wasVisible
        ? this.options.exitProjectedAreaThreshold
        : this.options.enterProjectedAreaThreshold;

      if (projectedArea < threshold) {
        if (wasVisible && retainVisibleMs > 0) {
          this.retainUntilByHash.set(hash, nowMs + retainVisibleMs);
          this._setVisible(item, hash, true, Math.max(this.workingSetWeights.get(hash) || 0, projectedArea), nowMs);
          retainedByDelay++;
          visibleCount++;
        } else {
          areaRejected++;
          this._setVisible(item, hash, false, Math.max(this.workingSetWeights.get(hash) || 0, projectedArea), nowMs);
        }
        this.dirtyHashes.delete(hash);
        continue;
      }

      if (!item.isInScene && this.loader.rootScene) {
        this.loader.rootScene.add(item.meshObject);
        item.isInScene = true;
      }

      this._setVisible(item, hash, true, Math.max(this.workingSetWeights.get(hash) || 0, projectedArea), nowMs);
      this.retainUntilByHash.delete(hash);
      this.dirtyHashes.delete(hash);
      visibleCount++;
    }

    for (const [hash, retainUntil] of Array.from(this.retainUntilByHash.entries())) {
      if (retainUntil <= nowMs && !this.workingSetHashes.has(hash)) {
        this.retainUntilByHash.delete(hash);
        const item = pool[hash];
        if (item && item.meshObject && item.isVisible) {
          hiddenOutsideWorkingSet++;
          this._setVisible(item, hash, false, 0, nowMs);
        }
      }
    }

    this.lastUpdateAt = nowMs;
    this._rememberCamera(camera);

    this.lastStats = {
      epoch: this.currentEpoch,
      workingSetSize: this.workingSetHashes.size,
      activeEvaluationSize: evaluationHashes.size,
      evaluatedCount,
      visibleCount,
      detachedCount: this.detachedCountThisUpdate,
      frustumRejected,
      areaRejected,
      hiddenOutsideWorkingSet,
      keptVisibleOutsideWorkingSet,
      retainedByDelay,
      preciseProjectionCount,
      fastProjectionCount,
      missingResidentCount,
      skippedByGate: false,
      durationMs: (typeof performance !== 'undefined' ? performance.now() : nowMs) - startMs,
    };

    return this.lastStats;
  }
}
