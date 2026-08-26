import { Euler, Vector3 } from 'three';

const _forward = new Vector3();
const _right = new Vector3();
const _move = new Vector3();
const _up = new Vector3(0, 1, 0);

function isTouchLikeDevice() {
  if (typeof window === 'undefined') return false;
  if ('ontouchstart' in window) return true;
  if (typeof navigator !== 'undefined' && navigator.maxTouchPoints > 0) return true;
  return typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches;
}

export class TouchMoveController {
  constructor(viewer) {
    this.viewer = viewer;
    this.camera = viewer.activeCamera;
    this.controls = viewer.controls;
    this.enabled = false;
    this.touchCapable = isTouchLikeDevice();
    this.euler = new Euler(0, 0, 0, 'YXZ');
    this.moveAxis = { x: 0, y: 0 };
    this.lookAxis = { x: 0, y: 0 };
    this.lookSensitivity = 0.0026;
    this.moveSpeed = 0.06;
    this.maxRadiusPx = 46;
    this.overlayEl = null;
    this.leftThumb = null;
    this.rightThumb = null;

    if (this.touchCapable) {
      this._buildUI();
    }
  }

  _buildUI() {
    const overlay = document.createElement('div');
    overlay.className = 'mobile-touch-overlay';

    const createStick = (side) => {
      const zone = document.createElement('div');
      zone.className = `touch-joystick touch-joystick-${side}`;
      const thumb = document.createElement('div');
      thumb.className = 'touch-joystick-thumb';
      zone.appendChild(thumb);
      overlay.appendChild(zone);
      return { zone, thumb };
    };

    const left = createStick('left');
    const right = createStick('right');

    this.leftThumb = left.thumb;
    this.rightThumb = right.thumb;
    this._bindStick(left.zone, left.thumb, (x, y) => {
      this.moveAxis.x = x;
      this.moveAxis.y = y;
      this.viewer.requestRender('touch-move-input');
    });
    this._bindStick(right.zone, right.thumb, (x, y) => {
      this.lookAxis.x = x;
      this.lookAxis.y = y;
      this.viewer.requestRender('touch-look-input');
    });

    this.viewer.el.appendChild(overlay);
    this.overlayEl = overlay;
    this._updateOverlayVisibility();
    this.viewer.requestRender('touch-control-mode');
  }

  _bindStick(zone, thumb, onAxisChange) {
    let pointerId = null;

    const reset = () => {
      pointerId = null;
      thumb.style.transform = 'translate(0px, 0px)';
      onAxisChange(0, 0);
    };

    const updateFromEvent = (event) => {
      const rect = zone.getBoundingClientRect();
      const centerX = rect.left + rect.width * 0.5;
      const centerY = rect.top + rect.height * 0.5;
      const dx = event.clientX - centerX;
      const dy = event.clientY - centerY;
      const distance = Math.hypot(dx, dy);
      const scale = distance > this.maxRadiusPx ? this.maxRadiusPx / distance : 1;
      const clampedX = dx * scale;
      const clampedY = dy * scale;
      thumb.style.transform = `translate(${clampedX}px, ${clampedY}px)`;
      onAxisChange(clampedX / this.maxRadiusPx, clampedY / this.maxRadiusPx);
    };

    zone.addEventListener('pointerdown', (event) => {
      if (!this.enabled) return;
      pointerId = event.pointerId;
      zone.setPointerCapture(pointerId);
      event.preventDefault();
      updateFromEvent(event);
    });

    zone.addEventListener('pointermove', (event) => {
      if (!this.enabled || pointerId !== event.pointerId) return;
      event.preventDefault();
      updateFromEvent(event);
    });

    const endPointer = (event) => {
      if (pointerId !== event.pointerId) return;
      event.preventDefault();
      reset();
    };

    zone.addEventListener('pointerup', endPointer);
    zone.addEventListener('pointercancel', endPointer);
    zone.addEventListener('pointerleave', (event) => {
      if (pointerId !== event.pointerId || event.buttons !== 0) return;
      reset();
    });
  }

  _updateOverlayVisibility() {
    if (!this.overlayEl) return;
    this.overlayEl.style.display = this.enabled && this.touchCapable ? 'block' : 'none';
  }

  setEnabled(enabled) {
    this.enabled = Boolean(enabled);
    if (!this.enabled) {
      this.moveAxis.x = 0;
      this.moveAxis.y = 0;
      this.lookAxis.x = 0;
      this.lookAxis.y = 0;
      if (this.leftThumb) this.leftThumb.style.transform = 'translate(0px, 0px)';
      if (this.rightThumb) this.rightThumb.style.transform = 'translate(0px, 0px)';
    }
    this._updateOverlayVisibility();
  }

  _syncOrbitTarget() {
    if (!this.controls) return;
    _forward.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
    this.controls.target.copy(this.camera.position).add(_forward.multiplyScalar(10));
    this.controls.update();
  }

  update(dt) {
    if (!this.enabled || !this.touchCapable) return false;

    const dtMs = Number(dt || 0);
    if (dtMs <= 0) return false;

    let changed = false;

    if (this.lookAxis.x !== 0 || this.lookAxis.y !== 0) {
      this.euler.setFromQuaternion(this.camera.quaternion);
      this.euler.y -= this.lookAxis.x * dtMs * this.lookSensitivity;
      this.euler.x -= this.lookAxis.y * dtMs * this.lookSensitivity;
      const piHalf = Math.PI * 0.5;
      this.euler.x = Math.max(-piHalf, Math.min(piHalf, this.euler.x));
      this.camera.quaternion.setFromEuler(this.euler);
      this._syncOrbitTarget();
      changed = true;
    }

    if (this.moveAxis.x === 0 && this.moveAxis.y === 0) {
      return changed;
    }

    _forward.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
    _forward.y = 0;
    if (_forward.lengthSq() < 1e-6) {
      _forward.set(0, 0, -1);
    }
    _forward.normalize();

    _right.set(1, 0, 0).applyQuaternion(this.camera.quaternion);
    _right.y = 0;
    if (_right.lengthSq() < 1e-6) {
      _right.set(1, 0, 0);
    }
    _right.normalize();

    _move.copy(_forward).multiplyScalar(-this.moveAxis.y).addScaledVector(_right, this.moveAxis.x);
    if (_move.lengthSq() < 1e-6) {
      return changed;
    }

    _move.normalize();
    _move.addScaledVector(_up, 0);
    _move.multiplyScalar(dtMs * this.moveSpeed * this.viewer.keyboardMgr.speedMultiplier);
    this.camera.position.add(_move);
    this._syncOrbitTarget();
    return true;
  }

  hasActiveInput() {
    return Boolean(this.enabled && this.touchCapable && (
      this.moveAxis.x !== 0 || this.moveAxis.y !== 0
      || this.lookAxis.x !== 0 || this.lookAxis.y !== 0
    ));
  }
}
