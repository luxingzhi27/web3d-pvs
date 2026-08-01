import { Vector3, Euler, MathUtils } from 'three';

export class keyboardMgr {
  constructor(options) {
    this.camera = options.activeCamera;
    this.controls = options.controls;
    this.enabled = true;
    this.isTouchDevice = typeof window !== 'undefined' && (('ontouchstart' in window) || (typeof navigator !== 'undefined' && navigator.maxTouchPoints > 0) || (typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches));

    // ==========================================
    // HOW TO CHANGE SPEED: 
    // Default is 0.5. Increase this value to move faster (e.g. 1.0, 5.0).
    // ==========================================
    this.speedMultiplier = 0.15;
    this.euler = new Euler(0, 0, 0, 'YXZ');
    this.mouseSensitivity = 0.002;
    this.keyMap = {};

    this.initEvents();
  }

  initEvents() {
    document.addEventListener('keydown', (e) => this.keydown(e));
    document.addEventListener('keyup', (e) => this.keyup(e));

    // PointerLock events
    document.addEventListener('click', (e) => {
      if (this.isTouchDevice) return;
      if (this.enabled && e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT') {
        document.body.requestPointerLock();
      }
    });

    document.addEventListener('mousemove', (e) => this.onMouseMove(e));

    document.addEventListener('pointerlockchange', () => {
      if (document.pointerLockElement === document.body) {
        // Lock OrbitControls when FPS is active
        if (this.controls) this.controls.enabled = false;
      } else {
        // Restore OrbitControls
        if (this.controls && this.enabled) this.controls.enabled = true;
      }
    });
  }

  keydown(keyEvent) {
    if (!this.enabled) return;

    if (keyEvent.key.toLowerCase() == 'w') this.keyMap['forward'] = 1;
    else if (keyEvent.key.toLowerCase() == 's') this.keyMap['forward'] = -1;
    else if (keyEvent.key.toLowerCase() == 'a') this.keyMap['right'] = -1;
    else if (keyEvent.key.toLowerCase() == 'd') this.keyMap['right'] = 1;
    else if (keyEvent.key == 'Shift') this.keyMap['shift'] = 1;
    else if (keyEvent.key == ' ') this.keyMap['up'] = 1;
    else if (keyEvent.key == 'Control') this.keyMap['up'] = -1;
    else if (keyEvent.key == 'Alt') {
      document.exitPointerLock();
      keyEvent.preventDefault();
    }
  }

  keyup(keyEvent) {
    if (!this.enabled) return;

    if (keyEvent.key.toLowerCase() == 'w') this.keyMap['forward'] = 0;
    else if (keyEvent.key.toLowerCase() == 's') this.keyMap['forward'] = 0;
    else if (keyEvent.key.toLowerCase() == 'a') this.keyMap['right'] = 0;
    else if (keyEvent.key.toLowerCase() == 'd') this.keyMap['right'] = 0;
    else if (keyEvent.key == 'Shift') this.keyMap['shift'] = 0;
    else if (keyEvent.key == ' ') this.keyMap['up'] = 0;
    else if (keyEvent.key == 'Control') this.keyMap['up'] = 0;
  }

  onMouseMove(event) {
    if (!this.enabled || document.pointerLockElement !== document.body) return;

    const movementX = event.movementX || event.mozMovementX || event.webkitMovementX || 0;
    const movementY = event.movementY || event.mozMovementY || event.webkitMovementY || 0;

    this.euler.setFromQuaternion(this.camera.quaternion);

    this.euler.y -= movementX * this.mouseSensitivity;
    this.euler.x -= movementY * this.mouseSensitivity;

    const PI_2 = Math.PI / 2;
    this.euler.x = Math.max(-PI_2, Math.min(PI_2, this.euler.x));

    this.camera.quaternion.setFromEuler(this.euler);

    this.syncOrbitTarget();
  }

  syncOrbitTarget() {
    if (this.controls) {
      const forward = new Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      this.controls.target.copy(this.camera.position).add(forward.multiplyScalar(10));
      this.controls.update();
    }
  }

  update(dt) {
    if (!this.enabled) return;
    // update camera
    if (!this.controls && !this.camera) return;

    var forward = this.keyMap['forward'] || 0;
    var right = this.keyMap['right'] || 0;
    var up = this.keyMap['up'] || 0;
    if (forward == 0 && right == 0 && up == 0) return;

    // Sync vector calculations with camera's actual pointer-locked rotation
    var cameraForward = new Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    cameraForward.normalize();

    var cameraRight = new Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    cameraRight.normalize();

    var cameraUp = new Vector3(0, 1, 0);

    var moveSpeed = 50 * 0.001;

    var moveForVec = cameraForward.multiplyScalar(forward * dt * moveSpeed);
    var moveLeftVec = cameraRight.multiplyScalar(right * dt * moveSpeed);
    var moveUpVec = cameraUp.multiplyScalar(up * dt * moveSpeed);

    var moveVec = moveForVec.add(moveLeftVec).add(moveUpVec);

    var finalSpeed = 500 * 0.001 * this.speedMultiplier;
    moveVec.multiplyScalar(dt * finalSpeed);

    this.camera.position.add(moveVec);
    this.syncOrbitTarget();

    this.controls.update();
  }
}
