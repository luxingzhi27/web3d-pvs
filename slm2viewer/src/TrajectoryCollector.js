import { Vector3 } from 'three';

export class TrajectoryCollector {
    constructor(viewer) {
        this.viewer = viewer;
        this.data = [];
        this.isRecording = false;
        this.recordInterval = 0.1; // 10Hz
        this.timeSinceLastRecord = 0;
        this.timeSinceLastBatch = 0;
        this.elapsedSeconds = 0;
        
        // UI Elements
        this.overlayEl = document.getElementById('trajectory-overlay');
        this.startBtn = document.getElementById('start-trajectory-btn');
        this.hudEl = document.getElementById('trajectory-hud');
        this.timerEl = document.getElementById('trajectory-timer');
        this.stopBtn = document.getElementById('stop-trajectory-btn');
        this.fullLoadToggleHud = document.getElementById('full-load-toggle-hud');
        this.neuralPVSToggleHud = document.getElementById('neural-pvs-toggle-hud');
        this.neuralDebugNoCacheToggleHud = document.getElementById('neural-debug-nocache-toggle-hud');
        this.trajectoryEnabled = this.viewer.paramJson &&
            (this.viewer.paramJson['trajectory'] === 'true' || this.viewer.paramJson['recordTrajectory'] === 'true');

        if (!this.trajectoryEnabled) {
            this.disableTrajectoryUI();
            return;
        }
        if (this.overlayEl) this.overlayEl.style.display = 'flex';

        if (this.viewer.slm2Loader) {
            if (this.neuralPVSToggleHud) {
                this.neuralPVSToggleHud.checked = Boolean(this.viewer.slm2Loader.useNeuralPVS);
            }
            if (this.neuralDebugNoCacheToggleHud) {
                this.neuralDebugNoCacheToggleHud.checked = Boolean(this.viewer.slm2Loader.neuralDebugNoCache);
            }
        }
        
        if (this.startBtn) {
            this.startBtn.addEventListener('click', () => this.start());
        }
        if (this.stopBtn) {
            this.stopBtn.addEventListener('click', () => this.stop());
        }
        
        const syncToggle = (e) => {
            const isChecked = e.target.checked;
            if (this.fullLoadToggleOverlay) this.fullLoadToggleOverlay.checked = isChecked;
            if (this.fullLoadToggleHud) this.fullLoadToggleHud.checked = isChecked;
            
            if (this.viewer.slm2Loader) {
                this.viewer.slm2Loader.fullLoadMode = isChecked;
                this.viewer.slm2Loader.hasFullLoaded = false;
            }
        };
        
        if (this.fullLoadToggleOverlay) {
            this.fullLoadToggleOverlay.addEventListener('change', syncToggle);
        }
        if (this.fullLoadToggleHud) {
            this.fullLoadToggleHud.addEventListener('change', syncToggle);
        }

        if (this.neuralPVSToggleHud) {
            this.neuralPVSToggleHud.addEventListener('change', (e) => {
                if (this.viewer.slm2Loader) {
                    this.viewer.slm2Loader.neuralPVSIdMode = 'global-glb-priority';
                    if (typeof this.viewer.slm2Loader.setUseNeuralPVS === 'function') {
                        this.viewer.slm2Loader.setUseNeuralPVS(e.target.checked);
                    } else {
                        this.viewer.slm2Loader.useNeuralPVS = e.target.checked;
                    }
                    console.log("[InstancePVS] Mode toggled:", e.target.checked, 'idMode=' + this.viewer.slm2Loader.neuralPVSIdMode);
                }
            });
        }

        if (this.neuralDebugNoCacheToggleHud) {
            this.neuralDebugNoCacheToggleHud.addEventListener('change', (e) => {
                if (this.viewer.slm2Loader) {
                    this.viewer.slm2Loader.neuralDebugNoCache = e.target.checked;
                    console.log('[InstancePVS] Neural debug no-cache toggled:', e.target.checked);
                }
            });
        }

        
        // Lock controls initially
        this.viewer.controls.enabled = false;
        if(this.viewer.keyboardMgr) this.viewer.keyboardMgr.enabled = false;
        if(this.viewer.touchMoveController) this.viewer.touchMoveController.setEnabled(false);
    }

    disableTrajectoryUI() {
        if (this.overlayEl) this.overlayEl.style.display = 'none';
        if (this.hudEl) this.hudEl.style.display = 'none';
        this.isRecording = false;
        if (this.viewer.controls) this.viewer.controls.enabled = true;
        if (this.viewer.keyboardMgr) this.viewer.keyboardMgr.enabled = true;
        if (this.viewer.touchMoveController) this.viewer.touchMoveController.setEnabled(true);
    }

    fallbackStart() {
        const bounds = this.viewer.sceneBounds;
        if (bounds && bounds.center && bounds.size) {
            const cy = bounds.center[1];
            // Center fallback
            const cameraOffset = [0, bounds.size[1] * 0.1, bounds.size[2] * 0.5];
            const camPos = new Vector3(bounds.center[0], cy + bounds.size[1] * 0.5 + cameraOffset[1], bounds.center[2] + cameraOffset[2]);
            const camTarget = new Vector3(bounds.center[0], cy, bounds.center[2]);
            this.viewer.setCamera(camPos, camTarget);
        }
    }

    async start() {
        if (this.isRecording) return;
        
        // Hide overlay immediately
        if(this.overlayEl) this.overlayEl.style.display = 'none';

        try {
            const res = await fetch('assets/custom_starts.json');
            if (res.ok) {
                const starts = await res.json();
                if (starts && starts.length > 0) {
                    const rIdx = Math.floor(Math.random() * starts.length);
                    const pt = starts[rIdx];
                    const camPos = new Vector3(pt.position.x, pt.position.y, pt.position.z);
                    const camTarget = new Vector3(pt.target.x, pt.target.y, pt.target.z);
                    this.viewer.setCamera(camPos, camTarget);
                    console.log("Started roaming from custom point index: ", rIdx);
                } else {
                    this.fallbackStart();
                }
            } else {
                this.fallbackStart();
            }
        } catch (e) {
            this.fallbackStart();
        }

        // Hide overlay, show HUD
        if(this.overlayEl) this.overlayEl.style.display = 'none';
        if(this.hudEl) this.hudEl.style.display = 'flex';
        
        // Sync logic - apply chosen loading strategy before starting
        const isChecked = this.fullLoadToggleOverlay ? this.fullLoadToggleOverlay.checked : false;
        if (this.viewer.slm2Loader) {
            this.viewer.slm2Loader.fullLoadMode = isChecked;
            this.viewer.slm2Loader.hasFullLoaded = false;
        }

        // Unlock controls
        this.viewer.controls.enabled = true;
        if(this.viewer.keyboardMgr) this.viewer.keyboardMgr.enabled = true;
        if(this.viewer.touchMoveController) this.viewer.touchMoveController.setEnabled(true);
        
        this.isRecording = true;
        this.data = [];
        this.elapsedSeconds = 0;
        this.timeSinceLastRecord = 0;
        this.timeSinceLastBatch = 0;
        this.updateTimerUI();
    }
    
    stop() {
        if (!this.isRecording) return;
        this.isRecording = false;
        
        // Lock controls again
        this.viewer.controls.enabled = false;
        if(this.viewer.keyboardMgr) this.viewer.keyboardMgr.enabled = false;
        if(this.viewer.touchMoveController) this.viewer.touchMoveController.setEnabled(false);
        
        if(this.hudEl) this.hudEl.style.display = 'none';
        if (this.overlayEl) {
            this.overlayEl.style.display = 'flex';
        }
        
        // Discard trailing data less than 1 minute
        const unuploadedFrames = this.data.length;
        this.data = [];
        
        if (unuploadedFrames > 0) {
            console.log(`Discarded ${unuploadedFrames} frames of incomplete 1-minute batch.`);
            alert("Roaming stopped. Partial track under 1 minute discarded.");
        } else {
            alert("Roaming stopped.");
        }
    }
    
    update(dt) {
        if (!this.isRecording) return;
        
        this.elapsedSeconds += dt;
        this.timeSinceLastBatch += dt;
        this.updateTimerUI();
        
        if (this.timeSinceLastBatch >= 60.0) {
            this.timeSinceLastBatch -= 60.0;
            if (this.data.length > 0) {
                this.uploadData(); 
            }
            this.data = []; 
        }
        
        // Record data at 10Hz
        this.timeSinceLastRecord += dt;
        if (this.timeSinceLastRecord >= this.recordInterval) {
            this.timeSinceLastRecord = 0;
            this.recordFrame();
        }
    }
    
    recordFrame() {
        const t = this.elapsedSeconds;
        const pos = this.viewer.activeCamera.position;
        const target = this.viewer.controls.target; 
        const rot = this.viewer.activeCamera.rotation;
        
        this.data.push({
            time: parseFloat(t.toFixed(3)),
            position: { x: pos.x, y: pos.y, z: pos.z },
            rotation: { x: rot.x, y: rot.y, z: rot.z },
            target: { x: target.x, y: target.y, z: target.z }
        });
    }
    
    generateCSV() {
        let csv = 'time,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,target_x,target_y,target_z\n';
        for (let i = 0; i < this.data.length; i++) {
            const f = this.data[i];
            csv += `${f.time},${f.position.x},${f.position.y},${f.position.z},${f.rotation.x},${f.rotation.y},${f.rotation.z},${f.target.x},${f.target.y},${f.target.z}\n`;
        }
        return csv;
    }
    
    updateTimerUI() {
        if (!this.timerEl) return;
        let totalSeconds = Math.max(0, Math.floor(this.elapsedSeconds));
        const m = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
        const s = (totalSeconds % 60).toString().padStart(2, '0');
        
        const uploads = Math.floor(this.elapsedSeconds / 60);
        this.timerEl.innerText = `${m}:${s} [Uploaded: ${uploads}]`;
    }
    
    async uploadData() {
        const csvContent = this.generateCSV();
        
        console.log(`Uploading 1-minute trajectory batch (${this.data.length} frames)...`);
        
        // Generate a downloadable file as backup
        try {
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            const filename = `trajectory_batch_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (e) {
            console.error("Failed to generate download backup", e);
        }
        
        // Try uploading to backend
        try {
            const response = await fetch('http://localhost:8050/api/trajectory/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ csvData: csvContent })
            });
            if (response.ok) {
                console.log("Data uploaded successfully.");
            } else {
                console.error("Failed to upload data.", response.statusText);
            }
        } catch (err) {
            console.error("Network error during upload.", err);
        }
    }
}

