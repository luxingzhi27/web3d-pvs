import * as ort from 'onnxruntime-web';

const ORT_WASM_MJS_PATH = './assets/ort/ort-wasm-simd-threaded.jsep.mjs';
const ORT_WASM_BINARY_PATH = './assets/ort/ort-wasm-simd-threaded.jsep.wasm';
const MODEL_EXTERNAL_DATA_PATH = 'neural_pvs.onnx.data';

/**
 * NeuralPVS: Neural Web3D Visibility Prediction System
 * 
 * Replaces the server-side BVH intersection with a local,
 * lightweight neural network inference in the browser via WebGPU.
 */
export class NeuralPVS {
  constructor(modelUrl) {
    this.modelUrl = modelUrl;
    this.modelDataUrl = modelUrl.replace(/\.onnx$/i, '.onnx.data');
    this.session = null;
    this.isReady = false;
    this.isPredicting = false;
    this.initError = null;
    this.initPromise = null;
    
    // Set webgpu backend as preferred, fallback to wasm
    ort.env.wasm.numThreads = 1; // Used if WebGPU is unavailable
    ort.env.wasm.wasmPaths = {
      mjs: ORT_WASM_MJS_PATH,
      wasm: ORT_WASM_BINARY_PATH,
    };
    
    this.boundsCenter = [-464.496, 26.124, 251.483];
    this.boundsHalf = [6784.186 / 2, 155.752 / 2, 5579.047 / 2];
    this.outputIdMode = 'global-glb';
  }

  async init() {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initError = null;
    console.log('[NeuralPVS] Initializing ONNX InferenceSession from:', this.modelUrl);
    console.log('[NeuralPVS] ORT runtime paths:', ort.env.wasm.wasmPaths);

    this.initPromise = Promise.all([
      fetch(this.modelUrl),
      fetch(this.modelDataUrl),
    ]).then(async ([modelResponse, externalDataResponse]) => {
      if (!modelResponse.ok) {
        throw new Error(`Failed to fetch model: ${modelResponse.status} ${modelResponse.statusText}`);
      }
      if (!externalDataResponse.ok) {
        throw new Error(`Failed to fetch external model data: ${externalDataResponse.status} ${externalDataResponse.statusText}`);
      }

      const modelBuffer = await modelResponse.arrayBuffer();
      const externalDataBuffer = await externalDataResponse.arrayBuffer();

      console.log('[NeuralPVS] Fetched model artifacts:', {
        modelBytes: modelBuffer.byteLength,
        externalDataBytes: externalDataBuffer.byteLength,
      });

      return ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['webgpu', 'wasm'],
        externalData: [{
          path: MODEL_EXTERNAL_DATA_PATH,
          data: externalDataBuffer,
        }],
      });
    }).then((session) => {
      this.session = session;
      this.isReady = true;
      const provider = Array.isArray(this.session.executionProviders) && this.session.executionProviders.length > 0
        ? this.session.executionProviders[0]
        : 'unknown';
      console.log('[NeuralPVS] Model loaded successfully! Provider:', provider);
      return session;
    }).catch((err) => {
      this.session = null;
      this.isReady = false;
      this.initError = err;
      this.initPromise = null;
      console.error('[NeuralPVS] Failed to load ONNX model:', err);
      return null;
    });

    return this.initPromise;
  }

  _normalizePos(x, y, z) {
    return [
      (x - this.boundsCenter[0]) / this.boundsHalf[0],
      (y - this.boundsCenter[1]) / this.boundsHalf[1],
      (z - this.boundsCenter[2]) / this.boundsHalf[2],
    ];
  }

  _encodeRot(rx, ry, rz) {
    // Return sin/cos encoded Euler angles
    return [
      Math.sin(rx), Math.sin(ry), Math.sin(rz),
      Math.cos(rx), Math.cos(ry), Math.cos(rz)
    ];
  }

  /**
   * Predict visible global GLB IDs from current camera pose.
   * @param {THREE.Vector3} position Camera position
   * @param {THREE.Euler} rotation Camera rotation
   * @returns {Array<number>} Array of visible global GLB IDs
   */
  async predict(position, rotation) {
    if (!this.isReady || this.isPredicting) return null;
    this.isPredicting = true;

    try {
      // 1. Prepare 9D input feature vector
      const posNorm = this._normalizePos(position.x, position.y, position.z);
      // THREE.Euler order should match Python's assumed [pitch, yaw, roll] -> [x, y, z]
      const rotEnc = this._encodeRot(rotation.x, rotation.y, rotation.z);
      
      const inputData = Float32Array.from([...posNorm, ...rotEnc]);
      const tensor = new ort.Tensor('float32', inputData, [1, 9]);

      // 2. Run Inference
      const feeds = { pose: tensor };
      const startTime = performance.now();
      const results = await this.session.run(feeds);
      const executionTime = performance.now() - startTime;

      // 3. Process logits -> probabilities -> threshold
      // The output tensor 'logits' has shape [1, num_visible_ids]
      const logits = results.logits.data; 
      const visibleIds = [];
      const weights = [];
      
      for (let i = 0; i < logits.length; i++) {
        // Since we use BCEWithLogitsLoss, probability > 0.5 means logit > 0.0
        if (logits[i] > 0.0) {
          visibleIds.push(i);
          // Assign max weight to ensure CacheMgr does not cull it
          weights.push(999); 
        }
      }

      console.log(`[NeuralPVS] Prediction took ${executionTime.toFixed(1)}ms. Visible ids: ${visibleIds.length} (${this.outputIdMode})`);
      
      this.isPredicting = false;
      return { modelList: visibleIds, weightList: weights, idMode: this.outputIdMode };
      
    } catch (err) {
      console.error('[NeuralPVS] Prediction error:', err);
      this.isPredicting = false;
      return null;
    }
  }
}

