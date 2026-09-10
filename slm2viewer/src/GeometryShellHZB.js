// GPU runtime for the lossless geometry-shell HZB baseline.
// It reads only the final visible instance IDs and GLB flags back to JavaScript.

import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import {
  GEOMETRY_SHELL_HZB_SCHEMA,
  LINEAR_DEPTH_ENCODING,
  cameraProjectionParameters,
} from './GeometryShellHZBCore.js';
import {
  GEOMETRY_SHELL_DEPTH_FRAGMENT_SHADER,
  GEOMETRY_SHELL_DEPTH_VERTEX_SHADER,
  GEOMETRY_SHELL_MIP_SHADER,
  GEOMETRY_SHELL_QUERY_SHADER,
} from './GeometryShellHZBShaders.js';

const SOFTWARE_ADAPTER_PATTERN = /swiftshader|llvmpipe|softpipe|swrast|software/i;
const RENDER_UNIFORM_BYTES = 144;
const QUERY_UNIFORM_BYTES = 128;
export const GEOMETRY_SHELL_HZB_SAMPLE_COUNT = 1;

function nowMs() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function align4(value) {
  return (Number(value) + 3) & ~3;
}

function asArray(value, size, name) {
  if (value == null) throw new Error(`${name} is required.`);
  if (value.length !== size) throw new Error(`${name} must contain ${size} values.`);
  const result = Array.from(value, Number);
  if (result.some((item) => !Number.isFinite(item))) throw new Error(`${name} must be finite.`);
  return result;
}

function matrixElements(value, name) {
  if (value?.elements) return asArray(value.elements, 16, name);
  return asArray(value, 16, name);
}

function multiplyMatrix(a, b) {
  const out = new Array(16).fill(0);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      for (let k = 0; k < 4; k += 1) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
    }
  }
  return out;
}

function vectorElements(value, size, name) {
  if (value?.toArray) return asArray(value.toArray(), size, name);
  return asArray(value, size, name);
}

function cameraState(camera) {
  const position = vectorElements(camera.position, 3, 'camera.position');
  const view = camera.viewMatrix
    ? matrixElements(camera.viewMatrix, 'camera.viewMatrix')
    : camera.view
      ? matrixElements(camera.view, 'camera.view')
      : camera.matrixWorldInverse
        ? matrixElements(camera.matrixWorldInverse, 'camera.matrixWorldInverse')
        : null;
  if (!view) throw new Error('camera.viewMatrix or camera.matrixWorldInverse is required.');
  let viewProjection = camera.viewProjectionMatrix
    ? matrixElements(camera.viewProjectionMatrix, 'camera.viewProjectionMatrix')
    : camera.viewProj
      ? matrixElements(camera.viewProj, 'camera.viewProj')
      : null;
  if (!viewProjection && camera.projectionMatrix) {
    viewProjection = multiplyMatrix(matrixElements(camera.projectionMatrix, 'camera.projectionMatrix'), view);
  }
  if (!viewProjection) throw new Error('camera.viewProjectionMatrix or projectionMatrix is required.');
  const projection = cameraProjectionParameters(camera);
  return {
    position,
    view,
    viewProjection,
    ...projection,
  };
}

function urlFor(baseUrl, file) {
  return new URL(file, `${String(baseUrl).replace(/\/+$/, '')}/`).toString();
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: HTTP ${response.status}`);
  return response.json();
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: HTTP ${response.status}`);
  return response.arrayBuffer();
}

function createGpuBuffer(device, data, usage, label) {
  const bytes = data instanceof ArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, align4(bytes.byteLength)),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  if (bytes.byteLength > 0) device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

function validateDescriptor(segment, stream, name) {
  const count = Number(segment.count);
  const stride = Number(segment.stride || stream.stride);
  const byteLength = Number(segment.byteLength);
  const decodedByteLength = Number(segment.decodedByteLength);
  if (!Number.isInteger(count) || count < 0 || !Number.isInteger(stride) || stride <= 0
      || !Number.isInteger(byteLength) || byteLength < 0
      || decodedByteLength !== count * stride) {
    throw new Error(`${name} has an invalid meshopt segment descriptor.`);
  }
  return { count, stride, byteLength, decodedByteLength, offset: Number(segment.offset) };
}

async function decodeStream(meta, streamName, baseUrl, targetBytes, segmentOffset) {
  const stream = meta.streams?.[streamName];
  if (!stream || stream.encoding !== 'meshopt' || !Array.isArray(stream.segments)) {
    throw new Error(`shell_meta.json is missing meshopt stream ${streamName}.`);
  }
  await MeshoptDecoder.ready;
  const source = new Uint8Array(await fetchBytes(urlFor(baseUrl, stream.file)));
  for (let index = 0; index < stream.segments.length; index += 1) {
    const descriptor = validateDescriptor(stream.segments[index], stream, `${streamName}[${index}]`);
    if (!Number.isInteger(descriptor.offset) || descriptor.offset < 0
        || descriptor.offset + descriptor.byteLength > source.byteLength) {
      throw new Error(`${streamName}[${index}] exceeds its source file.`);
    }
    const target = new Uint8Array(descriptor.decodedByteLength);
    MeshoptDecoder.decodeGltfBuffer(
      target,
      descriptor.count,
      descriptor.stride,
      source.subarray(descriptor.offset, descriptor.offset + descriptor.byteLength),
      descriptor.mode || stream.mode,
    );
    const destinationOffset = Number(segmentOffset(index));
    if (destinationOffset < 0 || destinationOffset + target.byteLength > targetBytes.byteLength) {
      throw new Error(`${streamName}[${index}] decoded data exceeds its target buffer.`);
    }
    targetBytes.set(target, destinationOffset);
  }
  return source.byteLength;
}

function expectArrayBufferLength(buffer, expected, name) {
  if (buffer.byteLength !== expected) throw new Error(`${name} byte length ${buffer.byteLength} does not match ${expected}.`);
}

function adapterInfo(adapter) {
  const info = adapter?.info || {};
  const result = {
    vendor: String(info.vendor || ''),
    architecture: String(info.architecture || ''),
    device: String(info.device || ''),
    description: String(info.description || ''),
  };
  const text = Object.values(result).join(' ').trim();
  return {
    ...result,
    hardware: Boolean(text) && !SOFTWARE_ADAPTER_PATTERN.test(text),
  };
}

function gpuErrorMessage(error) {
  return String(error?.message || error || 'unknown WebGPU error');
}

export class GeometryShellHZB {
  constructor(assetBaseUrl, options = {}) {
    this.assetBaseUrl = String(assetBaseUrl || '');
    if (!this.assetBaseUrl) throw new Error('GeometryShellHZB requires an asset base URL.');
    this.options = { height: 288, width: 0, ...options };
    this.meta = null;
    this.adapter = null;
    this.device = null;
    this.ready = false;
    this.targets = null;
    this.candidateCapacity = 0;
    this.gpuBuffers = [];
    this.gpuValidationErrors = [];
    this.deviceLostInfo = null;
  }

  async init() {
    if (this.ready) return this;
    if (!globalThis.navigator?.gpu) throw new Error('WebGPU is unavailable.');
    this.meta = await fetchJson(urlFor(this.assetBaseUrl, 'shell_meta.json'));
    this._validateMeta();
    for (let attempt = 0; attempt < 10 && !this.adapter; attempt += 1) {
      this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!this.adapter) await new Promise((resolve) => setTimeout(resolve, 200));
    }
    if (!this.adapter) throw new Error('WebGPU requestAdapter returned null.');
    this.webgpuInfo = { api: 'webgpu', adapter: adapterInfo(this.adapter) };
    const requiredMaxBufferSize = Math.max(
      this.decodedCounts.vertices * 12,
      this.decodedCounts.indices * 4,
      this.decodedCounts.instances * 64,
      Number(this.meta.instanceCount) * 6 * 4,
    );
    const supportedMaxBufferSize = Number(this.adapter.limits.maxBufferSize);
    if (!Number.isSafeInteger(requiredMaxBufferSize)
        || requiredMaxBufferSize <= 0
        || requiredMaxBufferSize > supportedMaxBufferSize) {
      throw new Error(
        `geometry shell requires maxBufferSize=${requiredMaxBufferSize}, `
        + `adapter supports ${supportedMaxBufferSize}`,
      );
    }
    this.device = await this.adapter.requestDevice({
      requiredLimits: { maxBufferSize: requiredMaxBufferSize },
    });
    this.webgpuInfo.requiredLimits = { maxBufferSize: requiredMaxBufferSize };
    this.gpuValidationErrors = [];
    this.device.addEventListener?.('uncapturederror', (event) => {
      this.gpuValidationErrors.push(gpuErrorMessage(event.error || 'uncaptured WebGPU error'));
    });
    this.device.lost?.then((info) => {
      this.deviceLostInfo = info || { reason: 'unknown', message: 'WebGPU device lost' };
    }).catch((error) => {
      this.deviceLostInfo = { reason: 'unknown', message: gpuErrorMessage(error) };
    });
    await MeshoptDecoder.ready;
    await this._loadAssets();
    this._throwGpuHealth('asset upload');
    await this._createPipelines();
    this._throwGpuHealth('pipeline creation');
    this.ready = true;
    return this;
  }

  _throwGpuHealth(stage) {
    const errors = [];
    if (this.deviceLostInfo) {
      errors.push(`device lost (${this.deviceLostInfo.reason || 'unknown'}): ${this.deviceLostInfo.message || ''}`.trim());
    }
    if (this.gpuValidationErrors.length > 0) {
      errors.push(...this.gpuValidationErrors.map((error) => `validation: ${error}`));
    }
    if (errors.length > 0) throw new Error(`${stage}: ${errors.join('; ')}`);
  }

  async _submitAndWait(encoder, stage) {
    let scopeOpen = false;
    let scopedError = null;
    try {
      if (this.device.pushErrorScope && this.device.popErrorScope) {
        this.device.pushErrorScope('validation');
        scopeOpen = true;
      }
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      if (scopeOpen) {
        scopedError = await this.device.popErrorScope();
        scopeOpen = false;
      }
    } catch (error) {
      if (scopeOpen) {
        try {
          scopedError = await this.device.popErrorScope();
        } catch (popError) {
          scopedError = popError;
        }
      }
      if (scopedError) this.gpuValidationErrors.push(gpuErrorMessage(scopedError));
      this._throwGpuHealth(stage);
      throw error;
    }
    if (scopedError) this.gpuValidationErrors.push(gpuErrorMessage(scopedError));
    await new Promise((resolve) => queueMicrotask(resolve));
    this._throwGpuHealth(stage);
  }

  _validateMeta() {
    if (this.meta?.schema !== GEOMETRY_SHELL_HZB_SCHEMA) {
      throw new Error(`unsupported geometry shell schema: ${this.meta?.schema}`);
    }
    if (Number(this.meta.queryContract?.sampleCount) !== GEOMETRY_SHELL_HZB_SAMPLE_COUNT
        || !String(this.meta.queryContract?.mipDimensions || '').includes('explicit')) {
      throw new Error('shell_meta.json does not declare the single-sample explicit-mip HZB contract.');
    }
    const occluderSet = this.meta.queryContract?.occluderSet;
    if (Number(occluderSet?.selectedValue) !== 1
        || Number(occluderSet?.nonSelectedValue) !== 0
        || !String(occluderSet?.selectedBehavior || '').includes('without uncertainty')
        || !String(occluderSet?.nonSelectedBehavior || '').includes('AABB/HZB')) {
      throw new Error('shell_meta.json does not declare the selected-self/non-selected-HZB occluder contract.');
    }
    for (const key of ['instanceCount', 'globalGlbCount', 'prototypeCount']) {
      if (!Number.isInteger(Number(this.meta[key])) || Number(this.meta[key]) < 0) {
        throw new Error(`shell_meta.json has an invalid ${key}.`);
      }
    }
    if (!this.meta.files?.instanceAabbs?.file
        || !this.meta.files?.instanceToGlb?.file
        || !this.meta.files?.instanceOccluder?.file) {
      throw new Error('shell_meta.json is missing candidate runtime files.');
    }
    const prototypes = this.meta.prototypes || [];
    if (prototypes.length !== Number(this.meta.prototypeCount)) throw new Error('prototype count mismatch.');
    let vertexEnd = 0;
    let indexEnd = 0;
    let instanceEnd = 0;
    for (const prototype of prototypes) {
      if (!Number.isInteger(Number(prototype.vertexOffset)) || Number(prototype.vertexOffset) !== vertexEnd
          || !Number.isInteger(Number(prototype.indexOffset)) || Number(prototype.indexOffset) !== indexEnd
          || !Number.isInteger(Number(prototype.instanceOffset)) || Number(prototype.instanceOffset) !== instanceEnd) {
        throw new Error('shell prototypes are not densely packed.');
      }
      if (Number(prototype.indexCount) % 3 !== 0 || Number(prototype.instanceCount) < 1) {
        throw new Error('shell prototype geometry counts are invalid.');
      }
      vertexEnd += Number(prototype.vertexCount);
      indexEnd += Number(prototype.indexCount);
      instanceEnd += Number(prototype.instanceCount);
    }
    if (this.meta.streams?.positions?.segments?.length !== prototypes.length
        || this.meta.streams?.indices?.segments?.length !== prototypes.length
        || this.meta.streams?.transforms?.segments?.length !== prototypes.length) {
      throw new Error('shell stream segment counts do not match prototypes.');
    }
    this.decodedCounts = { vertices: vertexEnd, indices: indexEnd, instances: instanceEnd };
  }

  async _loadAssets() {
    const instanceCount = Number(this.meta.instanceCount);
    const aabbBuffer = await fetchBytes(urlFor(this.assetBaseUrl, this.meta.files.instanceAabbs.file));
    const mappingBuffer = await fetchBytes(urlFor(this.assetBaseUrl, this.meta.files.instanceToGlb.file));
    const occluderBuffer = await fetchBytes(urlFor(this.assetBaseUrl, this.meta.files.instanceOccluder.file));
    expectArrayBufferLength(aabbBuffer, instanceCount * 6 * 4, 'instance AABBs');
    expectArrayBufferLength(mappingBuffer, instanceCount * 4, 'instance-to-GLB mapping');
    expectArrayBufferLength(occluderBuffer, instanceCount * 4, 'instance occluder mask');
    this.instanceAabbs = new Float32Array(aabbBuffer);
    this.instanceToGlb = new Uint32Array(mappingBuffer);
    this.instanceOccluder = new Uint32Array(occluderBuffer);
    this.vertexData = new Uint8Array(this.decodedCounts.vertices * 12);
    this.indexData = new Uint8Array(this.decodedCounts.indices * 4);
    this.transformData = new Uint8Array(this.decodedCounts.instances * 64);
    const positionBytes = await decodeStream(
      this.meta,
      'positions',
      this.assetBaseUrl,
      this.vertexData,
      (index) => Number(this.meta.prototypes[index].vertexOffset) * 12,
    );
    const indexBytes = await decodeStream(
      this.meta,
      'indices',
      this.assetBaseUrl,
      this.indexData,
      (index) => Number(this.meta.prototypes[index].indexOffset) * 4,
    );
    const transformBytes = await decodeStream(
      this.meta,
      'transforms',
      this.assetBaseUrl,
      this.transformData,
      (index) => Number(this.meta.prototypes[index].instanceOffset) * 64,
    );
    this.gpuBuffers.push(
      createGpuBuffer(this.device, this.vertexData, GPUBufferUsage.VERTEX, 'geometry-shell-positions'),
      createGpuBuffer(this.device, this.indexData, GPUBufferUsage.INDEX, 'geometry-shell-indices'),
      createGpuBuffer(this.device, this.transformData, GPUBufferUsage.VERTEX, 'geometry-shell-transforms'),
      createGpuBuffer(this.device, this.instanceAabbs, GPUBufferUsage.STORAGE, 'geometry-shell-aabbs'),
      createGpuBuffer(this.device, this.instanceToGlb, GPUBufferUsage.STORAGE, 'geometry-shell-instance-to-glb'),
      createGpuBuffer(this.device, this.instanceOccluder, GPUBufferUsage.STORAGE, 'geometry-shell-instance-occluder'),
    );
    [
      this.positionBuffer,
      this.indexBuffer,
      this.transformBuffer,
      this.aabbBuffer,
      this.instanceToGlbBuffer,
      this.instanceOccluderBuffer,
    ] = this.gpuBuffers;
    this.assetLoadInfo = {
      compressedPositionBytes: positionBytes,
      compressedIndexBytes: indexBytes,
      compressedTransformBytes: transformBytes,
      decodedGeometryBytes: this.vertexData.byteLength + this.indexData.byteLength + this.transformData.byteLength,
      runtimePayloadBytes: this.instanceAabbs.byteLength + this.instanceToGlb.byteLength + this.instanceOccluder.byteLength,
    };
  }

  async _createPipelines() {
    this.renderUniformBuffer = this.device.createBuffer({
      size: RENDER_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.queryUniformBuffer = this.device.createBuffer({
      size: QUERY_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.gpuBuffers.push(this.renderUniformBuffer, this.queryUniformBuffer);
    const depthShader = this.device.createShaderModule({
      code: `${GEOMETRY_SHELL_DEPTH_VERTEX_SHADER}\n${GEOMETRY_SHELL_DEPTH_FRAGMENT_SHADER}`,
    });
    const mipShader = this.device.createShaderModule({ code: GEOMETRY_SHELL_MIP_SHADER });
    const queryShader = this.device.createShaderModule({ code: GEOMETRY_SHELL_QUERY_SHADER });
    for (const [name, shader] of [['depth', depthShader], ['mip', mipShader], ['query', queryShader]]) {
      const compilation = await shader.getCompilationInfo();
      const errors = compilation.messages.filter((message) => message.type === 'error');
      if (errors.length) throw new Error(`${name} WGSL compilation failed: ${errors.map((message) => message.message).join('; ')}`);
    }
    this.renderPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: depthShader,
        entryPoint: 'depth_vertex_main',
        buffers: [
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          },
          {
            arrayStride: 64,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x4' },
              { shaderLocation: 2, offset: 16, format: 'float32x4' },
              { shaderLocation: 3, offset: 32, format: 'float32x4' },
              { shaderLocation: 4, offset: 48, format: 'float32x4' },
            ],
          },
        ],
      },
      fragment: {
        module: depthShader,
        entryPoint: 'depth_fragment_main',
        targets: [{ format: 'rgba32float' }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
        frontFace: 'ccw',
      },
      multisample: { count: GEOMETRY_SHELL_HZB_SAMPLE_COUNT },
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });
    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBuffer } },
      ],
    });
    this.mipPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: mipShader,
        entryPoint: 'main',
      },
    });
    this.queryPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: queryShader,
        entryPoint: 'main',
      },
    });
    this.queryResultBuffer = null;
    this.queryReadbackBuffer = null;
    this.visibleIdsBuffer = null;
    this.visibleIdsReadbackBuffer = null;
    this.glbFlagsBuffer = null;
    this.glbFlagsReadbackBuffer = null;
  }

  _ensureTargets(camera) {
    const height = Number(this.options.height || camera.height || 288);
    const width = Number(this.options.width || camera.width || Math.round(height * camera.aspect));
    if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1) {
      throw new Error('HZB target dimensions must be positive integers.');
    }
    if (this.targets?.width === width && this.targets?.height === height) return;
    this.targets?.destroy?.();
    const mipCount = 1 + Math.ceil(Math.log2(Math.max(width, height)));
    const linearDepthTexture = this.device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      mipLevelCount: mipCount,
      sampleCount: GEOMETRY_SHELL_HZB_SAMPLE_COUNT,
      format: 'rgba32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    const depthTexture = this.device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      sampleCount: GEOMETRY_SHELL_HZB_SAMPLE_COUNT,
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const mipUniformBuffers = [];
    const mipBindGroups = [];
    for (let mip = 1; mip < mipCount; mip += 1) {
      const sourceWidth = Math.max(1, Math.ceil(width / (2 ** (mip - 1))));
      const sourceHeight = Math.max(1, Math.ceil(height / (2 ** (mip - 1))));
      const targetWidth = Math.max(1, Math.ceil(width / (2 ** mip)));
      const targetHeight = Math.max(1, Math.ceil(height / (2 ** mip)));
      const uniform = new Uint32Array([sourceWidth, sourceHeight, targetWidth, targetHeight]);
      const uniformBuffer = createGpuBuffer(this.device, uniform, GPUBufferUsage.UNIFORM, `geometry-shell-mip-${mip}`);
      mipUniformBuffers.push(uniformBuffer);
      mipBindGroups.push(this.device.createBindGroup({
        layout: this.mipPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: linearDepthTexture.createView({ baseMipLevel: mip - 1, mipLevelCount: 1 }) },
          { binding: 1, resource: linearDepthTexture.createView({ baseMipLevel: mip, mipLevelCount: 1 }) },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      }));
    }
    const renderDepthView = linearDepthTexture.createView({ baseMipLevel: 0, mipLevelCount: 1 });
    const hzbView = linearDepthTexture.createView({ baseMipLevel: 0, mipLevelCount: mipCount });
    this.targets = {
      width,
      height,
      mipCount,
      sampleCount: GEOMETRY_SHELL_HZB_SAMPLE_COUNT,
      linearDepthTexture,
      depthTexture,
      mipUniformBuffers,
      mipBindGroups,
      renderDepthView,
      hzbView,
      destroy: () => {
        linearDepthTexture.destroy();
        depthTexture.destroy();
        for (const buffer of mipUniformBuffers) buffer.destroy();
      },
    };
    this.queryBindGroup = null;
  }

  _writeRenderUniform(camera) {
    const uniform = new Float32Array(RENDER_UNIFORM_BYTES / 4);
    uniform.set(camera.viewProjection, 0);
    uniform.set(camera.view, 16);
    uniform.set([camera.far, camera.near, 0, 0], 32);
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, uniform);
  }

  _writeQueryUniform(camera, candidateCount, depthBiasM) {
    const bytes = new ArrayBuffer(QUERY_UNIFORM_BYTES);
    const floats = new Float32Array(bytes);
    floats.set(camera.view, 0);
    floats.set([this.targets.width, this.targets.height, 0, 0], 16);
    floats.set([camera.tanX, camera.tanY, depthBiasM, camera.near], 20);
    floats.set([...camera.position, 0], 24);
    new Uint32Array(bytes, 112, 4).set([
      candidateCount,
      Number(this.meta.instanceCount),
      Number(this.meta.globalGlbCount),
      this.targets.mipCount,
    ]);
    this.device.queue.writeBuffer(this.queryUniformBuffer, 0, bytes);
  }

  _ensureQueryBuffers(candidateCount) {
    if (candidateCount <= this.candidateCapacity && this.queryResultBuffer) return;
    for (const buffer of [
      this.queryResultBuffer,
      this.queryReadbackBuffer,
      this.visibleIdsBuffer,
      this.visibleIdsReadbackBuffer,
    ]) buffer?.destroy();
    this.candidateCapacity = Math.max(1, candidateCount);
    const resultBytes = 16;
    const visibleBytes = Math.max(4, align4(this.candidateCapacity * 4));
    this.queryResultBuffer = this.device.createBuffer({
      size: resultBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.queryReadbackBuffer = this.device.createBuffer({
      size: resultBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.visibleIdsBuffer = this.device.createBuffer({
      size: visibleBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.visibleIdsReadbackBuffer = this.device.createBuffer({
      size: visibleBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    if (!this.glbFlagsBuffer) {
      this.glbFlagsBuffer = this.device.createBuffer({
        size: Math.max(4, align4(Number(this.meta.globalGlbCount) * 4)),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.glbFlagsReadbackBuffer = this.device.createBuffer({
        size: Math.max(4, align4(Number(this.meta.globalGlbCount) * 4)),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    this.gpuBuffers.push(
      this.queryResultBuffer,
      this.queryReadbackBuffer,
      this.visibleIdsBuffer,
      this.visibleIdsReadbackBuffer,
      this.glbFlagsBuffer,
      this.glbFlagsReadbackBuffer,
    );
    this.queryBindGroup = null;
  }

  _ensureQueryBindGroup() {
    if (this.queryBindGroup) return;
    this.queryBindGroup = this.device.createBindGroup({
      layout: this.queryPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.queryUniformBuffer } },
        { binding: 1, resource: { buffer: this.aabbBuffer } },
        { binding: 2, resource: { buffer: this.candidateBuffer } },
        { binding: 3, resource: { buffer: this.queryResultBuffer } },
        { binding: 4, resource: { buffer: this.visibleIdsBuffer } },
        { binding: 5, resource: this.targets.hzbView },
        { binding: 6, resource: { buffer: this.instanceToGlbBuffer } },
        { binding: 7, resource: { buffer: this.glbFlagsBuffer } },
        { binding: 8, resource: { buffer: this.instanceOccluderBuffer } },
      ],
    });
  }

  async _renderAndBuildHzb(camera) {
    this._ensureTargets(camera);
    this._writeRenderUniform(camera);
    const depthStartedAt = nowMs();
    const depthEncoder = this.device.createCommandEncoder();
    const pass = depthEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.targets.renderDepthView,
        clearValue: { r: camera.far, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.targets.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBindGroup);
    pass.setVertexBuffer(0, this.positionBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    for (const prototype of this.meta.prototypes) {
      pass.setVertexBuffer(1, this.transformBuffer, Number(prototype.instanceOffset) * 64);
      pass.drawIndexed(
        Number(prototype.indexCount),
        Number(prototype.instanceCount),
        Number(prototype.indexOffset),
        Number(prototype.vertexOffset),
        0,
      );
    }
    pass.end();
    await this._submitAndWait(depthEncoder, 'depth raster');
    const depthRasterMs = nowMs() - depthStartedAt;

    const hzbStartedAt = nowMs();
    const hzbEncoder = this.device.createCommandEncoder();
    const mipPass = hzbEncoder.beginComputePass();
    mipPass.setPipeline(this.mipPipeline);
    for (let index = 0; index < this.targets.mipBindGroups.length; index += 1) {
      const mip = index + 1;
      const width = Math.max(1, Math.ceil(this.targets.width / (2 ** mip)));
      const height = Math.max(1, Math.ceil(this.targets.height / (2 ** mip)));
      mipPass.setBindGroup(0, this.targets.mipBindGroups[index]);
      mipPass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    }
    mipPass.end();
    await this._submitAndWait(hzbEncoder, 'HZB build');
    const hzbBuildMs = nowMs() - hzbStartedAt;
    return {
      depthRasterMs,
      hzbBuildMs,
      depthRasterAndMipMs: depthRasterMs + hzbBuildMs,
    };
  }

  async _queryCurrent(camera, candidateIds, options = {}) {
    const ids = Uint32Array.from(candidateIds || []);
    const depthBiasM = Number(options.depthBiasM ?? 0.001);
    if (!Number.isFinite(depthBiasM) || depthBiasM < 0) throw new Error('depthBiasM must be non-negative and finite.');
    this._ensureQueryBuffers(ids.length);
    if (!this.candidateBuffer || this.candidateBuffer.size < Math.max(4, ids.byteLength)) {
      this.candidateBuffer?.destroy();
      this.candidateBuffer = this.device.createBuffer({
        size: Math.max(4, align4(ids.byteLength)),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.gpuBuffers.push(this.candidateBuffer);
      this.queryBindGroup = null;
    }
    if (ids.length > 0) this.device.queue.writeBuffer(this.candidateBuffer, 0, ids);
    this._ensureQueryBindGroup();
    this._writeQueryUniform(camera, ids.length, depthBiasM);
    const aabbStartedAt = nowMs();
    const queryEncoder = this.device.createCommandEncoder();
    queryEncoder.clearBuffer(this.queryResultBuffer);
    queryEncoder.clearBuffer(this.visibleIdsBuffer);
    queryEncoder.clearBuffer(this.glbFlagsBuffer);
    const pass = queryEncoder.beginComputePass();
    pass.setPipeline(this.queryPipeline);
    pass.setBindGroup(0, this.queryBindGroup);
    if (ids.length > 0) pass.dispatchWorkgroups(Math.ceil(ids.length / 64));
    pass.end();
    await this._submitAndWait(queryEncoder, 'AABB/HZB candidate test');
    const aabbTestMs = nowMs() - aabbStartedAt;

    const compactionStartedAt = nowMs();
    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(this.queryResultBuffer, 0, this.queryReadbackBuffer, 0, 16);
    if (ids.length > 0) {
      copyEncoder.copyBufferToBuffer(
        this.visibleIdsBuffer,
        0,
        this.visibleIdsReadbackBuffer,
        0,
        align4(ids.length * 4),
      );
    }
    copyEncoder.copyBufferToBuffer(
      this.glbFlagsBuffer,
      0,
      this.glbFlagsReadbackBuffer,
      0,
      Math.max(4, align4(Number(this.meta.globalGlbCount) * 4)),
    );
    await this._submitAndWait(copyEncoder, 'HZB result compaction');
    const readbackStartedAt = nowMs();
    await this.queryReadbackBuffer.mapAsync(GPUMapMode.READ, 0, 16);
    const resultBytes = this.queryReadbackBuffer.getMappedRange(0, 16).slice(0);
    this.queryReadbackBuffer.unmap();
    const visibleReadbackBytes = Math.max(4, align4(ids.length * 4));
    await this.visibleIdsReadbackBuffer.mapAsync(GPUMapMode.READ, 0, visibleReadbackBytes);
    const visibleBytes = this.visibleIdsReadbackBuffer.getMappedRange(0, visibleReadbackBytes).slice(0);
    this.visibleIdsReadbackBuffer.unmap();
    await this.glbFlagsReadbackBuffer.mapAsync(GPUMapMode.READ, 0, Math.max(4, align4(Number(this.meta.globalGlbCount) * 4)));
    const flagsBytes = this.glbFlagsReadbackBuffer.getMappedRange(0, Math.max(4, align4(Number(this.meta.globalGlbCount) * 4))).slice(0);
    this.glbFlagsReadbackBuffer.unmap();
    const readbackMs = nowMs() - readbackStartedAt;
    const words = new Uint32Array(resultBytes);
    const counter = Number(words[0] || 0);
    const uncertainCount = Number(words[1] || 0);
    const overflowCount = Number(words[2] || 0);
    const visibleCount = Math.min(counter, ids.length);
    const decodedVisibleIds = new Uint32Array(visibleBytes);
    let visibleIds = Uint32Array.from(decodedVisibleIds.subarray(0, visibleCount));
    if (overflowCount > 0) visibleIds = Uint32Array.from(ids);
    visibleIds.sort();
    const flags = new Uint32Array(flagsBytes);
    const modelList = [];
    for (let index = 0; index < Number(this.meta.globalGlbCount); index += 1) {
      if (flags[index] & 1) modelList.push(index);
    }
    const glbQueue = modelList.map((globalGlbId) => ({
      globalGlbId,
      modelVisible: true,
      renderVisible: true,
      confidence: 1,
      importance: 1,
      downloadPriority: 1,
      prioritySource: 'geometry-shell-hzb-visible-instance-aggregation',
    }));
    const compactionMs = nowMs() - compactionStartedAt;
    return {
      mode: options.mode || 'Point60',
      idMode: 'component',
      visibleIds,
      componentModelList: visibleIds,
      modelList,
      glbQueue,
      candidateCount: ids.length,
      visibleCount: visibleIds.length,
      uncertainCount,
      overflowCount,
      timings: {
        aabbTestMs,
        compactionMs,
        queryMs: aabbTestMs,
        readbackMs,
        totalMs: aabbTestMs + compactionMs,
        depthEncoding: LINEAR_DEPTH_ENCODING,
        sampleCount: GEOMETRY_SHELL_HZB_SAMPLE_COUNT,
        depthBiasM,
        hzbMipCount: this.targets.mipCount,
        readbackBytes: resultBytes.byteLength + visibleBytes.byteLength + flagsBytes.byteLength,
      },
      gpuValidationErrors: [...this.gpuValidationErrors],
    };
  }

  async queryPoint60(cameraInput, candidateIds, options = {}) {
    if (!this.ready) await this.init();
    const camera = cameraState(cameraInput);
    if (Math.abs(camera.fovYDeg - 60) > 1e-3) throw new Error('Point60 requires a 60-degree camera.');
    const renderTiming = await this._renderAndBuildHzb(camera);
    const result = await this._queryCurrent(camera, candidateIds, { ...options, mode: 'Point60' });
    result.timings.depthRasterMs = renderTiming.depthRasterMs;
    result.timings.hzbBuildMs = renderTiming.hzbBuildMs;
    result.timings.depthRasterAndMipMs = renderTiming.depthRasterAndMipMs;
    result.timings.totalMs = renderTiming.depthRasterAndMipMs + result.timings.aabbTestMs + result.timings.compactionMs;
    result.backend = 'geometry-shell-hzb-webgpu';
    return result;
  }

  async queryRegion66(cameras, candidateIds, options = {}) {
    if (!this.ready) await this.init();
    if (!Array.isArray(cameras) || cameras.length === 0) throw new Error('Region66 requires at least one camera.');
    const ids = Uint32Array.from(candidateIds || []);
    const union = new Set();
    const perPose = [];
    let depthRasterMs = 0;
    let hzbBuildMs = 0;
    let queryMs = 0;
    let readbackMs = 0;
    let aabbTestMs = 0;
    let compactionMs = 0;
    let regionUnionCompactionMs = 0;
    let uncertainCount = 0;
    for (const camera of cameras) {
      const state = cameraState(camera);
      if (Math.abs(state.fovYDeg - 66) > 1e-3) throw new Error('Region66 requires 66-degree cameras.');
      const renderTiming = await this._renderAndBuildHzb(state);
      const result = await this._queryCurrent(state, ids, { ...options, mode: 'Region66' });
      result.timings.depthRasterMs = renderTiming.depthRasterMs;
      result.timings.hzbBuildMs = renderTiming.hzbBuildMs;
      result.timings.depthRasterAndMipMs = renderTiming.depthRasterAndMipMs;
      result.timings.totalMs = renderTiming.depthRasterAndMipMs + result.timings.aabbTestMs + result.timings.compactionMs;
      queryMs += Number(result.timings.queryMs || 0);
      readbackMs += Number(result.timings.readbackMs || 0);
      depthRasterMs += Number(result.timings.depthRasterMs || 0);
      hzbBuildMs += Number(result.timings.hzbBuildMs || 0);
      aabbTestMs += Number(result.timings.aabbTestMs || 0);
      compactionMs += Number(result.timings.compactionMs || 0);
      uncertainCount += Number(result.uncertainCount || 0);
      perPose.push(result);
      const unionStartedAt = nowMs();
      for (const id of result.visibleIds) union.add(Number(id));
      regionUnionCompactionMs += nowMs() - unionStartedAt;
    }
    const finalCompactionStartedAt = nowMs();
    const visibleIds = Uint32Array.from([...union].sort((a, b) => a - b));
    const glbSet = new Set();
    for (const id of visibleIds) glbSet.add(Number(this.instanceToGlb[id]));
    const modelList = [...glbSet].sort((a, b) => a - b);
    regionUnionCompactionMs += nowMs() - finalCompactionStartedAt;
    compactionMs += regionUnionCompactionMs;
    return {
      mode: 'Region66',
      idMode: 'component',
      visibleIds,
      componentModelList: visibleIds,
      modelList,
      glbQueue: modelList.map((globalGlbId) => ({
        globalGlbId,
        modelVisible: true,
        renderVisible: true,
        confidence: 1,
        importance: 1,
        downloadPriority: 1,
        prioritySource: 'geometry-shell-hzb-region-union',
      })),
      candidateCount: ids.length,
      visibleCount: visibleIds.length,
      uncertainCount,
      passCount: cameras.length,
      perPose,
      backend: 'geometry-shell-hzb-webgpu',
      timings: {
        depthRasterMs,
        hzbBuildMs,
        depthRasterAndMipMs: depthRasterMs + hzbBuildMs,
        aabbTestMs,
        compactionMs,
        regionUnionCompactionMs,
        queryMs,
        readbackMs,
        totalMs: depthRasterMs + hzbBuildMs + aabbTestMs + compactionMs,
        depthEncoding: LINEAR_DEPTH_ENCODING,
        sampleCount: GEOMETRY_SHELL_HZB_SAMPLE_COUNT,
        depthBiasM: Number(options.depthBiasM ?? 0.001),
        hzbPassCount: cameras.length,
      },
    };
  }

  dispose() {
    this.targets?.destroy?.();
    for (const buffer of this.gpuBuffers) buffer?.destroy?.();
    this.gpuBuffers = [];
    this.ready = false;
    this.targets = null;
  }
}

export { adapterInfo, cameraState, SOFTWARE_ADAPTER_PATTERN };
