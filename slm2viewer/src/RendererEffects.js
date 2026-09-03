import { RenderPipeline } from 'three/webgpu';
import { float, mix, pass, uniform } from 'three/tsl';
import { ao } from 'three/examples/jsm/tsl/display/GTAONode.js';
import { smaa } from 'three/examples/jsm/tsl/display/SMAANode.js';

const DEFAULTS = Object.freeze({
  enabled: true,
  aoEnabled: true,
  aoResolutionScale: 0.5,
  aoRadius: 5,
  aoThickness: 1,
  aoDistanceFallOff: 0.2,
  aoSamples: 8,
  aoIntensity: 1,
  smaaEnabled: true,
});

export class RendererEffects {
  constructor(renderer, scene, camera, options = {}) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.options = { ...DEFAULTS, ...options };
    this.pipeline = null;
    this.scenePass = null;
    this.aoNode = null;
    this.smaaNode = null;
    this.aoIntensity = uniform(this.options.aoIntensity);
    this._rebuild();
  }

  _disposeNodes() {
    this.pipeline?.dispose();
    this.scenePass?.dispose();
    this.aoNode?.dispose();
    this.smaaNode?.dispose();
    this.pipeline = null;
    this.scenePass = null;
    this.aoNode = null;
    this.smaaNode = null;
  }

  _rebuild() {
    this._disposeNodes();
    if (!this.options.enabled) return;

    this.scenePass = pass(this.scene, this.camera);
    let outputNode = this.scenePass.getTextureNode();
    if (this.options.aoEnabled) {
      this.aoNode = ao(this.scenePass.getTextureNode('depth'), null, this.camera);
      this._updateAo();
      const aoFactor = mix(float(1), this.aoNode.getTextureNode().r, this.aoIntensity);
      outputNode = outputNode.mul(aoFactor);
    }
    if (this.options.smaaEnabled) {
      this.smaaNode = smaa(outputNode);
      outputNode = this.smaaNode;
    }

    this.pipeline = new RenderPipeline(this.renderer);
    this.pipeline.outputNode = outputNode;
    this.pipeline.needsUpdate = true;
  }

  _updateAo() {
    if (!this.aoNode) return;
    this.aoNode.resolutionScale = this.options.aoResolutionScale;
    this.aoNode.radius.value = this.options.aoRadius;
    this.aoNode.thickness.value = this.options.aoThickness;
    this.aoNode.distanceFallOff.value = this.options.aoDistanceFallOff;
    this.aoNode.samples.value = this.options.aoSamples;
    this.aoIntensity.value = this.options.aoIntensity;
  }

  configure(options = {}) {
    const rebuild = ['enabled', 'aoEnabled', 'smaaEnabled']
      .some((key) => options[key] != null && Boolean(options[key]) !== Boolean(this.options[key]));
    this.options = { ...this.options, ...options };
    if (rebuild) this._rebuild();
    else this._updateAo();
  }

  render() {
    if (this.pipeline) this.pipeline.render();
    else this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this._disposeNodes();
  }
}
