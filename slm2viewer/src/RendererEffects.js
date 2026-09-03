import { RenderPipeline } from 'three/webgpu';
import { pass, pow, uniform } from 'three/tsl';
import { ao } from 'three/examples/jsm/tsl/display/GTAONode.js';

const DEFAULTS = Object.freeze({
  enabled: true,
  aoEnabled: true,
  aoResolutionScale: 0.5,
  aoRadius: 5,
  aoThickness: 1,
  aoDistanceFallOff: 0.2,
  aoSamples: 8,
  aoStrength: 2,
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
    this.aoStrength = uniform(this.options.aoStrength);
    this._rebuild();
  }

  _disposeNodes() {
    this.pipeline?.dispose();
    this.scenePass?.dispose();
    this.aoNode?.dispose();
    this.pipeline = null;
    this.scenePass = null;
    this.aoNode = null;
  }

  _rebuild() {
    this._disposeNodes();
    if (!this.options.enabled) return;

    this.scenePass = pass(this.scene, this.camera);
    let outputNode = this.scenePass.getTextureNode();
    if (this.options.aoEnabled) {
      this.aoNode = ao(this.scenePass.getTextureNode('depth'), null, this.camera);
      this._updateAo();
      const aoFactor = pow(this.aoNode.getTextureNode().r.clamp(0, 1), this.aoStrength);
      outputNode = outputNode.mul(aoFactor);
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
    this.aoStrength.value = this.options.aoStrength;
  }

  configure(options = {}) {
    const rebuild = ['enabled', 'aoEnabled']
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
