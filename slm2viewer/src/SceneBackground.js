import { Color } from 'three';
import { mix, smoothstep, uniform, viewportUV } from 'three/tsl';

export class SceneBackground {
  constructor(top = '#ffffff', bottom = '#353588') {
    this.top = uniform(new Color(top));
    this.bottom = uniform(new Color(bottom));
    this.node = mix(this.bottom, this.top, smoothstep(0.05, 0.95, viewportUV.y));
  }

  setColors(top, bottom) {
    this.top.value.set(top);
    this.bottom.value.set(bottom);
  }
}
