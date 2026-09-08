import WebGL from 'three/examples/jsm/capabilities/WebGL.js';
import { Viewer } from './viewer.js';
import queryString from 'query-string';
import { startupLog } from './startupTimeline.js';

startupLog('app:module-evaluated');

if (!(window.File && window.FileReader && window.FileList && window.Blob)) 
{
  console.error('The File APIs are not fully supported in this browser.');
}
else if (!WebGL.isWebGL2Available()) 
{
  console.error('WebGL is not supported in this browser.');
}

class App 
{
  constructor (el, location) 
  {
    const hash = location.hash ? queryString.parse(location.hash) : {};
    this.options = {
      kiosk: Boolean(hash.kiosk),
      model: hash.model || '',
      preset: hash.preset || '',
      cameraPosition: hash.cameraPosition ? hash.cameraPosition.split(',').map(Number) : null
    };

    this.viewer = null;
    this.viewerEl = null;
    this.root = el.querySelector('.wrap');

    this.view();
  }

  createViewer() 
  {
    this.viewerEl = document.createElement('div');
    this.viewerEl.classList.add('viewer');
    this.root.appendChild(this.viewerEl);
    this.viewer = new Viewer(this.viewerEl, this.options);
    return this.viewer;
  }

  view() 
  {
    if (this.viewer) this.viewer.clear();

    const viewer = this.viewer || this.createViewer();

    viewer.load();
  }

}

document.addEventListener('DOMContentLoaded', () => 
{
  startupLog('app:DOMContentLoaded');
  const app = new App(document.body, location);
  window.__slmApp = app;
  startupLog('app:created');
});
