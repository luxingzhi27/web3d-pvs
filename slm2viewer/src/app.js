import { Viewer } from './viewer.js';
import { RendererRuntime } from './RendererRuntime.js';
import queryString from 'query-string';
import { startupLog } from './startupTimeline.js';

startupLog('app:module-evaluated');

if (!(window.File && window.FileReader && window.FileList && window.Blob)) 
{
  console.error('The File APIs are not fully supported in this browser.');
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
      cameraPosition: hash.cameraPosition ? hash.cameraPosition.split(',').map(Number) : null,
      renderBackend: new URLSearchParams(location.search).get('renderBackend') || 'auto',
    };

    this.el = el;
    this.viewer = null;
    this.error = null;
    this.viewerEl = null;
    this.root = el.querySelector('.wrap');

    this.ready = this.view().catch((error) => this.onError(error));
  }

  async createViewer()
  {
    this.viewerEl = document.createElement('div');
    this.viewerEl.classList.add('viewer');
    this.root.appendChild(this.viewerEl);
    const rendererRuntime = await RendererRuntime.create({
      backendPreference: this.options.renderBackend,
      allowSoftwareAdapter: globalThis.__SLM_ALLOW_SOFTWARE_WEBGPU__ === true,
    });
    this.viewer = new Viewer(this.viewerEl, this.options, rendererRuntime);
    return this.viewer;
  }

  async view()
  {
    if (this.viewer) this.viewer.clear();

    const viewer = this.viewer || await this.createViewer();

    viewer.load();
    return viewer;
  }

  onError (error) 
  {
    this.error = String(error?.stack || error);
    console.error(error);
  }

  showSpinner () {
    this.spinnerEl.style.display = '';
  }

  hideSpinner () {
    this.spinnerEl.style.display = 'none';
  }
}

document.addEventListener('DOMContentLoaded', () => 
{
  startupLog('app:DOMContentLoaded');
  const app = new App(document.body, location);
  window.__slmApp = app;
  startupLog('app:created');
});
