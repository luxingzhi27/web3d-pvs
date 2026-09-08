import { resolveRenderSurface, sameRenderSurface } from './RenderSurfacePolicy.js';
import { ViewerDiagnostics } from './ViewerDiagnostics.js';

export class ViewerRenderLoop extends ViewerDiagnostics {
  keydown(keyEvent) 
  {
    //console.log(keyEvent);
    // if (keyEvent.key == 'e')
    // {
      
    // }
    // else if (keyEvent.key == ' ')
    // {
    //   // this.clearLoadedForDebug();

    //   // this.sceneCulling();
    // }
    // else if (keyEvent.key == 't')
    // {
    //   this.loadingTimescale = (this.loadingTimescale == 1 ? 200 : 1);
    // }
    // else 
    if (keyEvent.key == '1')
    {
      console.log(
        '\"cameraPostion\": [' + this.activeCamera.position.x.toFixed(2) + ',' + this.activeCamera.position.y.toFixed(2) + ',' + this.activeCamera.position.z.toFixed(2) + '],\n' + 
        '\"cameraTarget\": [' + this.controls.target.x.toFixed(2) + ',' + this.controls.target.y.toFixed(2) + ',' + this.controls.target.z.toFixed(2) + '],');
    }
  }

  requestRender(reason = 'viewer')
  {
    this.renderRequested = true;
    if (this.maintenanceTimer != null)
    {
      clearTimeout(this.maintenanceTimer);
      this.maintenanceTimer = null;
    }
    if (this.animationFrameId == null)
    {
      this.animationFrameId = requestAnimationFrame(this.animate);
    }
  }

  _scheduleMaintenance(delayMs)
  {
    if (!Number.isFinite(Number(delayMs)) || this.animationFrameId != null)
    {
      return;
    }
    const delay = Math.max(16, Math.min(1000, Number(delayMs)));
    if (this.maintenanceTimer != null)
    {
      clearTimeout(this.maintenanceTimer);
    }
    this.maintenanceTimer = setTimeout(() =>
    {
      this.maintenanceTimer = null;
      this.requestRender('maintenance');
    }, delay);
  }

  _hasContinuousFrameActivity()
  {
    return Boolean(
      this.orbitInteractionActive
      || this.controls.autoRotate
      || this.controls.enableDamping
      || this.keyboardMgr?.hasActiveInput()
      || this.touchMoveController?.hasActiveInput()
      || this.trajectoryCollector?.isRecording
      || (this.gui && !this.gui.closed)
    );
  }

  animate(time)
  {
    this.animationFrameId = null;
    this.renderRequested = false;

    const dt = this.prevTime > 0 ? Math.min(100, time - this.prevTime) : 16.67;
    this.prevTime = time;

    let cameraChanged = false;
    if (this.controls.autoRotate || this.controls.enableDamping)
    {
      cameraChanged = Boolean(this.controls.update()) || cameraChanged;
    }
    cameraChanged = Boolean(this.keyboardMgr.update(dt)) || cameraChanged;
    if (this.touchMoveController)
    {
      cameraChanged = Boolean(this.touchMoveController.update(dt)) || cameraChanged;
    }
    if (cameraChanged && this.slm2Loader)
    {
      this.slm2Loader.notifyCameraChanged();
    }

    if (this.trajectoryCollector)
    {
      this.trajectoryCollector.update(dt / 1000.0);
    }

    // Reconcile resource and visibility deltas before drawing the frame.
    this.slm2Loader.update(dt, time);
    if (this.predictionDebugEnabled)
    {
      this.updatePredictionDebugOverlay();
    }
    this.render();

    const debugPanelExpanded = Boolean(this.gui && !this.gui.closed);
    if (debugPanelExpanded)
    {
      this.stats.update();
      this.updateRuntimeDebugGui(time);
    }
    if (this.queueDebugEnabled)
    {
      this.updateQueueDebugPanel(time);
    }

    const continueFrames = this._hasContinuousFrameActivity()
      || this.slm2Loader.hasImmediateFrameWork();
    if (continueFrames || this.renderRequested)
    {
      this.requestRender('frame-continuation');
      return;
    }

    let maintenanceDelay = this.slm2Loader.getMaintenanceDelayMs();
    if (this.queueDebugEnabled)
    {
      maintenanceDelay = maintenanceDelay == null
        ? this.queueDebugIntervalMs
        : Math.min(maintenanceDelay, this.queueDebugIntervalMs);
    }
    this._scheduleMaintenance(maintenanceDelay);
  }

  render()
  {
    if (this.usePostEffect)
    {
      this.composer.render();
    }
    else
    {
      this.renderer.clear();
      this.renderer.render( this.scene, this.activeCamera );
    }
  }

  _resizeRenderTargets()
  {
    const parent = this.el.parentElement || this.el;
    const clientWidth = Math.max(1, Number(parent.clientWidth || this.el.clientWidth || 1));
    const clientHeight = Math.max(1, Number(parent.clientHeight || this.el.clientHeight || 1));
    const nextSurface = resolveRenderSurface({
      width: clientWidth,
      height: clientHeight,
      devicePixelRatio: window.devicePixelRatio,
    });
    if (sameRenderSurface(this.renderSurface, nextSurface))
    {
      this.renderSurface = nextSurface;
      return false;
    }
    this.renderSurface = nextSurface;
    this.renderer.setDrawingBufferSize(
      nextSurface.cssWidth,
      nextSurface.cssHeight,
      nextSurface.pixelRatio
    );
    this.renderer.domElement.style.width = nextSurface.cssWidth + 'px';
    this.renderer.domElement.style.height = nextSurface.cssHeight + 'px';
    if (this.composer)
    {
      this.composer.setSize(nextSurface.cssWidth, nextSurface.cssHeight);
    }
    return true;
  }

  scheduleResize()
  {
    if (this.resizeTimer != null)
    {
      clearTimeout(this.resizeTimer);
    }
    this.resizeTimer = setTimeout(() =>
    {
      this.resizeTimer = null;
      this.resize();
    }, this.resizeDebounceMs);
  }

  resize()
  {
    const parent = this.el.parentElement || this.el;
    const clientWidth = Math.max(1, Number(parent.clientWidth || this.el.clientWidth || 1));
    const clientHeight = Math.max(1, Number(parent.clientHeight || this.el.clientHeight || 1));

    const nextAspect = clientWidth / clientHeight;
    if (this.activeCamera.aspect !== nextAspect)
    {
      this.activeCamera.aspect = nextAspect;
      this.activeCamera.updateProjectionMatrix();
    }

    this._resizeRenderTargets();

    if (this.slm2Loader.clientWidth !== clientWidth || this.slm2Loader.clientHeight !== clientHeight)
    {
      this.slm2Loader.setSize(clientWidth, clientHeight);
    }

    if (this.DebugMode) console.log(clientWidth, clientHeight);
    this.requestRender('resize');
  }
}

