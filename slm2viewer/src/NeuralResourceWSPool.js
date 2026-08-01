export class NeuralResourceWSPool {
  constructor(options = {}) {
    this.url = options.url || null;
    this.connectionCount = Math.max(1, Number(options.connectionCount || 1));
    this.batchTimeoutMs = Math.max(1000, Number(options.batchTimeoutMs || 20000));
    this.debug = Boolean(options.debug);
    this.onBatchResult = typeof options.onBatchResult === 'function' ? options.onBatchResult : function() {};
    this.onBatchError = typeof options.onBatchError === 'function' ? options.onBatchError : function() {};
    this.onStatsChange = typeof options.onStatsChange === 'function' ? options.onStatsChange : function() {};

    this.sockets = [];
    this.started = false;
    this.closed = false;
    this.nextBatchId = 1;
    this.startedAt = 0;
    this.stats = {
      configuredConnections: this.connectionCount,
      readyConnections: 0,
      inFlightBatches: 0,
      inFlightModels: 0,
      completedBatches: 0,
      failedBatches: 0,
      timedOutBatches: 0,
      bytesReceived: 0,
      lastBatchLatencyMs: 0,
      avgBatchLatencyMs: 0,
      throughputMBps: 0,
    };
  }

  configure(options = {}) {
    const nextUrl = options.url || this.url;
    const nextCount = Math.max(1, Number(options.connectionCount || this.connectionCount));
    const nextTimeout = Math.max(1000, Number(options.batchTimeoutMs || this.batchTimeoutMs));
    const shouldRestart = nextUrl !== this.url || nextCount !== this.connectionCount;

    this.url = nextUrl;
    this.connectionCount = nextCount;
    this.batchTimeoutMs = nextTimeout;
    this.debug = Boolean(options.debug);
    this.stats.configuredConnections = this.connectionCount;

    if (shouldRestart && this.started) {
      this.close();
      this.started = false;
      this.closed = false;
      this.sockets = [];
      this.start();
    }
  }

  start() {
    if (this.started || !this.url) return;
    this.started = true;
    this.closed = false;
    this.startedAt = this._now();

    for (let i = 0; i < this.connectionCount; i++) {
      this._connect(i);
    }
  }

  close() {
    this.closed = true;
    for (const socket of this.sockets) {
      if (socket.reconnectTimer) {
        clearTimeout(socket.reconnectTimer);
        socket.reconnectTimer = null;
      }
      if (socket.timeout) {
        clearTimeout(socket.timeout);
        socket.timeout = null;
      }
      socket.closedByPool = true;
      try {
        if (socket.ws) socket.ws.close();
      } catch (err) {
        // Ignore close errors during teardown.
      }
      socket.ready = false;
      socket.inflight = null;
    }
    this._updateStats();
  }

  hasReadyConnection() {
    return this.sockets.some((socket) => this._isSocketDispatchable(socket));
  }

  getReadyCount() {
    return this.sockets.filter((socket) => socket.ready && socket.ws && socket.ws.readyState === WebSocket.OPEN).length;
  }

  getInFlightBatchCount() {
    return this.sockets.filter((socket) => socket.inflight).length;
  }

  getInFlightModelCount() {
    return this.sockets.reduce((sum, socket) => {
      return sum + (socket.inflight ? socket.inflight.reqDescs.length : 0);
    }, 0);
  }

  getStats() {
    this._collectStats();
    return Object.assign({}, this.stats);
  }

  dispatchBatch(reqData, reqDescs) {
    if (!this.started) this.start();
    if (!Array.isArray(reqData) || reqData.length === 0) return false;

    const socket = this.sockets.find((item) => this._isSocketDispatchable(item));
    if (!socket) {
      return false;
    }

    const batch = {
      id: this.nextBatchId++,
      type: 'req',
      reqData,
      reqDescs,
      socketId: socket.id,
      startedAt: this._now(),
    };
    socket.inflight = batch;
    socket.timeout = setTimeout(() => {
      if (socket.inflight !== batch) return;
      this.stats.timedOutBatches++;
      this._failSocketBatch(socket, 'timeout', new Error('resourcesWS batch timeout'));
      try {
        socket.ws.close();
      } catch (err) {
        // The close path will reconnect.
      }
    }, this.batchTimeoutMs);

    try {
      socket.ws.send(JSON.stringify({
        type: batch.type,
        data: batch.reqData,
      }));
      if (this.debug) {
        console.log('[NeuralResourceWSPool] batch sent', {
          socketId: socket.id,
          batchId: batch.id,
          count: batch.reqData.length,
        });
      }
      this._updateStats();
      return true;
    } catch (err) {
      this._failSocketBatch(socket, 'send-error', err);
      socket.ready = false;
      try {
        socket.ws.close();
      } catch (closeErr) {
        // Closing an already-failed socket is best-effort.
      }
      return true;
    }
  }

  _isSocketDispatchable(socket) {
    return Boolean(socket &&
      socket.ready &&
      !socket.inflight &&
      socket.ws &&
      socket.ws.readyState === WebSocket.OPEN);
  }

  _connect(id) {
    if (this.closed || !this.url) return;

    const socket = this.sockets[id] || {
      id,
      ws: null,
      ready: false,
      inflight: null,
      timeout: null,
      reconnectAttempt: 0,
      reconnectTimer: null,
      closedByPool: false,
    };
    socket.closedByPool = false;
    this.sockets[id] = socket;

    try {
      socket.ws = new WebSocket(this.url);
      socket.ws.binaryType = 'arraybuffer';
    } catch (err) {
      this._scheduleReconnect(socket);
      return;
    }

    socket.ws.onopen = () => {
      socket.ready = true;
      socket.reconnectAttempt = 0;
      if (this.debug) console.log('[NeuralResourceWSPool] connected', { socketId: socket.id });
      this._updateStats();
    };

    socket.ws.onmessage = (event) => {
      this._handleMessage(socket, event);
    };

    socket.ws.onerror = (event) => {
      if (this.debug) console.warn('[NeuralResourceWSPool] socket error', { socketId: socket.id, event });
    };

    socket.ws.onclose = () => {
      socket.ready = false;
      if (socket.inflight) {
        this._failSocketBatch(socket, 'socket-closed', new Error('resourcesWS socket closed'));
      }
      this._updateStats();
      if (!socket.closedByPool && !this.closed) {
        this._scheduleReconnect(socket);
      }
    };
  }

  async _handleMessage(socket, event) {
    const batch = socket.inflight;
    if (!batch) return;

    try {
      const arrayBuffer = typeof Blob !== 'undefined' && event.data instanceof Blob
        ? await event.data.arrayBuffer()
        : event.data;
      const parsed = this._parseBatchPayload(arrayBuffer);
      const latencyMs = this._now() - batch.startedAt;

      if (socket.timeout) {
        clearTimeout(socket.timeout);
        socket.timeout = null;
      }
      socket.inflight = null;

      this.stats.completedBatches++;
      this.stats.bytesReceived += parsed.byteLength;
      this.stats.lastBatchLatencyMs = latencyMs;
      this.stats.avgBatchLatencyMs = this.stats.completedBatches <= 1
        ? latencyMs
        : (this.stats.avgBatchLatencyMs * 0.9) + (latencyMs * 0.1);
      const elapsedSeconds = Math.max(0.001, (this._now() - this.startedAt) / 1000);
      this.stats.throughputMBps = (this.stats.bytesReceived / 1048576) / elapsedSeconds;

      if (this.debug) {
        console.log('[NeuralResourceWSPool] batch received', {
          socketId: socket.id,
          batchId: batch.id,
          requested: batch.reqData.length,
          returned: parsed.buffers.length,
          latencyMs,
          bytes: parsed.byteLength,
        });
      }

      this.onBatchResult(batch, parsed.buffers, {
        socketId: socket.id,
        latencyMs,
        bytesReceived: parsed.byteLength,
        responseType: parsed.desc.type,
      });
      this._updateStats();
    } catch (err) {
      this._failSocketBatch(socket, 'parse-response-error', err);
      socket.ready = false;
      try {
        socket.ws.close();
      } catch (closeErr) {
        // The close path will reconnect.
      }
    }
  }

  _parseBatchPayload(arrayBuffer) {
    let bufferOffset = 0;
    const uint32View = new Uint32Array(arrayBuffer.slice(bufferOffset, bufferOffset + 4));
    bufferOffset += 4;

    const descJsonLength = uint32View[0];
    const jsonDescString = new TextDecoder().decode(arrayBuffer.slice(bufferOffset, bufferOffset + descJsonLength));
    bufferOffset += descJsonLength;
    const desc = JSON.parse(jsonDescString);
    const buffers = [];

    for (let i = 0; i < (desc.bufLengths || []).length; i++) {
      const length = desc.bufLengths[i];
      buffers.push(arrayBuffer.slice(bufferOffset, bufferOffset + length));
      bufferOffset += length;
    }

    return {
      desc,
      buffers,
      byteLength: arrayBuffer.byteLength || 0,
    };
  }

  _failSocketBatch(socket, reason, error) {
    const batch = socket.inflight;
    if (!batch) return;

    if (socket.timeout) {
      clearTimeout(socket.timeout);
      socket.timeout = null;
    }
    socket.inflight = null;
    this.stats.failedBatches++;

    if (reason === 'timeout' || reason === 'socket-closed' || reason === 'send-error' || reason === 'parse-response-error') {
      socket.ready = false;
    }

    if (this.debug) {
      console.warn('[NeuralResourceWSPool] batch failed', {
        socketId: socket.id,
        batchId: batch.id,
        reason,
        message: error && error.message ? error.message : error,
      });
    }

    this.onBatchError(batch, reason, error);
    this._updateStats();
  }

  _scheduleReconnect(socket) {
    if (socket.reconnectTimer || this.closed) return;
    const delays = [500, 1000, 2000, 5000];
    const delay = delays[Math.min(socket.reconnectAttempt, delays.length - 1)];
    socket.reconnectAttempt++;
    socket.reconnectTimer = setTimeout(() => {
      socket.reconnectTimer = null;
      this._connect(socket.id);
    }, delay);
  }

  _updateStats() {
    this._collectStats();
    this.onStatsChange(this.getStats());
  }

  _collectStats() {
    this.stats.configuredConnections = this.connectionCount;
    this.stats.readyConnections = this.getReadyCount();
    this.stats.inFlightBatches = this.getInFlightBatchCount();
    this.stats.inFlightModels = this.getInFlightModelCount();
  }

  _now() {
    return typeof performance !== 'undefined' ? performance.now() : Date.now();
  }
}
