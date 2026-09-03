const TIER_RANK = Object.freeze({
  urgent: 0,
  startup: 1,
  warm: 2,
  speculative: 3,
});

const ACTIVE_STATES = new Set(['fetching', 'parsing', 'mounting']);

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

function validItem(item) {
  return Boolean(item && item.hash && Number.isInteger(Number(item.glbId)) && Number(item.glbId) >= 0);
}

function finiteScore(item) {
  const score = Number(item?.score ?? item?.modelInfo?.weight ?? 0);
  return Number.isFinite(score) ? score : 0;
}

export function classifyGlbSchedule(modelInfos, prefetchInfos, renderGlbIds, mode = 'viewcell-priority') {
  const predictedById = new Map();
  for (const item of modelInfos || []) {
    const id = Number(item?.id);
    if (Number.isInteger(id) && id >= 0) predictedById.set(id, item);
  }
  const renderIds = Array.from(new Set(Array.from(renderGlbIds || [])
    .map(Number)
    .filter((id) => Number.isInteger(id) && id >= 0)));
  const renderIdSet = new Set(renderIds);
  const urgent = renderIds.map((id) => ({
    ...(predictedById.get(id) || { id, weight: 1, idMode: 'global-glb' }),
    prefetch: false,
    renderVisible: true,
  }));
  if (mode === 'raw-visible') {
    const urgentIds = new Set(urgent.map((item) => Number(item.id)));
    for (const item of modelInfos || []) {
      if (!urgentIds.has(Number(item.id))) urgent.push({ ...item, prefetch: false });
    }
    return { urgent, warm: [], speculative: [] };
  }

  const warm = (modelInfos || []).filter((item) => !renderIdSet.has(Number(item.id)))
    .map((item) => ({ ...item, prefetch: true, deferredVisible: true }));
  const claimed = new Set([...urgent, ...warm].map((item) => Number(item.id)));
  const speculative = (prefetchInfos || []).filter((item) => !claimed.has(Number(item.id)))
    .map((item) => ({ ...item, prefetch: true, speculative: true }));
  return { urgent, warm, speculative };
}

function nodeBefore(a, b) {
  if (a.readyAt !== b.readyAt) return a.readyAt < b.readyAt;
  if (a.tierRank !== b.tierRank) return a.tierRank < b.tierRank;
  if (a.score !== b.score) return a.score > b.score;
  return a.sequence < b.sequence;
}

export class GlbResourceScheduler {
  constructor(options = {}) {
    this.clock = options.clock || nowMs;
    this.onWorkAvailable = typeof options.onWorkAvailable === 'function'
      ? options.onWorkAvailable
      : null;
    this.maxRetries = Math.max(0, Number(options.maxRetries ?? 4));
    this.retryDelaysMs = options.retryDelaysMs || [250, 1000, 3000, 10000];
    this.entries = new Map();
    this.glbIdToHash = new Map();
    this.heap = [];
    this.sequence = 0;
    this.revision = 0;
    this.wakeTimer = null;
  }

  reset() {
    if (this.wakeTimer != null) clearTimeout(this.wakeTimer);
    this.wakeTimer = null;
    this.entries.clear();
    this.glbIdToHash.clear();
    this.heap = [];
    this.sequence = 0;
    this.revision = 0;
  }

  _pushNode(entry) {
    entry.version = Number(entry.version || 0) + 1;
    const node = {
      hash: entry.hash,
      version: entry.version,
      readyAt: Number(entry.readyAt || 0),
      tierRank: TIER_RANK[entry.tier],
      score: entry.score,
      sequence: entry.sequence,
    };
    this.heap.push(node);
    let index = this.heap.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (nodeBefore(this.heap[parent], this.heap[index])) break;
      [this.heap[parent], this.heap[index]] = [this.heap[index], this.heap[parent]];
      index = parent;
    }
  }

  _popNode() {
    if (this.heap.length === 0) return null;
    const first = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = last;
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        const right = left + 1;
        let best = index;
        if (left < this.heap.length && nodeBefore(this.heap[left], this.heap[best])) best = left;
        if (right < this.heap.length && nodeBefore(this.heap[right], this.heap[best])) best = right;
        if (best === index) break;
        [this.heap[index], this.heap[best]] = [this.heap[best], this.heap[index]];
        index = best;
      }
    }
    return first;
  }

  _peekValidNode() {
    while (this.heap.length > 0) {
      const node = this.heap[0];
      const entry = this.entries.get(node.hash);
      if (entry && entry.desired && entry.status === 'queued' && entry.version === node.version) {
        return { node, entry };
      }
      this._popNode();
    }
    return null;
  }

  _scheduleWake() {
    if (!this.onWorkAvailable) return;
    const next = this._peekValidNode();
    if (!next) return;
    const delay = Math.max(0, next.node.readyAt - this.clock());
    if (this.wakeTimer != null) clearTimeout(this.wakeTimer);
    this.wakeTimer = setTimeout(() => {
      this.wakeTimer = null;
      this.onWorkAvailable();
    }, delay);
  }

  _upsert(item, tier, revision) {
    if (!validItem(item)) return null;
    const hash = String(item.hash);
    const glbId = Number(item.glbId);
    const score = finiteScore(item);
    let entry = this.entries.get(hash);
    if (!entry) {
      entry = {
        hash,
        glbId,
        modelInfo: { ...(item.modelInfo || {}) },
        tier,
        score,
        status: 'queued',
        desired: true,
        revision,
        retries: 0,
        readyAt: 0,
        sequence: this.sequence++,
        version: 0,
        lastError: null,
      };
      this.entries.set(hash, entry);
      this.glbIdToHash.set(glbId, hash);
      this._pushNode(entry);
      return entry;
    }

    const wasDesired = entry.desired;
    const previousTierRank = TIER_RANK[entry.tier];
    entry.glbId = glbId;
    entry.modelInfo = { ...(entry.modelInfo || {}), ...(item.modelInfo || {}) };
    entry.tier = !wasDesired || TIER_RANK[tier] < previousTierRank ? tier : entry.tier;
    entry.score = !wasDesired ? score : Math.max(entry.score, score);
    entry.desired = true;
    if (entry.revision !== revision && entry.status === 'failed') {
      entry.status = 'queued';
      entry.retries = 0;
      entry.readyAt = 0;
      entry.lastError = null;
    }
    entry.revision = revision;
    if (entry.status === 'queued') this._pushNode(entry);
    return entry;
  }

  setPlan(groups = {}, revision = this.revision + 1) {
    this.revision = Number(revision || this.revision + 1);
    for (const entry of this.entries.values()) entry.desired = false;

    for (const tier of ['urgent', 'warm', 'speculative']) {
      for (const item of groups[tier] || []) this._upsert(item, tier, this.revision);
    }

    for (const [hash, entry] of this.entries) {
      if (entry.desired) continue;
      entry.version += 1;
      if (!ACTIVE_STATES.has(entry.status) && entry.status !== 'resident') {
        this.entries.delete(hash);
        this.glbIdToHash.delete(entry.glbId);
      }
    }
    this._scheduleWake();
    return this.getSnapshot();
  }

  enqueueStartup(items = []) {
    for (const item of items) this._upsert(item, 'startup', this.revision);
    this._scheduleWake();
    return this.getSnapshot();
  }

  promote(items = []) {
    let promoted = 0;
    for (const item of items) {
      const existing = validItem(item) ? this.entries.get(String(item.hash)) : null;
      const previousTier = existing?.tier;
      const entry = this._upsert(item, 'urgent', this.revision);
      if (!entry) continue;
      entry.tier = 'urgent';
      entry.modelInfo.prefetch = false;
      if (entry.status === 'failed') {
        entry.status = 'queued';
        entry.retries = 0;
        entry.readyAt = 0;
        entry.lastError = null;
        this._pushNode(entry);
      }
      if (previousTier !== 'urgent') promoted += 1;
    }
    this._scheduleWake();
    return promoted;
  }

  reclassify(items = [], tier = 'warm') {
    if (!Object.prototype.hasOwnProperty.call(TIER_RANK, tier)) {
      throw new Error(`Unknown GLB resource tier: ${tier}`);
    }
    let changed = 0;
    for (const item of items) {
      if (!validItem(item)) continue;
      const existing = this.entries.get(String(item.hash));
      let entry = existing;
      if (!entry) entry = this._upsert(item, tier, this.revision);
      if (!entry) continue;
      if (entry.tier !== tier) changed += 1;
      entry.tier = tier;
      entry.modelInfo = { ...(entry.modelInfo || {}), ...(item.modelInfo || {}) };
      entry.modelInfo.prefetch = tier !== 'urgent';
      entry.score = finiteScore(item);
      entry.desired = true;
      if (existing && entry.status === 'queued') this._pushNode(entry);
    }
    this._scheduleWake();
    return changed;
  }

  takeNext() {
    const now = this.clock();
    while (true) {
      const next = this._peekValidNode();
      if (!next || next.node.readyAt > now) {
        this._scheduleWake();
        return null;
      }
      this._popNode();
      const entry = next.entry;
      entry.status = 'fetching';
      entry.modelInfo.prefetch = entry.tier !== 'urgent';
      return entry;
    }
  }

  _transition(hash, status) {
    const entry = this.entries.get(String(hash));
    if (!entry) return null;
    entry.status = status;
    return entry;
  }

  markParsing(hash) {
    return this._transition(hash, 'parsing');
  }

  markMounting(hash) {
    return this._transition(hash, 'mounting');
  }

  markResident(hash) {
    const entry = this._transition(hash, 'resident');
    if (entry) {
      entry.retries = 0;
      entry.readyAt = 0;
      entry.lastError = null;
    }
    return entry;
  }

  markDiscarded(hash) {
    const entry = this.entries.get(String(hash));
    if (!entry) return;
    if (entry.desired) {
      entry.status = 'queued';
      entry.readyAt = 0;
      this._pushNode(entry);
      this._scheduleWake();
    } else {
      this.entries.delete(String(hash));
      this.glbIdToHash.delete(entry.glbId);
    }
  }

  markFailed(hash, error = null) {
    const entry = this.entries.get(String(hash));
    if (!entry) return { retrying: false, delayMs: 0 };
    entry.retries += 1;
    entry.lastError = String(error?.message || error || 'resource load failed').slice(0, 240);
    if (!entry.desired || entry.retries > this.maxRetries) {
      entry.status = 'failed';
      return { retrying: false, delayMs: 0 };
    }
    const delayIndex = Math.min(entry.retries - 1, this.retryDelaysMs.length - 1);
    const delayMs = Math.max(0, Number(this.retryDelaysMs[delayIndex] || 0));
    entry.status = 'queued';
    entry.readyAt = this.clock() + delayMs;
    this._pushNode(entry);
    this._scheduleWake();
    return { retrying: true, delayMs };
  }

  markEvicted(hash) {
    const entry = this.entries.get(String(hash));
    if (!entry) return;
    if (!entry.desired) {
      this.entries.delete(String(hash));
      this.glbIdToHash.delete(entry.glbId);
      return;
    }
    entry.status = 'queued';
    entry.readyAt = 0;
    this._pushNode(entry);
    this._scheduleWake();
  }

  isWanted(hash) {
    return Boolean(this.entries.get(String(hash))?.desired);
  }

  getWantedHashes() {
    return new Set(Array.from(this.entries.values())
      .filter((entry) => entry.desired)
      .map((entry) => entry.hash));
  }

  hasForegroundWork() {
    return Array.from(this.entries.values()).some((entry) => (
      entry.desired && (entry.tier === 'urgent' || entry.tier === 'startup')
        && entry.status !== 'resident'
    ));
  }

  hasReadyFetch() {
    const next = this._peekValidNode();
    return Boolean(next && next.node.readyAt <= this.clock());
  }

  hasPendingWork() {
    return Array.from(this.entries.values()).some((entry) => (
      entry.desired && ['queued', 'fetching', 'parsing', 'mounting'].includes(entry.status)
    ));
  }

  nextWakeDelay() {
    const next = this._peekValidNode();
    return next ? Math.max(0, next.node.readyAt - this.clock()) : null;
  }

  getSnapshot() {
    const counts = {};
    const immediatePreview = [];
    const prefetchPreview = [];
    let wanted = 0;
    let urgentWanted = 0;
    let urgentResident = 0;
    let urgentFailed = 0;
    for (const entry of this.entries.values()) {
      if (!entry.desired) continue;
      wanted += 1;
      const key = `${entry.tier}:${entry.status}`;
      counts[key] = Number(counts[key] || 0) + 1;
      if (entry.tier === 'urgent') {
        urgentWanted += 1;
        if (entry.status === 'resident') urgentResident += 1;
        if (entry.status === 'failed') urgentFailed += 1;
      }
      if (entry.status === 'queued') {
        const target = entry.tier === 'urgent' ? immediatePreview : prefetchPreview;
        if (target.length >= 8) continue;
        target.push({
          id: entry.glbId,
          tier: entry.tier,
          score: entry.score,
          retries: entry.retries,
        });
      }
    }
    const sortPreview = (a, b) => TIER_RANK[a.tier] - TIER_RANK[b.tier]
      || b.score - a.score || a.id - b.id;
    immediatePreview.sort(sortPreview);
    prefetchPreview.sort(sortPreview);
    return {
      revision: this.revision,
      wanted,
      urgentWanted,
      urgentResident,
      urgentMissing: Math.max(0, urgentWanted - urgentResident),
      urgentFailed,
      queuedImmediate: Number(counts['urgent:queued'] || 0),
      queuedPrefetch: Number(counts['startup:queued'] || 0)
        + Number(counts['warm:queued'] || 0)
        + Number(counts['speculative:queued'] || 0),
      counts,
      immediatePreview,
      prefetchPreview,
    };
  }
}
