import { bitsetFromIds, countBitsetIds, diffIdBitsets } from './sortedIdDelta.js';

function appendWordIds(output, rawBits, wordIndex, bitCount) {
  let bits = rawBits >>> 0;
  while (bits !== 0) {
    const leastBit = bits & -bits;
    const bitIndex = 31 - Math.clz32(leastBit);
    const id = wordIndex * 32 + bitIndex;
    if (id < bitCount) output.push(id);
    bits = (bits & (bits - 1)) >>> 0;
  }
}

export class IdBitsetState {
  constructor(bitCount = 0) {
    this.reset(bitCount);
  }

  reset(bitCount = this.bitCount) {
    this.bitCount = Math.max(0, Number(bitCount) || 0);
    this.words = new Uint32Array(Math.ceil(this.bitCount / 32));
    this.count = 0;
    return this;
  }

  replace(ids, bitCount = this.bitCount) {
    const nextBitCount = Math.max(0, Number(bitCount) || 0);
    if (nextBitCount !== this.bitCount) this.reset(nextBitCount);
    const nextWords = bitsetFromIds(ids, this.bitCount);
    const delta = diffIdBitsets(this.words, nextWords, this.bitCount);
    this.words = nextWords;
    this.count = countBitsetIds(nextWords, this.bitCount);
    return { ...delta, count: this.count };
  }

  applyDelta(addedIds, removedIds) {
    const actualAdded = [];
    const actualRemoved = [];
    for (const rawId of removedIds || []) {
      const id = Number(rawId);
      if (!Number.isInteger(id) || id < 0 || id >= this.bitCount) continue;
      const wordIndex = id >>> 5;
      const mask = (1 << (id & 31)) >>> 0;
      if ((this.words[wordIndex] & mask) === 0) continue;
      this.words[wordIndex] = (this.words[wordIndex] & ~mask) >>> 0;
      this.count -= 1;
      actualRemoved.push(id);
    }
    for (const rawId of addedIds || []) {
      const id = Number(rawId);
      if (!Number.isInteger(id) || id < 0 || id >= this.bitCount) continue;
      const wordIndex = id >>> 5;
      const mask = (1 << (id & 31)) >>> 0;
      if ((this.words[wordIndex] & mask) !== 0) continue;
      this.words[wordIndex] = (this.words[wordIndex] | mask) >>> 0;
      this.count += 1;
      actualAdded.push(id);
    }
    return {
      added: Uint32Array.from(actualAdded),
      removed: Uint32Array.from(actualRemoved),
      count: this.count,
    };
  }

  has(rawId) {
    const id = Number(rawId);
    if (!Number.isInteger(id) || id < 0 || id >= this.bitCount) return false;
    return (this.words[id >>> 5] & ((1 << (id & 31)) >>> 0)) !== 0;
  }

  toIds() {
    const ids = [];
    for (let wordIndex = 0; wordIndex < this.words.length; wordIndex += 1) {
      appendWordIds(ids, this.words[wordIndex], wordIndex, this.bitCount);
    }
    return ids;
  }
}
