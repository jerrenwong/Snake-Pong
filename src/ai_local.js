// In-browser ONNX inference for the trained Snake-Pong Q-net (plays as P2).
//
// The pure obs-building logic lives in ./ai_local_obs.js so that a parity test
// can import it under Node without dragging in onnxruntime-web. This file only
// handles model loading, caching, and inference.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.mjs';
import {
  ACTION_DELTAS,
  COLS,
  MIRROR_ACTION,
  ROWS,
  buildObsP2,
  updatePhase,
} from './ai_local_obs.js';

// Safety filter: compute which of the 4 ego actions would cause the AI's next
// head to land on a fatal cell (out-of-bounds, own body-except-tail, opponent
// body). Returns a length-4 boolean array where `true = safe`.
//
// Why this helps: v12's policy learned a strong "chase ball" reflex and can
// commit to a direction that walks straight into the opponent's trail (see
// the black-divider failure mode). Masking illegal actions before argmax
// removes the worst-case crash without changing the policy's preferences
// on safe moves.
function _legalActionsP2(s1, s2) {
  const head = s2.body[0];
  const curDir = s2.dir;
  // Self-body cells the head CAN'T move into. Exclude the tail (body[L-1])
  // because that cell vacates on the next step.
  const selfBlocked = new Set();
  for (let i = 0; i < s2.body.length - 1; i++) {
    selfBlocked.add(`${s2.body[i].x},${s2.body[i].y}`);
  }
  const oppBlocked = new Set();
  for (const c of s1.body) oppBlocked.add(`${c.x},${c.y}`);

  const legal = [false, false, false, false];
  for (let ego = 0; ego < 4; ego++) {
    const realA = MIRROR_ACTION[ego];
    let [rdx, rdy] = ACTION_DELTAS[realA];
    // Anti-reverse: if picked action is opposite current dir, the engine
    // ignores it and the snake keeps moving forward — so that's the cell to
    // check, not the "would-be" reversed one.
    if (rdx === -curDir.x && rdy === -curDir.y) {
      rdx = curDir.x; rdy = curDir.y;
    }
    const nx = head.x + rdx;
    const ny = head.y + rdy;
    if (nx < 0 || nx >= COLS || ny < 0 || ny >= ROWS) continue;
    const key = `${nx},${ny}`;
    if (selfBlocked.has(key) || oppBlocked.has(key)) continue;
    legal[ego] = true;
  }
  return legal;
}

// Stochastic-sampling config. When `temperature` > 0 the policy samples
// from the legal actions instead of taking argmax: scale logits by 1/T,
// softmax over legal actions, keep the smallest nucleus whose cumulative
// prob ≥ topP, sample. Currently disabled (every variant uses pure argmax).
const SAMPLING_BY_VARIANT = {};

export class LocalAI {
  constructor() {
    this.session = null;
    this.meta = null;
    this.lastBallKey = null;
    this.phase = 0;
    this.ready = false;
    this.variant = null;
  }

  async load(onnxPath, metaPath, variant = null) {
    this.variant = variant;
    // Point WASM runtime at the same CDN as the module (avoids CORS/COEP fuss).
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
    // Force single-threaded WASM: cross-origin isolation isn't guaranteed in this
    // deployment, and the tiny MLP is fast enough that threads wouldn't help.
    ort.env.wasm.numThreads = 1;

    const metaResp = await fetch(metaPath);
    if (!metaResp.ok) throw new Error(`failed to fetch ${metaPath}: ${metaResp.status}`);
    this.meta = await metaResp.json();

    this.session = await ort.InferenceSession.create(onnxPath, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    this.ready = true;
  }

  reset() {
    this.lastBallKey = null;
    this.phase = 0;
  }

  // Run one inference step for s2 (P2). Returns { dx, dy, action, q }.
  async decide(s1, s2, ball) {
    if (!this.ready) return null;

    const updated = updatePhase(this.phase, this.lastBallKey, ball, this.meta.snake_multiplier);
    this.phase = updated.phase;
    this.lastBallKey = updated.lastBallKey;

    const obs = buildObsP2(s1, s2, ball, this.meta, this.phase);
    const tensor = new ort.Tensor('float32', obs, [1, obs.length]);
    const results = await this.session.run({ [this.meta.input_name]: tensor });
    const qRaw = results[this.meta.output_name].data;
    const q = Array.from(qRaw);

    // Pick action among the LEGAL set. Default = argmax (greedy). For
    // variants in SAMPLING_BY_VARIANT, do top-p nucleus sampling at the
    // configured temperature instead.
    const legal = _legalActionsP2(s1, s2);
    const sampling = SAMPLING_BY_VARIANT[this.variant];
    let bestA = -1;
    let safetyOverride = false;

    const legalIdx = [];
    for (let i = 0; i < legal.length; i++) if (legal[i]) legalIdx.push(i);

    if (legalIdx.length === 0) {
      // No safe move — doomed regardless. Take the model's top pick.
      bestA = 0;
      let bestQ = q[0];
      for (let i = 1; i < q.length; i++) {
        if (q[i] > bestQ) { bestQ = q[i]; bestA = i; }
      }
    } else if (sampling) {
      // Temperature softmax over legal logits.
      const T = Math.max(sampling.temperature, 1e-6);
      const scaled = legalIdx.map(i => q[i] / T);
      const m = Math.max.apply(null, scaled);
      const exps = scaled.map(v => Math.exp(v - m));
      const Z = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(e => e / Z);
      const order = probs.map((_, j) => j).sort((a, b) => probs[b] - probs[a]);
      let cum = 0, cutoff = order.length;
      for (let k = 0; k < order.length; k++) {
        cum += probs[order[k]];
        if (cum >= sampling.topP) { cutoff = k + 1; break; }
      }
      const nucleus = order.slice(0, cutoff);
      const nucleusZ = nucleus.reduce((s, j) => s + probs[j], 0);
      const r = Math.random() * nucleusZ;
      let acc = 0, picked = nucleus[nucleus.length - 1];
      for (const j of nucleus) {
        acc += probs[j];
        if (r <= acc) { picked = j; break; }
      }
      bestA = legalIdx[picked];
      safetyOverride = bestA !== legalIdx[order[0]];
    } else {
      let bestQ = -Infinity;
      for (const i of legalIdx) {
        if (q[i] > bestQ) { bestQ = q[i]; bestA = i; }
      }
      let rawBest = 0, rawBestQ = q[0];
      for (let i = 1; i < q.length; i++) {
        if (q[i] > rawBestQ) { rawBestQ = q[i]; rawBest = i; }
      }
      safetyOverride = rawBest !== bestA;
    }
    const realA = MIRROR_ACTION[bestA];
    const [dx, dy] = ACTION_DELTAS[realA];
    return { dx, dy, action: bestA, q, legal, safetyOverride };
  }
}
