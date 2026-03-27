// src/audio.js — Web Audio API chiptune music + SFX
// All sound is synthesised: no external files required.
// AudioContext is created lazily on first call so browsers don't block it
// before a user gesture.

let ctx     = null;   // AudioContext
let master  = null;   // master GainNode
let bgmLoop = null;   // setTimeout handle for loop rescheduling
let bgmOn   = false;

// ── Context ───────────────────────────────────────────────────────────────────

function getCtx() {
  if (!ctx) {
    ctx    = new (window.AudioContext || window.webkitAudioContext)();
    master = ctx.createGain();
    master.gain.value = 0.28;
    master.connect(ctx.destination);
  }
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}

// ── Tone primitive ────────────────────────────────────────────────────────────

function tone(freq, startTime, durationSec, type = 'square', vol = 0.15, dest = null) {
  if (!freq) return;
  const c   = getCtx();
  const osc = c.createOscillator();
  const g   = c.createGain();
  osc.type            = type;
  osc.frequency.value = freq;
  g.gain.setValueAtTime(vol, startTime);
  g.gain.exponentialRampToValueAtTime(0.0001, startTime + durationSec);
  osc.connect(g);
  g.connect(dest ?? master);
  osc.start(startTime);
  osc.stop(startTime + durationSec + 0.02);
}

// ── Note name → Hz ────────────────────────────────────────────────────────────

const SEMITONES = { C:-9, 'C#':-8, D:-7, 'D#':-6, E:-5, F:-4, 'F#':-3, G:-2, 'G#':-1, A:0, 'A#':1, B:2 };

function hz(name) {
  const m = name.match(/^([A-G]#?)(\d)$/);
  if (!m) return 0;
  return 440 * 2 ** ((SEMITONES[m[1]] + (parseInt(m[2]) - 4) * 12) / 12);
}

// ── Background music ──────────────────────────────────────────────────────────
// 4-bar loop, I–V–vi–IV in C major (C–G–Am–F), 155 BPM.
// Each entry: [ noteName, beatOffset, durationBeats ]

const BPM      = 155;
const BEAT     = 60 / BPM;
const LOOP_LEN = 16; // beats (4 bars × 4 beats)

const MELODY = [
  // Bar 1 — C
  ['E5', 0,    0.45], ['G5', 0.5,  0.45], ['C5', 1,    0.45], ['E5', 1.5,  0.45],
  ['G5', 2,    0.45], ['A5', 2.5,  0.45], ['G5', 3,    0.45], ['E5', 3.5,  0.45],
  // Bar 2 — G
  ['D5', 4,    0.45], ['G5', 4.5,  0.45], ['B4', 5,    0.45], ['D5', 5.5,  0.45],
  ['G5', 6,    0.45], ['B5', 6.5,  0.45], ['G5', 7,    0.45], ['D5', 7.5,  0.45],
  // Bar 3 — Am
  ['E5', 8,    0.45], ['A4', 8.5,  0.45], ['C5', 9,    0.45], ['E5', 9.5,  0.45],
  ['A5', 10,   0.9 ],                     ['E5', 11,   0.45], ['C5', 11.5, 0.45],
  // Bar 4 — F
  ['F4', 12,   0.45], ['A4', 12.5, 0.45], ['C5', 13,   0.45], ['F5', 13.5, 0.45],
  ['C5', 14,   0.45], ['A4', 14.5, 0.45], ['F4', 15,   0.9 ],
];

const BASS = [
  ['C3', 0,  0.5], ['G2', 0.5, 0.3], ['C3', 1,  0.5], ['G2', 1.5, 0.3],
  ['C3', 2,  0.5], ['G2', 2.5, 0.3], ['C3', 3,  0.5], ['G2', 3.5, 0.3],

  ['G2', 4,  0.5], ['D3', 4.5, 0.3], ['G2', 5,  0.5], ['D3', 5.5, 0.3],
  ['G2', 6,  0.5], ['D3', 6.5, 0.3], ['G2', 7,  0.5], ['D3', 7.5, 0.3],

  ['A2', 8,  0.5], ['E3', 8.5, 0.3], ['A2', 9,  0.5], ['E3', 9.5, 0.3],
  ['A2', 10, 0.5], ['E3', 10.5,0.3], ['A2', 11, 0.5], ['E3', 11.5,0.3],

  ['F2', 12, 0.5], ['C3', 12.5,0.3], ['F2', 13, 0.5], ['C3', 13.5,0.3],
  ['F2', 14, 0.5], ['C3', 14.5,0.3], ['F2', 15, 0.5], ['C3', 15.5,0.3],
];

// Rhythmic hi-hat / pulse using white noise for a subtle percussion feel
function schedulePercussion(startTime) {
  const c = getCtx();
  for (let i = 0; i < LOOP_LEN * 2; i++) {     // 8th-note pulses
    const t    = startTime + i * BEAT * 0.5;
    const dur  = 0.04;
    const buf  = c.createBuffer(1, c.sampleRate * dur, c.sampleRate);
    const data = buf.getChannelData(0);
    for (let s = 0; s < data.length; s++) data[s] = (Math.random() * 2 - 1);
    const src = c.createBufferSource();
    src.buffer = buf;
    const g = c.createGain();
    // Accent on beats 1 and 3 of each bar
    const beat = i * 0.5;
    const accent = (beat % 4 === 0 || beat % 4 === 2);
    g.gain.setValueAtTime(accent ? 0.04 : 0.018, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(g);
    g.connect(master);
    src.start(t);
    src.stop(t + dur + 0.01);
  }
}

function scheduleLoop(startTime) {
  if (!bgmOn) return;
  const c = getCtx();

  for (const [note, beat, dur] of MELODY) {
    tone(hz(note), startTime + beat * BEAT, dur * BEAT, 'square', 0.07);
  }
  for (const [note, beat, dur] of BASS) {
    tone(hz(note), startTime + beat * BEAT, dur * BEAT, 'triangle', 0.11);
  }
  schedulePercussion(startTime);

  // Reschedule 300 ms before this loop ends to ensure seamless looping
  const loopSec = LOOP_LEN * BEAT;
  bgmLoop = setTimeout(() => scheduleLoop(startTime + loopSec), (loopSec - 0.3) * 1000);
}

export function startBgm() {
  if (bgmOn) return;
  bgmOn = true;
  scheduleLoop(getCtx().currentTime + 0.05);
}

export function stopBgm() {
  bgmOn = false;
  clearTimeout(bgmLoop);
  bgmLoop = null;
}

export function pauseBgm() { stopBgm(); }
export function resumeBgm() { startBgm(); }

// ── Sound effects ─────────────────────────────────────────────────────────────

// Short crisp click on ball deflection
export function sfxBallHit() {
  const t = getCtx().currentTime;
  tone(900,  t,        0.035, 'square',   0.20);
  tone(1200, t + 0.01, 0.025, 'square',   0.10);
}

// Upward 3-note sting on scoring
export function sfxScore() {
  const t = getCtx().currentTime;
  tone(hz('C5'), t,        0.09, 'square', 0.22);
  tone(hz('E5'), t + 0.09, 0.09, 'square', 0.22);
  tone(hz('G5'), t + 0.18, 0.18, 'square', 0.22);
}

// Downward buzz on snake death
export function sfxDeath() {
  const t = getCtx().currentTime;
  tone(320, t,        0.09, 'sawtooth', 0.22);
  tone(210, t + 0.09, 0.09, 'sawtooth', 0.22);
  tone(110, t + 0.18, 0.22, 'sawtooth', 0.22);
}

// 3-note ascending chime on power-up collect
export function sfxPowerup() {
  const t = getCtx().currentTime;
  tone(hz('G5'), t,        0.07, 'square', 0.18);
  tone(hz('B5'), t + 0.07, 0.07, 'square', 0.18);
  tone(hz('E6'), t + 0.14, 0.14, 'square', 0.18);
}

// Rising 4-note fanfare on game win
export function sfxWin() {
  const t = getCtx().currentTime;
  [hz('C5'), hz('E5'), hz('G5'), hz('C6')].forEach((f, i) => {
    tone(f, t + i * 0.13, 0.18, 'square', 0.22);
  });
}
