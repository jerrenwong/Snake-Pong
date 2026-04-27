// Standalone "INSANE @ 3×" page. Loads the INSANE ONNX (trained at
// snake_multiplier=2) and pits it against the player at snake_multiplier=3
// — strictly off-distribution for the policy. Reuses the existing engine
// and renderer; no online play, no power-ups, no AI tier picker, no boss.

import { createRenderer } from './renderer.js';
import { registerInput }  from './input.js';
import {
  createSnakes, createBall,
  stepSnake, snakeHitsDeath, snakesCollide, stepBall,
  getBallTps,
} from './logic.js';
import { LocalAI } from './ai_local.js';

const SNAKE_LEN  = 4;
const MULT       = 3;       // forced: play boss-speed, regardless of training mult.
const BALL_SPEED = 3;       // matches default difficulty slider.
const WIN_SCORE  = 7;

const canvas   = document.getElementById('game');
const overlay  = document.getElementById('overlay');
const ovTitle  = document.getElementById('ov-title');
const ovMsg    = document.getElementById('ov-msg');
const startBtn = document.getElementById('start-btn');
const p1Pts    = document.getElementById('p1-pts');
const p2Pts    = document.getElementById('p2-pts');

const { draw } = createRenderer(canvas);

let phase  = 'menu'; // menu | playing | gameover
let s1 = null, s2 = null, ball = null;
let score1 = 0, score2 = 0;

// `liftStart=true` shifts the spawn row up by 1 — the same trick the main
// page uses for 3× speed so the player isn't immediately backed into a wall.
const LIFT_START = true;

let ballTickMs = 1000 / getBallTps(BALL_SPEED);
let tickMs     = ballTickMs / MULT;
let tickAccum  = 0;
let ballAccum  = 0;
let lastTs     = 0;
let rafId      = null;

// AI state
const ai = new LocalAI();
let aiPendingDir = null;
let aiInferring  = false;

// ── AI inference plumbing ────────────────────────────────────────────────
function applyAIDirToS2() {
  if (!aiPendingDir || !s2) return;
  const { dx, dy } = aiPendingDir;
  if (dx !== 0 && s2.dir.x === -dx) return;
  if (dy !== 0 && s2.dir.y === -dy) return;
  s2.nextDir = { x: dx, y: dy };
}

async function runInference() {
  if (!ai.ready || aiInferring) return;
  if (phase !== 'playing' || !s1 || !s2 || !ball) return;
  aiInferring = true;
  try {
    const res = await ai.decide(s1, s2, ball);
    if (res) aiPendingDir = { dx: res.dx, dy: res.dy };
  } catch (e) {
    console.error('[insane-3x] inference error:', e);
  } finally {
    aiInferring = false;
  }
}

// ── Game tick ────────────────────────────────────────────────────────────
function tick() {
  applyAIDirToS2();
  stepSnake(s1);
  stepSnake(s2);

  const d1 = snakeHitsDeath(s1, null);
  const d2 = snakeHitsDeath(s2, null);
  if (d1 && d2) { endRound(0); return; }
  if (d1)       { endRound(2); return; }
  if (d2)       { endRound(1); return; }

  const sc = snakesCollide(s1, s2);
  if (sc === 'both') { endRound(0); return; }
  if (sc === 's1')   { endRound(2); return; }
  if (sc === 's2')   { endRound(1); return; }
}

function loop(ts) {
  rafId = requestAnimationFrame(loop);
  const dt = Math.min(ts - lastTs, 150);
  lastTs = ts;

  if (phase === 'playing') {
    tickAccum += dt;
    let stateChanged = false;
    while (tickAccum >= tickMs) {
      tick();
      tickAccum -= tickMs;
      stateChanged = true;
      if (phase !== 'playing') { tickAccum = 0; ballAccum = 0; break; }
    }
    if (stateChanged) runInference();

    if (phase === 'playing') {
      ballAccum += dt;
      while (ballAccum >= ballTickMs) {
        const scorer = stepBall(ball, s1, s2, null);
        ballAccum -= ballTickMs;
        if (scorer !== null) { awardPoint(scorer); ballAccum = 0; break; }
      }
    }
  }

  draw(s1, s2, ball, [], [], []);
}

// ── State transitions ────────────────────────────────────────────────────
function startMatch() {
  score1 = 0; score2 = 0;
  p1Pts.textContent = 0;
  p2Pts.textContent = 0;
  startRound();
}

function startRound() {
  const fresh = createSnakes(SNAKE_LEN, LIFT_START);
  s1 = fresh.s1; s1.speedMult = 1;
  s2 = fresh.s2; s2.speedMult = 1;
  // Tint P2 to match the INSANE tier badge in the main UI.
  s2.color = '#e22';
  s2.effect = 'insane-glow';
  ball = createBall(-1);  // serve from P1's side so the human gets a beat.
  ai.reset();
  aiPendingDir = null;
  tickAccum = 0;
  ballAccum = 0;
  overlay.style.display = 'none';
  phase = 'playing';
  if (rafId === null) {
    lastTs = performance.now();
    rafId = requestAnimationFrame(loop);
  }
}

function endRound(scorer) {
  // scorer = 0 (draw / both-died), 1 (P1 wins ball), 2 (P2 wins ball)
  if (scorer === 1 || scorer === 2) awardPoint(scorer);
  else setTimeout(startRound, 800);
}

function awardPoint(player) {
  if (player === 1) score1++; else score2++;
  p1Pts.textContent = score1;
  p2Pts.textContent = score2;
  if (score1 >= WIN_SCORE || score2 >= WIN_SCORE) {
    endMatch(score1 >= WIN_SCORE ? 1 : 2);
  } else {
    setTimeout(startRound, 700);
  }
}

function endMatch(winner) {
  phase = 'gameover';
  ovTitle.textContent = winner === 1 ? 'YOU WIN' : 'INSANE WINS';
  ovTitle.style.color = winner === 1 ? '#0ff' : '#f88';
  ovMsg.innerHTML = `Final score: ${score1} – ${score2}`;
  startBtn.textContent = 'PLAY AGAIN';
  startBtn.disabled = false;
  overlay.style.display = 'flex';
}

// ── Input ────────────────────────────────────────────────────────────────
registerInput({
  onEscape() { /* no pause for the standalone page */ },
  onDirectionP1(dx, dy) {
    if (phase !== 'playing' || !s1) return;
    if (dx !== 0 && s1.dir.x === -dx) return;
    if (dy !== 0 && s1.dir.y === -dy) return;
    s1.nextDir = { x: dx, y: dy };
  },
  onDirectionP2() { /* AI controls P2 */ },
});

// ── Boot ─────────────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  if (!ai.ready) {
    startBtn.disabled = true;
    startBtn.textContent = 'LOADING…';
    try {
      await ai.load(
        'models/snake-pong-insane.onnx',
        'models/snake-pong-insane.json',
        'insane',
      );
    } catch (e) {
      console.error('[insane-3x] load failed:', e);
      ovMsg.textContent = 'AI LOAD FAILED: ' + (e && e.message || e);
      startBtn.textContent = 'RETRY';
      startBtn.disabled = false;
      return;
    }
    startBtn.disabled = false;
  }
  startBtn.textContent = 'START';
  startMatch();
});
