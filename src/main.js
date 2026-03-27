import { createRenderer }                                from './renderer.js';
import { registerInput }                                  from './input.js';
import { createSnakes, createBall,
         stepSnake, snakeHitsDeath, snakesCollide, stepBall,
         getBallTps }                                     from './logic.js';
import { startBgm, stopBgm, pauseBgm, resumeBgm,
         sfxBallHit, sfxScore, sfxDeath, sfxWin }        from './audio.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const lenSl  = document.getElementById('len-sl');
const lenV   = document.getElementById('len-v');
const bSpdSl = document.getElementById('bspd-sl');
const bSpdV  = document.getElementById('bspd-v');
const winSl  = document.getElementById('win-sl');
const winV   = document.getElementById('win-v');
const overlay    = document.getElementById('overlay');
const ovTitle    = document.getElementById('ov-title');
const ovMsg      = document.getElementById('ov-msg');
const startBtn   = document.getElementById('start-btn');
const p1Pts      = document.getElementById('p1-pts');
const p2Pts      = document.getElementById('p2-pts');
const settingsBtn   = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const multGroup     = document.getElementById('mult-group');

// Slider display sync
lenSl.addEventListener('input',  () => lenV.textContent  = lenSl.value);
bSpdSl.addEventListener('input', () => bSpdV.textContent = bSpdSl.value);
winSl.addEventListener('input',  () => winV.textContent  = winSl.value);

// ── Renderer ──────────────────────────────────────────────────────────────────
const { draw } = createRenderer(document.getElementById('game'));

// ── Game state ────────────────────────────────────────────────────────────────
let phase  = 'menu'; // menu | playing | paused | roundend | gameover
let score1 = 0, score2 = 0;
let s1 = null, s2 = null, ball = null;
let snakeMultiplier = 1; // snake moves N× per ball tick (1, 2, or 3)

// ── RAF loop ──────────────────────────────────────────────────────────────────
let rafId = null, lastTs = 0;
let tickAccum     = 0, tickMs     = 400;
let ballTickAccum = 0, ballTickMs = 200;

function startLoop() {
  if (rafId !== null) cancelAnimationFrame(rafId);
  lastTs = performance.now();
  tickAccum = 0;
  ballTickAccum = 0;
  rafId = requestAnimationFrame(loop);
}

function loop(ts) {
  rafId = requestAnimationFrame(loop);
  const dt = Math.min(ts - lastTs, 150);
  lastTs = ts;

  if (phase === 'playing') {
    // Snake ticks (N× per ball tick)
    tickAccum += dt;
    while (tickAccum >= tickMs) {
      tick();
      tickAccum -= tickMs;
      if (phase !== 'playing') { tickAccum = 0; ballTickAccum = 0; break; }
    }

    // Ball ticks — base rate
    if (phase === 'playing') {
      ballTickAccum += dt;
      while (ballTickAccum >= ballTickMs) {
        const prevVx = ball.vx, prevVy = ball.vy;
        const scorer = stepBall(ball, s1, s2);
        ballTickAccum -= ballTickMs;
        if (scorer !== null) {
          sfxScore();
          awardPoint(scorer);
          ballTickAccum = 0;
          break;
        } else if (ball.vx !== prevVx || ball.vy !== prevVy) {
          sfxBallHit();
        }
      }
    }
  }

  draw(s1, s2, ball);
}

// ── Game logic ────────────────────────────────────────────────────────────────
function tick() {
  stepSnake(s1);
  stepSnake(s2);

  const d1 = snakeHitsDeath(s1);
  const d2 = snakeHitsDeath(s2);
  if (d1 && d2) { sfxDeath(); endRound(); return; }
  if (d1)       { sfxDeath(); awardPoint(2); return; }
  if (d2)       { sfxDeath(); awardPoint(1); return; }

  const sc = snakesCollide(s1, s2);
  if (sc === 'both') { sfxDeath(); endRound(); return; }
  if (sc === 's1')   { sfxDeath(); awardPoint(2); return; }
  if (sc === 's2')   { sfxDeath(); awardPoint(1); return; }
}

function awardPoint(player) {
  if (player === 1) score1++; else score2++;
  p1Pts.textContent = score1;
  p2Pts.textContent = score2;
  const win = parseInt(winSl.value);
  if (score1 >= win || score2 >= win) endGame(score1 >= win ? 1 : 2);
  else endRound();
}

// ── State transitions ─────────────────────────────────────────────────────────
function startGame() {
  score1 = 0; score2 = 0;
  p1Pts.textContent = 0; p2Pts.textContent = 0;
  startBgm();
  startRound();
}

function startRound() {
  // Ball is the base rate; snake ticks snakeMultiplier times per ball tick.
  ballTickMs = 1000 / getBallTps(parseInt(bSpdSl.value));
  tickMs     = ballTickMs / snakeMultiplier;
  const { s1: ns1, s2: ns2 } = createSnakes(parseInt(lenSl.value));
  s1 = ns1; s2 = ns2;
  ball = createBall();
  overlay.style.display = 'none';
  phase = 'playing';
  startLoop();
}

function endRound() {
  phase = 'roundend';
  setTimeout(() => { if (phase === 'roundend') startRound(); }, 1000);
}

function pause() {
  phase = 'paused';
  pauseBgm();
  ovTitle.textContent  = 'PAUSED';
  ovMsg.textContent    = 'Press ESC or click Resume to continue.';
  startBtn.textContent = 'RESUME';
  overlay.style.display = 'flex';
}

function resume() {
  overlay.style.display = 'none';
  phase  = 'playing';
  lastTs = performance.now();
  tickAccum = 0;
  resumeBgm();
}

function endGame(winner) {
  phase = 'gameover';
  stopBgm();
  sfxWin();
  ovTitle.textContent  = `PLAYER ${winner} WINS!`;
  ovMsg.textContent    = `Final score: ${score1} – ${score2}`;
  startBtn.textContent = 'PLAY AGAIN';
  overlay.style.display = 'flex';
}

// ── Settings modal ────────────────────────────────────────────────────────────
settingsBtn.addEventListener('click', () => settingsModal.classList.add('open'));

document.getElementById('close-settings').addEventListener('click', () => {
  settingsModal.classList.remove('open');
});

// Close on backdrop click
settingsModal.addEventListener('click', e => {
  if (e.target === settingsModal) settingsModal.classList.remove('open');
});

// Snake multiplier toggle
multGroup.addEventListener('click', e => {
  const btn = e.target.closest('.mult-btn');
  if (!btn) return;
  snakeMultiplier = parseInt(btn.dataset.mult);
  multGroup.querySelectorAll('.mult-btn').forEach(b => b.classList.toggle('active', b === btn));
});

// ── Input ─────────────────────────────────────────────────────────────────────
registerInput({
  onEscape() {
    if (settingsModal.classList.contains('open')) {
      settingsModal.classList.remove('open');
      return;
    }
    if (phase === 'playing') pause();
    else if (phase === 'paused') resume();
  },
  onDirectionP1(dx, dy) {
    if (phase !== 'playing' || !s1) return;
    if (dx !== 0 && s1.dir.x === -dx) return;
    if (dy !== 0 && s1.dir.y === -dy) return;
    s1.nextDir = { x: dx, y: dy };
  },
  onDirectionP2(dx, dy) {
    if (phase !== 'playing' || !s2) return;
    if (dx !== 0 && s2.dir.x === -dx) return;
    if (dy !== 0 && s2.dir.y === -dy) return;
    s2.nextDir = { x: dx, y: dy };
  },
});

// ── UI events ─────────────────────────────────────────────────────────────────
startBtn.addEventListener('click', () => {
  if (phase === 'paused') { resume(); return; }
  startGame();
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────
startLoop();
