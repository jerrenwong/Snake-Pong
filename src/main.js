import { createRenderer }                                from './renderer.js';
import { registerInput }                                  from './input.js';
import { createSnakes, createBall,
         stepSnake, snakeHitsDeath, snakesCollide, stepBall,
         getBallTps }                                     from './logic.js';
import { startBgm, stopBgm, pauseBgm, resumeBgm,
         sfxBallHit, sfxScore, sfxDeath, sfxWin,
         sfxPowerup }                                     from './audio.js';
import { POWERUP_DEFS, spawnPowerup,
         SPAWN_COOLDOWN_MS, SPAWN_EXPECTED_MS,
         MAX_ON_FIELD }                                   from './powerups.js';
import { WALL_L, WALL_R }                                 from './constants.js';

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
const puToggle      = document.getElementById('pu-toggle');
const puLegend      = document.getElementById('powerup-legend');

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

// ── Power-up state ────────────────────────────────────────────────────────────
let powerupsEnabled      = false;
let powerupsOnField      = [];   // [{ uid, type, x, y }]
let activeEffects        = [];   // [{ uid, type, player, expiresAt, totalDuration }]
let ballSpeedBoostPlayer = null; // 1 | 2 | null
let nextSpawnAt          = 0;    // performance.now() timestamp
let _effectUid           = 0;

function resetPowerupState() {
  powerupsOnField      = [];
  activeEffects        = [];
  ballSpeedBoostPlayer = null;
  nextSpawnAt          = performance.now() + SPAWN_COOLDOWN_MS;
}

function applyEffect(type, player) {
  const def = POWERUP_DEFS[type];

  // If this effect type is already active for this player, extend it
  const existing = activeEffects.find(e => e.type === type && e.player === player);
  if (existing) {
    existing.expiresAt = performance.now() + def.duration;
    return;
  }

  const uid = ++_effectUid;
  activeEffects.push({ uid, type, player, expiresAt: performance.now() + def.duration, totalDuration: def.duration });

  const snake = player === 1 ? s1 : s2;
  switch (type) {
    case 'length_boost':
      // Grow 5 cells by appending tail duplicates
      for (let i = 0; i < 5; i++) snake.body.push({ ...snake.body[snake.body.length - 1] });
      break;
    case 'snake_speed':
      snake.speedMult = 2;
      break;
    case 'ball_speed':
      ballSpeedBoostPlayer = player;
      break;
  }
}

function removeEffect(eff) {
  activeEffects = activeEffects.filter(e => e.uid !== eff.uid);
  const snake = eff.player === 1 ? s1 : s2;
  switch (eff.type) {
    case 'length_boost':
      // Trim 5 cells from tail (clamped to min 3)
      snake.body.splice(Math.max(3, snake.body.length - 5));
      break;
    case 'snake_speed':
      snake.speedMult = 1;
      break;
    case 'ball_speed':
      if (ballSpeedBoostPlayer === eff.player) ballSpeedBoostPlayer = null;
      break;
  }
}

function collectPowerup(pu, player) {
  powerupsOnField = powerupsOnField.filter(p => p.uid !== pu.uid);
  sfxPowerup();
  applyEffect(pu.type, player);
}

// ── Legend builder ────────────────────────────────────────────────────────────
function buildLegend() {
  puLegend.innerHTML = '';
  if (!powerupsEnabled) { puLegend.style.display = 'none'; return; }
  puLegend.style.display = 'flex';
  for (const [id, def] of Object.entries(POWERUP_DEFS)) {
    const item = document.createElement('div');
    item.className = 'pu-legend-item';
    item.innerHTML =
      `<span class="pu-badge" style="background:${def.color};color:#000">${def.label}</span>` +
      `<span class="pu-desc">${def.description}</span>`;
    puLegend.appendChild(item);
  }
}

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
    // Power-up spawn: 5s cooldown after each spawn, then geometric (E[wait]=10s)
    if (powerupsEnabled && ts >= nextSpawnAt && powerupsOnField.length < MAX_ON_FIELD) {
      if (Math.random() < dt / SPAWN_EXPECTED_MS) {
        const pu = spawnPowerup(s1, s2, powerupsOnField);
        if (pu) {
          powerupsOnField.push(pu);
          nextSpawnAt = ts + SPAWN_COOLDOWN_MS;
        }
      }
    }

    // Effect expiry
    if (powerupsEnabled) {
      const expired = activeEffects.filter(e => ts >= e.expiresAt);
      for (const e of expired) removeEffect(e);
    }

    // Snake ticks (N× per ball tick, N = snakeMultiplier × speedMult)
    tickAccum += dt;
    while (tickAccum >= tickMs) {
      tick();
      tickAccum -= tickMs;
      if (phase !== 'playing') { tickAccum = 0; ballTickAccum = 0; break; }
    }

    // Ball ticks — base rate, halved if ball-speed boost is active in correct half
    if (phase === 'playing') {
      ballTickAccum += dt;
      let effMs = ballTickMs;
      if (powerupsEnabled && ballSpeedBoostPlayer !== null) {
        const inOpponentHalf =
          (ballSpeedBoostPlayer === 1 && ball.x >= WALL_R) ||
          (ballSpeedBoostPlayer === 2 && ball.x <  WALL_L);
        if (inOpponentHalf) effMs = ballTickMs / 2;
      }
      while (ballTickAccum >= effMs) {
        const prevVx = ball.vx, prevVy = ball.vy;
        const scorer = stepBall(ball, s1, s2);
        ballTickAccum -= effMs;
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

  draw(s1, s2, ball,
    powerupsEnabled ? powerupsOnField : [],
    powerupsEnabled ? activeEffects   : []);
}

// ── Game logic ────────────────────────────────────────────────────────────────
function tick() {
  // Step each snake (speedMult times)
  for (let i = 0; i < (s1.speedMult || 1); i++) stepSnake(s1);
  for (let i = 0; i < (s2.speedMult || 1); i++) stepSnake(s2);

  // Check power-up collection
  if (powerupsEnabled) {
    for (const pu of [...powerupsOnField]) {
      const h1 = s1.body[0], h2 = s2.body[0];
      if (h1.x === pu.x && h1.y === pu.y) { collectPowerup(pu, 1); continue; }
      if (h2.x === pu.x && h2.y === pu.y) { collectPowerup(pu, 2); }
    }
  }

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
  s1 = ns1; s1.speedMult = 1;
  s2 = ns2; s2.speedMult = 1;
  ball = createBall();
  if (powerupsEnabled) resetPowerupState();
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

// Power-up toggle
puToggle.addEventListener('click', () => {
  powerupsEnabled = !powerupsEnabled;
  puToggle.classList.toggle('active', powerupsEnabled);
  puToggle.textContent = powerupsEnabled ? 'ON' : 'OFF';
  buildLegend();
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
buildLegend();
startLoop();
