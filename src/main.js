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
         FIELD_EXPIRE_MS }                                from './powerups.js';
import { WALL_L, WALL_R }                                 from './constants.js';
import { createNetwork }                                  from './network.js';
import { LocalAI }                                        from './ai_local.js';

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
// Step-1 mode buttons. `localBtn` doubles as the pause/gameover action
// button (text swaps to RESUME / PLAY AGAIN), so it's the canonical handle
// for "the big action button at the bottom of the menu overlay".
const localBtn   = document.getElementById('local-btn');
const startBtn   = localBtn;  // backwards-compatible alias
const p1Pts      = document.getElementById('p1-pts');
const p2Pts      = document.getElementById('p2-pts');
const settingsBtn   = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const multGroup     = document.getElementById('mult-group');
const puToggle      = document.getElementById('pu-toggle');
const puLegend      = document.getElementById('powerup-legend');

// Online UI refs
const mainMenu     = document.getElementById('main-menu');
const onlinePanel  = document.getElementById('online-panel');
const onlineBtn    = document.getElementById('online-btn');
const onlineBack   = document.getElementById('online-back');
const onlineStatus = document.getElementById('online-status');
const hostBtn      = document.getElementById('host-btn');
const roomCodeEl   = document.getElementById('room-code');
const joinCodeEl   = document.getElementById('join-code');
const joinBtn      = document.getElementById('join-btn');
const aiLocalBtn   = document.getElementById('ai-local-btn');
const aiVariantGroup = document.getElementById('ai-variant-group');
const aiBtn       = document.getElementById('ai-btn');
const aiPanel     = document.getElementById('ai-panel');
const aiBack      = document.getElementById('ai-back');

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
let snakeMultiplier = 2;

// ── Power-up state ────────────────────────────────────────────────────────────
let powerupsEnabled      = false;
let powerupsOnField      = [];
let activeEffects        = [];
let ballSpeedBoostPlayer = null;
let nextSpawnAt1         = 0;
let nextSpawnAt2         = 0;
let _effectUid           = 0;

// ── Online state ──────────────────────────────────────────────────────────────
let onlineRole          = null;  // null | 'host' | 'guest'
let net                 = null;
let pendingSfx          = [];    // SFX events bundled into next state send
let guestTickAccum      = 0;     // guest-local snake tick accumulator
let guestBallAccum      = 0;     // guest-local ball tick accumulator
let guestTickMs         = 400;   // snake tick interval from host
let guestBallTickMs     = 200;   // ball tick interval from host

// ── Local AI state (bypasses WebSocket; ONNX model plays P2) ──────────────────
// Model was trained with snake_length=4, snake_multiplier=2. If the UI is set
// differently we still feed the model — it tolerates length mismatch (pad/truncate)
// but will degrade if snake_multiplier doesn't match training.
let aiLocalMode      = false;    // true when "PLAY VS AI (LOCAL)" is active
let aiLocal          = null;     // LocalAI instance (lazy-init)
let aiLoading        = false;
let aiInferring      = false;    // prevent overlapping inference calls
let aiPendingDir     = null;     // latest {dx,dy} from AI — consumed on next tick
let aiVariant        = 'hard';  // default tier; updated by ai-variant-btn
let aiLoadedVariant  = null;     // which variant is currently loaded in aiLocal

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;

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

  // ── Guest: local simulation between host state updates ───────────────────
  if (onlineRole === 'guest') {
    if (phase === 'playing' && s1 && s2 && ball) {
      // Step snakes locally so they move smoothly between host updates
      guestTickAccum += dt;
      while (guestTickAccum >= guestTickMs) {
        for (let i = 0; i < (s1.speedMult || 1); i++) stepSnake(s1);
        for (let i = 0; i < (s2.speedMult || 1); i++) stepSnake(s2);
        guestTickAccum -= guestTickMs;
      }
      // Step ball locally (ignore scoring — host is authoritative)
      guestBallAccum += dt;
      while (guestBallAccum >= guestBallTickMs) {
        const scorer = stepBall(ball, s1, s2, null);
        guestBallAccum -= guestBallTickMs;
        if (scorer !== null) { guestBallAccum = 0; break; }
      }
    }
    draw(s1, s2, ball,
      powerupsEnabled ? powerupsOnField : [],
      powerupsEnabled ? activeEffects   : [],
      []);
    return;
  }

  let stateChanged = false;

  if (phase === 'playing') {
    // Power-up spawn (per-player, max 1 each, 5s cooldown + geometric E[wait]=10s)
    if (powerupsEnabled) {
      for (const [player, getTimer, setTimer] of [
        [1, () => nextSpawnAt1, v => { nextSpawnAt1 = v; }],
        [2, () => nextSpawnAt2, v => { nextSpawnAt2 = v; }],
      ]) {
        const hasOne = powerupsOnField.some(p => p.player === player);
        if (!hasOne && ts >= getTimer() && Math.random() < dt / SPAWN_EXPECTED_MS) {
          const pu = spawnPowerup(s1, s2, powerupsOnField, player, ts, null);
          if (pu) {
            powerupsOnField.push(pu);
            setTimer(ts + SPAWN_COOLDOWN_MS);
            stateChanged = true;
          }
        }
      }

      // Expire uncollected field power-ups
      const fieldExpired = powerupsOnField.filter(p => ts >= p.fieldExpiresAt);
      for (const p of fieldExpired) {
        powerupsOnField = powerupsOnField.filter(fp => fp.uid !== p.uid);
        if (p.player === 1) nextSpawnAt1 = ts + SPAWN_COOLDOWN_MS;
        else                nextSpawnAt2 = ts + SPAWN_COOLDOWN_MS;
        stateChanged = true;
      }
    }

    // Effect expiry
    if (powerupsEnabled) {
      const expired = activeEffects.filter(e => ts >= e.expiresAt);
      for (const e of expired) { removeEffect(e); stateChanged = true; }
    }

    // Snake ticks (snakeMultiplier × speedMult per ball tick)
    tickAccum += dt;
    while (tickAccum >= tickMs) {
      tick();
      tickAccum -= tickMs;
      stateChanged = true;
      if (phase !== 'playing') { tickAccum = 0; ballTickAccum = 0; break; }
    }

    // Kick off local-AI inference for P2 between ticks. Observation reflects
    // post-tick state; result is consumed by the next tick() via aiPendingDir.
    if (aiLocalMode && stateChanged) runAILocalInference();

    // Ball ticks
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
        const scorer = stepBall(ball, s1, s2, null);
        ballTickAccum -= effMs;
        stateChanged = true;
        if (scorer !== null) {
          sfxScore();
          pendingSfx.push('score');
          awardPoint(scorer);
          ballTickAccum = 0;
          break;
        } else if (ball.vx !== prevVx || ball.vy !== prevVy) {
          sfxBallHit();
          pendingSfx.push('ballHit');
        }
      }
    }
  }

  draw(s1, s2, ball,
    powerupsEnabled ? powerupsOnField : [],
    powerupsEnabled ? activeEffects   : [],
    []);

  // Send state only when game state actually changed (on ticks), not every frame.
  // This reduces network traffic from ~60 msg/sec to ~5–10 msg/sec.
  if (onlineRole === 'host' && phase === 'playing' && stateChanged) sendState();
}

// ── Local AI helpers ──────────────────────────────────────────────────────────
function applyAIDirToS2() {
  // Apply the pending AI action to s2.nextDir, respecting reversal guard.
  if (!aiLocalMode || !aiPendingDir || !s2) return;
  const { dx, dy } = aiPendingDir;
  if (dx !== 0 && s2.dir.x === -dx) return;
  if (dy !== 0 && s2.dir.y === -dy) return;
  s2.nextDir = { x: dx, y: dy };
}

async function runAILocalInference() {
  // Kick off one inference based on the CURRENT post-tick state. Result is
  // stored in aiPendingDir and applied on the next snake tick.
  if (!aiLocalMode || !aiLocal || !aiLocal.ready) return;
  if (aiInferring) return;
  if (phase !== 'playing' || !s1 || !s2 || !ball) return;
  aiInferring = true;
  try {
    const res = await aiLocal.decide(s1, s2, ball);
    if (res) aiPendingDir = { dx: res.dx, dy: res.dy };
  } catch (e) {
    console.error('[ai-local] inference error:', e);
  } finally {
    aiInferring = false;
  }
}

// ── Game logic ────────────────────────────────────────────────────────────────
function tick() {
  if (aiLocalMode) applyAIDirToS2();
  for (let i = 0; i < (s1.speedMult || 1); i++) stepSnake(s1);
  for (let i = 0; i < (s2.speedMult || 1); i++) stepSnake(s2);

  if (powerupsEnabled) {
    for (const pu of [...powerupsOnField]) {
      const h1 = s1.body[0], h2 = s2.body[0];
      if (h1.x === pu.x && h1.y === pu.y) { collectPowerup(pu, 1); continue; }
      if (h2.x === pu.x && h2.y === pu.y) { collectPowerup(pu, 2); }
    }
  }

  const d1 = snakeHitsDeath(s1, null);
  const d2 = snakeHitsDeath(s2, null);
  if (d1 && d2) { sfxDeath(); pendingSfx.push('death'); endRound(); return; }
  if (d1)       { sfxDeath(); pendingSfx.push('death'); awardPoint(2); return; }
  if (d2)       { sfxDeath(); pendingSfx.push('death'); awardPoint(1); return; }

  const sc = snakesCollide(s1, s2);
  if (sc === 'both') { sfxDeath(); pendingSfx.push('death'); endRound(); return; }
  if (sc === 's1')   { sfxDeath(); pendingSfx.push('death'); awardPoint(2); return; }
  if (sc === 's2')   { sfxDeath(); pendingSfx.push('death'); awardPoint(1); return; }
}

function awardPoint(player) {
  if (player === 1) score1++; else score2++;
  p1Pts.textContent = score1;
  p2Pts.textContent = score2;
  const win = parseInt(winSl.value);
  if (score1 >= win || score2 >= win) endGame(score1 >= win ? 1 : 2);
  else endRound();
}

// ── Power-up logic ────────────────────────────────────────────────────────────
function resetPowerupState() {
  powerupsOnField      = [];
  activeEffects        = [];
  ballSpeedBoostPlayer = null;
  const now = performance.now();
  nextSpawnAt1 = now + SPAWN_COOLDOWN_MS;
  nextSpawnAt2 = now + SPAWN_COOLDOWN_MS;
}

function applyEffect(type, player) {
  const def = POWERUP_DEFS[type];
  const existing = activeEffects.find(e => e.type === type && e.player === player);
  if (existing) { existing.expiresAt = performance.now() + def.duration; return; }
  const uid = ++_effectUid;
  activeEffects.push({ uid, type, player, expiresAt: performance.now() + def.duration, totalDuration: def.duration });
  const snake = player === 1 ? s1 : s2;
  switch (type) {
    case 'length_boost':
      for (let i = 0; i < 5; i++) snake.body.push({ ...snake.body[snake.body.length - 1] });
      break;
    case 'snake_speed': snake.speedMult = 2; break;
    case 'ball_speed':  ballSpeedBoostPlayer = player; break;
  }
}

function removeEffect(eff) {
  activeEffects = activeEffects.filter(e => e.uid !== eff.uid);
  const snake = eff.player === 1 ? s1 : s2;
  switch (eff.type) {
    case 'length_boost':
      snake.body.splice(Math.max(3, snake.body.length - 5));
      break;
    case 'snake_speed': snake.speedMult = 1; break;
    case 'ball_speed':
      if (ballSpeedBoostPlayer === eff.player) ballSpeedBoostPlayer = null;
      break;
  }
}

function collectPowerup(pu, player) {
  powerupsOnField = powerupsOnField.filter(p => p.uid !== pu.uid);
  sfxPowerup();
  pendingSfx.push('powerup');
  applyEffect(pu.type, player);
}

// ── Legend ────────────────────────────────────────────────────────────────────
function buildLegend() {
  puLegend.innerHTML = '';
  if (!powerupsEnabled) { puLegend.style.display = 'none'; return; }
  puLegend.style.display = 'flex';
  for (const [, def] of Object.entries(POWERUP_DEFS)) {
    const item = document.createElement('div');
    item.className = 'pu-legend-item';
    item.innerHTML =
      `<span class="pu-badge" style="background:${def.color};color:#000">${def.label}</span>` +
      `<span class="pu-desc">${def.description}</span>`;
    puLegend.appendChild(item);
  }
}

// ── State transitions ─────────────────────────────────────────────────────────
function startGame() {
  score1 = 0; score2 = 0;
  p1Pts.textContent = 0; p2Pts.textContent = 0;
  startBgm();
  startRound();
}

function startRound() {
  ballTickMs = 1000 / getBallTps(parseInt(bSpdSl.value));
  tickMs     = ballTickMs / snakeMultiplier;
  const { s1: ns1, s2: ns2 } = createSnakes(parseInt(lenSl.value));
  s1 = ns1; s1.speedMult = 1;
  s2 = ns2; s2.speedMult = 1;
  // Playing locally vs the AI: always serve from P1's side so the human
  // doesn't have to scramble on the opening tick.
  ball = createBall(aiLocalMode ? -1 : undefined);
  if (powerupsEnabled) resetPowerupState();
  if (aiLocalMode && aiLocal) { aiLocal.reset(); aiPendingDir = null; }
  overlay.style.display = 'none';
  phase = 'playing';
  if (onlineRole === 'host') sendState();
  startLoop();
}

function endRound() {
  phase = 'roundend';
  if (onlineRole === 'host') sendState();
  setTimeout(() => { if (phase === 'roundend') startRound(); }, 1000);
}

function _showActionOnly(text) {
  // Single-button overlay for pause / gameover: hide the LOCAL/ONLINE/AI mode
  // picker plus the AI difficulty panel, keep only the action button visible
  // with the requested label.
  localBtn.textContent = text;
  localBtn.style.display = '';
  if (aiBtn) aiBtn.style.display = 'none';
  if (onlineBtn) onlineBtn.style.display = 'none';
  if (aiPanel) aiPanel.style.display = 'none';
  mainMenu.style.display = 'flex';
}

function pause() {
  phase = 'paused';
  pauseBgm();
  ovTitle.textContent  = 'PAUSED';
  ovMsg.textContent    = 'Press ESC or click Resume to continue.';
  _showActionOnly('RESUME');
  overlay.style.display = 'flex';
  if (onlineRole === 'host') sendState();
}

function resume() {
  overlay.style.display = 'none';
  phase  = 'playing';
  lastTs = performance.now();
  tickAccum = 0;
  resumeBgm();
  if (onlineRole === 'host') sendState();
}

function endGame(winner) {
  phase = 'gameover';
  stopBgm();
  sfxWin();
  pendingSfx.push('win');
  ovTitle.textContent  = `PLAYER ${winner} WINS!`;
  ovMsg.textContent    = `Final score: ${score1} – ${score2}`;
  _showActionOnly('PLAY AGAIN');
  overlay.style.display = 'flex';
  if (onlineRole === 'host') sendState(winner);
}

// ── Online: state serialisation ───────────────────────────────────────────────
function sendState(winner = null) {
  if (!net) return;
  const sfxToSend = pendingSfx.splice(0); // drain and send
  net.send({
    type: 'relay',
    payload: {
      type:   'state',
      phase,  score1, score2, winner,
      s1:   s1   ? { body: s1.body,   dir: s1.dir,   color: s1.color,   speedMult: s1.speedMult   } : null,
      s2:   s2   ? { body: s2.body,   dir: s2.dir,   color: s2.color,   speedMult: s2.speedMult   } : null,
      ball:           ball   ? { x: ball.x, y: ball.y, vx: ball.vx, vy: ball.vy } : null,
      tickMs,
      ballTickMs,
      powerupsOnField,
      activeEffects,
      powerupsEnabled,
      sfx: sfxToSend,
    }
  });
}

function applyRemoteState(payload) {
  const prevPhase = phase;

  s1    = payload.s1;
  s2    = payload.s2;
  ball  = payload.ball;
  // Reset guest simulation accumulators — start fresh from authoritative state
  guestTickAccum = 0;
  guestBallAccum = 0;
  if (payload.tickMs)     guestTickMs     = payload.tickMs;
  if (payload.ballTickMs) guestBallTickMs = payload.ballTickMs;
  score1 = payload.score1; score2 = payload.score2;
  phase  = payload.phase;
  p1Pts.textContent = score1;
  p2Pts.textContent = score2;
  powerupsOnField  = payload.powerupsOnField || [];
  activeEffects    = payload.activeEffects   || [];
  powerupsEnabled  = payload.powerupsEnabled ?? powerupsEnabled;

  // BGM management
  if (phase === 'playing'  && prevPhase !== 'playing'  && prevPhase !== 'roundend') startBgm();
  if (phase === 'paused'   && prevPhase === 'playing')  pauseBgm();
  if (phase === 'playing'  && prevPhase === 'paused')   resumeBgm();
  if (phase === 'gameover' && prevPhase !== 'gameover') stopBgm();

  // Overlay
  if (phase === 'playing' || phase === 'roundend') {
    overlay.style.display = 'none';
  } else if (phase === 'gameover') {
    ovTitle.textContent   = `PLAYER ${payload.winner} WINS!`;
    ovMsg.textContent     = `Final score: ${score1} – ${score2}`;
    startBtn.style.display = 'none';
    overlay.style.display = 'flex';
  } else if (phase === 'paused') {
    ovTitle.textContent   = 'PAUSED';
    ovMsg.textContent     = 'Host has paused the game.';
    startBtn.style.display = 'none';
    overlay.style.display = 'flex';
  }

  // SFX
  for (const sfx of (payload.sfx || [])) {
    if (sfx === 'ballHit') sfxBallHit();
    else if (sfx === 'score')   sfxScore();
    else if (sfx === 'death')   sfxDeath();
    else if (sfx === 'win')     sfxWin();
    else if (sfx === 'powerup') sfxPowerup();
  }
}

function applyGuestInput({ dx, dy }) {
  if (phase !== 'playing' || !s2) return;
  if (dx !== 0 && s2.dir.x === -dx) return;
  if (dy !== 0 && s2.dir.y === -dy) return;
  s2.nextDir = { x: dx, y: dy };
}

// ── Online UI ─────────────────────────────────────────────────────────────────
async function connectNet() {
  if (net) return; // already connected
  const n = createNetwork(WS_URL);
  await n.connect();  // throws on failure
  net = n;
  net.on('opponent_left', () => {
    const wasPlaying = phase === 'playing' || phase === 'paused';
    onlineRole = null;
    net = null;
    if (wasPlaying) { stopBgm(); phase = 'gameover'; }
    ovTitle.textContent   = 'OPPONENT LEFT';
    ovMsg.textContent     = 'Your opponent disconnected.';
    startBtn.textContent  = 'PLAY AGAIN';
    startBtn.style.display = '';
    overlay.style.display = 'flex';
  });
  net.on('disconnect', () => { onlineRole = null; net = null; });
}

onlineBtn.addEventListener('click', () => {
  mainMenu.style.display = 'none';
  onlinePanel.style.display = 'flex';
});

onlineBack.addEventListener('click', () => {
  if (net) { net.close(); net = null; }
  onlineRole = null;
  onlinePanel.style.display = 'none';
  // Restore main-menu buttons (in case any were hidden by pause/gameover).
  localBtn.textContent = 'LOCAL';
  localBtn.style.display = '';
  aiBtn.style.display = '';
  onlineBtn.style.display = '';
  mainMenu.style.display = 'flex';
  roomCodeEl.textContent  = '';
  onlineStatus.textContent = '';
  joinCodeEl.value = '';
  hostBtn.disabled = false;
  joinBtn.disabled = false;
});

hostBtn.addEventListener('click', async () => {
  hostBtn.disabled = true;
  onlineStatus.textContent = 'Connecting…';
  try {
    await connectNet();
  } catch {
    onlineStatus.textContent = 'Cannot reach server. Is it running?';
    hostBtn.disabled = false;
    return;
  }
  // Read head selection (optional) and include in host message.
  const headSel = document.getElementById('ai-head-sel');
  const head = headSel ? parseInt(headSel.value) : undefined;
  const msg = { type: 'host' };
  if (Number.isInteger(head)) msg.head = head;
  net.send(msg);
  net.on('hosted', ({ code }) => {
    roomCodeEl.textContent   = code;
    onlineStatus.textContent = 'Share this code. Waiting for opponent…';
  });
  net.on('guest_joined', () => {
    onlineStatus.textContent = 'Opponent joined!';
    onlineRole = 'host';
    net.on('relay', ({ payload }) => {
      if (payload.type === 'input') applyGuestInput(payload);
      if (payload.type === 'pause'  && phase === 'playing') pause();
      if (payload.type === 'resume' && phase === 'paused')  resume();
    });
    // Switch back to main menu overlay and start
    onlinePanel.style.display = 'none';
    mainMenu.style.display    = 'flex';
    startBtn.style.display    = '';
    startGame();
  });
});

joinBtn.addEventListener('click', async () => {
  const code = joinCodeEl.value.trim().toUpperCase();
  if (!code) { onlineStatus.textContent = 'Enter a room code.'; return; }
  joinBtn.disabled = true;
  onlineStatus.textContent = 'Connecting…';
  try {
    await connectNet();
  } catch {
    onlineStatus.textContent = 'Cannot reach server. Is it running?';
    joinBtn.disabled = false;
    return;
  }
  net.send({ type: 'join', code });
  net.on('error', ({ reason }) => {
    onlineStatus.textContent = reason;
    joinBtn.disabled = false;
  });
  net.on('joined', () => {
    onlineRole = 'guest';
    onlinePanel.style.display = 'none';
    mainMenu.style.display    = 'flex';
    ovTitle.textContent       = 'CONNECTED';
    ovMsg.textContent         = 'Waiting for host to start…';
    startBtn.style.display    = 'none';
    overlay.style.display     = 'flex';
    startLoop();
    net.on('relay', ({ payload }) => {
      if (payload.type === 'state') applyRemoteState(payload);
    });
  });
});

// ── Settings modal ────────────────────────────────────────────────────────────
function openSettings() {
  settingsModal.classList.add('open');
  if (phase === 'playing') {
    if (onlineRole === 'guest') {
      if (net) net.send({ type: 'relay', payload: { type: 'pause' } });
    } else {
      pause();
    }
  }
}

function closeSettings() {
  settingsModal.classList.remove('open');
  if (phase === 'paused') {
    if (onlineRole === 'guest') {
      if (net) net.send({ type: 'relay', payload: { type: 'resume' } });
    } else {
      resume();
    }
  }
}

settingsBtn.addEventListener('click', openSettings);

document.getElementById('close-settings').addEventListener('click', closeSettings);

settingsModal.addEventListener('click', e => {
  if (e.target === settingsModal) closeSettings();
});

multGroup.addEventListener('click', e => {
  const btn = e.target.closest('.mult-btn');
  if (!btn) return;
  snakeMultiplier = parseInt(btn.dataset.mult);
  multGroup.querySelectorAll('.mult-btn').forEach(b => b.classList.toggle('active', b === btn));
});

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
      closeSettings();
      return;
    }
    if (onlineRole === 'guest') {
      if (phase === 'playing' && net) net.send({ type: 'relay', payload: { type: 'pause' } });
      else if (phase === 'paused' && net) net.send({ type: 'relay', payload: { type: 'resume' } });
      return;
    }
    if (phase === 'playing') pause();
    else if (phase === 'paused') resume();
  },
  onDirectionP1(dx, dy) {
    if (onlineRole === 'guest') return; // guest controls P2 only
    if (phase !== 'playing' || !s1) return;
    if (dx !== 0 && s1.dir.x === -dx) return;
    if (dy !== 0 && s1.dir.y === -dy) return;
    s1.nextDir = { x: dx, y: dy };
  },
  onDirectionP2(dx, dy) {
    if (onlineRole === 'guest') {
      if (!net) return;
      // Client-side prediction: update direction locally for immediate visual
      // feedback; host confirms on next state update.
      if (s2 && phase === 'playing') {
        const reversing = (dx !== 0 && s2.dir.x === -dx) || (dy !== 0 && s2.dir.y === -dy);
        if (!reversing) { s2.dir = { x: dx, y: dy }; s2.nextDir = { x: dx, y: dy }; }
      }
      net.send({ type: 'relay', payload: { type: 'input', dx, dy } });
      return;
    }
    if (aiLocalMode) return;  // AI controls P2
    if (phase !== 'playing' || !s2) return;
    if (dx !== 0 && s2.dir.x === -dx) return;
    if (dy !== 0 && s2.dir.y === -dy) return;
    s2.nextDir = { x: dx, y: dy };
  },
});

// ── UI events ─────────────────────────────────────────────────────────────────
localBtn.addEventListener('click', () => {
  if (onlineRole === 'guest') return;
  if (phase === 'paused') { resume(); return; }
  // From a fresh main-menu click: this is a 2-human local game, not AI.
  // From a gameover replay: keep the current mode (AI mode persists across
  // PLAY AGAIN; user can refresh to switch).
  if (phase !== 'gameover') {
    aiLocalMode = false;
  }
  startGame();
});

// AI sub-panel show/hide — mirrors the online-panel pattern.
aiBtn.addEventListener('click', () => {
  mainMenu.style.display = 'none';
  aiPanel.style.display = 'flex';
});
aiBack.addEventListener('click', () => {
  aiPanel.style.display = 'none';
  // Restore main-menu buttons in case a prior pause/gameover hid them.
  localBtn.textContent = 'LOCAL';
  localBtn.style.display = '';
  aiBtn.style.display = '';
  onlineBtn.style.display = '';
  mainMenu.style.display = 'flex';
});

// ── Local AI mode ─────────────────────────────────────────────────────────────
function lockSettingsForAi() {
  // Force the in-browser settings to match what the model was trained on so
  // performance is predictable. These are hidden from the user while AI mode
  // is active — no accidental mid-game tuning.
  lenSl.value = '4';        lenV.textContent  = '4';
  bSpdSl.value = '3';       bSpdV.textContent = '3';
  snakeMultiplier = 2;
  multGroup.querySelectorAll('.mult-btn').forEach(b =>
    b.classList.toggle('active', parseInt(b.dataset.mult) === 2));
  if (powerupsEnabled) {
    powerupsEnabled = false;
    puToggle.classList.remove('active');
    puToggle.textContent = 'OFF';
    buildLegend();
  }
  settingsBtn.style.display = 'none';
}

// Tier picker: clicking a difficulty button loads that model (if not
// already loaded) and starts a game immediately. No separate Play button.
async function _loadAndPlayVariant(variant, btn) {
  if (aiLoading) return;
  aiVariant = variant;
  // Visual: clear other actives, add 'loading' to the one we're about to load.
  aiVariantGroup.querySelectorAll('.ai-variant-btn').forEach(b => {
    b.classList.toggle('active', b === btn);
    b.classList.remove('loading');
  });
  const needReload = !aiLocal || aiLoadedVariant !== aiVariant;
  if (needReload) {
    aiLoading = true;
    btn.classList.add('loading');
    const originalText = btn.textContent;
    btn.textContent = 'LOADING…';
    try {
      aiLocal = new LocalAI();
      await aiLocal.load(
        `/models/snake-pong-${aiVariant}.onnx`,
        `/models/snake-pong-${aiVariant}.json`,
      );
      aiLoadedVariant = aiVariant;
    } catch (e) {
      console.error('[ai-local] load failed:', e);
      btn.textContent = 'LOAD FAILED';
      btn.classList.remove('loading');
      btn.title = String(e && e.message || e);
      ovMsg.textContent = `AI LOAD FAILED: ${String(e && e.message || e).slice(0, 200)}`;
      aiLoading = false;
      return;
    }
    aiLoading = false;
    btn.textContent = originalText;
    btn.classList.remove('loading');
  }
  aiLocalMode = true;
  aiPendingDir = null;
  aiLocal.reset();
  lockSettingsForAi();
  startGame();
}

aiVariantGroup.addEventListener('click', e => {
  const btn = e.target.closest('.ai-variant-btn');
  if (!btn) return;
  _loadAndPlayVariant(btn.dataset.variant, btn);
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────
buildLegend();
startLoop();
