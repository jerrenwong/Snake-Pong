import { createRenderer }                                from './renderer.js';
import { registerInput }                                  from './input.js';
import { createSnakes, createBall,
         stepSnake, snakeHitsDeath, snakesCollide, stepBall,
         getBallTps }                                     from './logic.js';
import { startBgm, stopBgm, pauseBgm, resumeBgm, setBgmStyle,
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
const celebrationContinue = document.getElementById('celebration-continue');
const celebrationBack     = document.getElementById('celebration-back');
const victoryContinue     = document.getElementById('victory-continue');
const victoryBack         = document.getElementById('victory-back');
const victoryNameInput    = document.getElementById('victory-name-input');
const victoryNameField    = document.getElementById('victory-name-field');
const victoryNameSubmit   = document.getElementById('victory-name-submit');
const victoryReplayStage  = document.getElementById('victory-replay-stage');
const victoryReplayCanvas = document.getElementById('victory-replay-canvas');
const victoryReplayCaption = document.getElementById('victory-replay-caption');
const victorySkip          = document.getElementById('victory-skip');

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
// Boss tier — unlocked the first time the player beats INSANE. State is
// persisted in localStorage so the unlock survives a reload. While
// `bossModeActive` is true the score has no cap; play continues forever.
let bossUnlocked     = (typeof localStorage !== 'undefined' &&
                        localStorage.getItem('snakepong_boss_unlocked') === '1');
// Set the first time the player reaches the win score in BOSS mode. Used
// only as a persistent badge — the victory page is shown on every win.
let bossDefeated     = (typeof localStorage !== 'undefined' &&
                        localStorage.getItem('snakepong_boss_defeated') === '1');
let bossModeActive   = false;

// Hidden INSANE shortcut: at the start of an INSANE round, if P1 walks
// straight up to the top wall and then straight left into the corner
// without any other input, BOSS unlocks immediately. `_insaneShortcut`
// holds the expected path and the head-position cursor along it.
let _insaneShortcut = null;

// Replay buffer: captured each snake tick during BOSS mode so the victory
// page can play back the full match the moment the player wins. Reset on
// every BOSS round start; played back only on a BOSS-mode win.
let _bossReplay = [];

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

// ── INSANE → BOSS shortcut ────────────────────────────────────────────────────
// Walk P1's head up to the top wall, then left into the (0, 0) corner — no
// down/right inputs after the first W, no death — and BOSS unlocks. Uses a
// 3-phase state machine (right → up → left) so it tolerates any number of
// default-right steps before the player presses W (the first tick fires
// ~145ms after spawn, well below human reaction time, so a strict
// fixed-path matcher would never trigger).
function _armInsaneShortcut() {
  _insaneShortcut = null;
  if (!aiLocalMode || aiLoadedVariant !== 'insane') return;
  // Arm on every INSANE round, not just before BOSS is unlocked. Once
  // BOSS is already unlocked the shortcut still fires the celebration —
  // it's a recognised secret, not a one-time gate.
  _insaneShortcut = { phase: 'right', alive: true, completed: false, prev: null };
}

function _stepInsaneShortcut() {
  const sc = _insaneShortcut;
  if (!sc || !sc.alive || !s1) return;
  const head = s1.body[0];
  if (sc.prev === null) {
    sc.prev = { x: head.x, y: head.y };
    return;
  }
  const dx = head.x - sc.prev.x;
  const dy = head.y - sc.prev.y;
  sc.prev = { x: head.x, y: head.y };

  if (sc.phase === 'right') {
    if (dx === 1 && dy === 0) return;                       // default-right ok
    if (dx === 0 && dy === -1) { sc.phase = 'up'; return; } // first W press
    sc.alive = false; return;
  }
  if (sc.phase === 'up') {
    if (dx === 0 && dy === -1) return;                      // keep going up
    if (dx === -1 && dy === 0 && head.y === 0) {            // pivot at top row
      sc.phase = 'left';
      if (head.x === 0) sc.completed = true;
      return;
    }
    sc.alive = false; return;
  }
  if (sc.phase === 'left') {
    if (dx === -1 && dy === 0 && head.y === 0) {
      if (head.x === 0) sc.completed = true;
      return;
    }
    sc.alive = false;
  }
}

function _triggerInsaneShortcut() {
  _insaneShortcut = null;
  // Persist on first trigger, but always show the celebration — the secret
  // is supposed to feel ceremonial every time you pull it off.
  if (!bossUnlocked) {
    bossUnlocked = true;
    try { localStorage.setItem('snakepong_boss_unlocked', '1'); } catch (e) {}
  }
  phase = 'gameover';
  stopBgm();
  sfxWin();
  pendingSfx.push('win');
  _showBossUnlockCelebration();
  if (onlineRole === 'host') sendState();
}

// ── Game logic ────────────────────────────────────────────────────────────────
function tick() {
  if (aiLocalMode) applyAIDirToS2();
  for (let i = 0; i < (s1.speedMult || 1); i++) {
    stepSnake(s1);
    _stepInsaneShortcut();
  }
  for (let i = 0; i < (s2.speedMult || 1); i++) stepSnake(s2);
  if (_insaneShortcut && _insaneShortcut.completed) {
    _triggerInsaneShortcut();
    return;
  }
  if (bossModeActive) _captureBossReplayFrame();

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
  // Boss mode: BOSS itself never "wins" by score — its score is just a
  // counter — but the player can win by reaching the win threshold first.
  // (The boss is "endless" from its side; it's the player who has to
  // outlast it.)
  const win = parseInt(winSl.value);
  if (bossModeActive) {
    if (score1 >= win) endGame(1);
    else endRound();
    return;
  }
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
  // Fresh replay buffer at the start of every match — only used for BOSS,
  // but cheap to reset universally.
  _bossReplay = [];
  // Pick the BGM style BEFORE starting the loop so the new music style takes
  // effect on the very first scheduled note instead of waiting ~6s for the
  // next 16-beat loop boundary.
  if (aiLocalMode && aiLoadedVariant === 'insane') setBgmStyle('intense');
  else if (aiLocalMode && aiLoadedVariant === 'boss') setBgmStyle('celebration');
  else setBgmStyle('normal');
  stopBgm();
  startBgm();
  startRound();
}

function startRound() {
  ballTickMs = 1000 / getBallTps(parseInt(bSpdSl.value));
  tickMs     = ballTickMs / snakeMultiplier;
  const { s1: ns1, s2: ns2 } = createSnakes(parseInt(lenSl.value),
                                            snakeMultiplier === 3);
  // Tint the AI's snake to match the difficulty-tier badge so the player
  // can tell which model they're playing at a glance.
  if (aiLocalMode && aiLoadedVariant) {
    // Saturated tones in the same family as the default P1 #2af / P2 #f62.
    const tierColors = {
      easy:   '#2c2',  // deep green
      medium: '#cc0',  // deep yellow
      hard:   '#f72',  // deep orange (close to original)
      master: '#c3c',  // deep magenta
      insane: '#e22',  // deep red
      boss:   '#fb0',  // deep gold
    };
    if (tierColors[aiLoadedVariant]) ns2.color = tierColors[aiLoadedVariant];
    // Animated effects mirroring the tier button — only top tiers get them.
    if (aiLoadedVariant === 'insane') ns2.effect = 'insane-glow';
    else if (aiLoadedVariant === 'boss') ns2.effect = 'boss-shimmer';
    // Tint the P2 scoreboard so it matches the AI's snake.
    if (tierColors[aiLoadedVariant]) p2Pts.style.color = tierColors[aiLoadedVariant];
  } else {
    // Reset P2 scoreboard outside AI mode.
    p2Pts.style.color = '';
  }
  s1 = ns1; s1.speedMult = 1;
  s2 = ns2; s2.speedMult = 1;
  _armInsaneShortcut();
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
  // Special case: human (P1) just beat INSANE in local-AI mode → unlock BOSS
  // and show a celebration overlay before the regular gameover screen.
  if (winner === 1 && aiLocalMode && aiLoadedVariant === 'insane' && !bossUnlocked) {
    bossUnlocked = true;
    try { localStorage.setItem('snakepong_boss_unlocked', '1'); } catch (e) {}
    _showBossUnlockCelebration();
    if (onlineRole === 'host') sendState(winner);
    return;
  }
  // Special case: P1 beat the BOSS → iridescent celebration sequence with
  // hero-name input and a replay of the match.
  if (winner === 1 && aiLocalMode && bossModeActive) {
    if (!bossDefeated) {
      bossDefeated = true;
      try { localStorage.setItem('snakepong_boss_defeated', '1'); } catch (e) {}
    }
    _showBossVictoryCelebration();
    if (onlineRole === 'host') sendState(winner);
    return;
  }
  ovTitle.textContent  = `PLAYER ${winner} WINS!`;
  ovMsg.textContent    = `Final score: ${score1} – ${score2}`;
  _showActionOnly('PLAY AGAIN');
  overlay.style.display = 'flex';
  if (onlineRole === 'host') sendState(winner);
}

// Sentence-by-sentence reveal of the boss-unlock celebration. Each sentence
// fades in (0.9s), holds, then fades out and is replaced by the next. The
// final sentence stays on screen and reveals the action buttons below.
const _CELEBRATION_SENTENCES = [
  'CONGRATULATIONS',
  'NO HAND THAT SHAPED THIS WORLD<br>HAS WALKED THIS FAR.',
  'A HIDDEN MONSTER<br>HAS BEEN UNLEASHED.',
  'FEARLESS &nbsp; RESTLESS &nbsp; ENDLESS',
  'AT 3 TIMES SPEED.',
];
const _CELEBRATION_HOLD_MS  = 4000;   // how long each sentence stays at full opacity
const _CELEBRATION_FADE_MS  = 900;    // fade in / out duration (matches CSS transition)
let _celebrationTimeouts = [];

function _showBossUnlockCelebration() {
  const cel = document.getElementById('boss-celebration');
  const msg = document.getElementById('celebration-message');
  const actions = document.getElementById('celebration-actions');
  if (!cel || !msg || !actions) return;

  _celebrationTimeouts.forEach(clearTimeout);
  _celebrationTimeouts = [];
  msg.style.opacity = 0;
  actions.classList.remove('visible');
  // Hide every other overlay child so only the celebration is visible —
  // the overlay was display:none during gameplay, so without this the
  // celebration would render under a hidden parent and the screen
  // would just freeze on the last game frame.
  if (mainMenu) mainMenu.style.display = 'none';
  if (aiPanel) aiPanel.style.display = 'none';
  if (onlinePanel) onlinePanel.style.display = 'none';
  cel.style.display = 'flex';
  overlay.style.display = 'flex';

  function showSentence(idx) {
    if (idx >= _CELEBRATION_SENTENCES.length) {
      // After the last sentence has fully faded, present the FIGHT-THE-BOSS
      // button as a fresh page (centred, message slot empty).
      msg.innerHTML = '&nbsp;';
      _celebrationTimeouts.push(setTimeout(() => {
        actions.classList.add('visible');
      }, _CELEBRATION_FADE_MS));
      return;
    }
    msg.innerHTML = _CELEBRATION_SENTENCES[idx];
    requestAnimationFrame(() => { msg.style.opacity = 1; });
    _celebrationTimeouts.push(setTimeout(() => {
      msg.style.opacity = 0;
      _celebrationTimeouts.push(setTimeout(() => showSentence(idx + 1),
                                            _CELEBRATION_FADE_MS));
    }, _CELEBRATION_HOLD_MS));
  }
  showSentence(0);
}

function _hideBossUnlockCelebration() {
  _celebrationTimeouts.forEach(clearTimeout);
  _celebrationTimeouts = [];
  const cel = document.getElementById('boss-celebration');
  const actions = document.getElementById('celebration-actions');
  if (actions) actions.classList.remove('visible');
  if (cel) cel.style.display = 'none';
}

// ── BOSS-defeated celebration ─────────────────────────────────────────────────
// Three-stage iridescent sequence + canvas replay of the match. Triggered
// from endGame() once when P1 reaches the win threshold against BOSS.

const HERO_NAME_KEY = 'snakepong_hero_name';
const _VICTORY_HOLD_MS  = 3500;   // sentence dwell time at full opacity
const _VICTORY_FADE_MS  = 900;    // matches CSS transition
const _REPLAY_FRAME_MS  = 45;     // ~22 fps playback (snappy but readable)
let _victoryTimeouts = [];
let _victoryReplayRenderer = null;  // lazy-init cached renderer for the replay canvas
let _heroName = '';

function _captureBossReplayFrame() {
  if (!s1 || !s2 || !ball) return;
  // Deep copy positions only — colors/effects come from live state during
  // playback. Cap buffer at ~3000 frames (~5 min of boss play) so a stuck
  // game doesn't balloon memory.
  if (_bossReplay.length >= 3000) return;
  _bossReplay.push({
    s1: { body: s1.body.map(p => ({ x: p.x, y: p.y })),
          dir:  { x: s1.dir.x, y: s1.dir.y }, color: s1.color, effect: s1.effect },
    s2: { body: s2.body.map(p => ({ x: p.x, y: p.y })),
          dir:  { x: s2.dir.x, y: s2.dir.y }, color: s2.color, effect: s2.effect },
    ball: { x: ball.x, y: ball.y, vx: ball.vx, vy: ball.vy },
    score1, score2,
  });
}

function _showBossVictoryCelebration() {
  const cel = document.getElementById('boss-victory');
  const msg = victoryReplayCaption ? document.getElementById('victory-message') : null;
  if (!cel || !msg) return;

  _victoryTimeouts.forEach(clearTimeout);
  _victoryTimeouts = [];

  // Hide every other overlay child so only the celebration is visible.
  if (mainMenu) mainMenu.style.display = 'none';
  if (aiPanel) aiPanel.style.display = 'none';
  if (onlinePanel) onlinePanel.style.display = 'none';
  const bossCelEl = document.getElementById('boss-celebration');
  if (bossCelEl) bossCelEl.style.display = 'none';
  cel.style.display = 'flex';
  overlay.style.display = 'flex';

  // Reset all stages.
  msg.style.opacity = 0;
  msg.style.display = 'block';
  if (victoryNameInput)   victoryNameInput.style.display   = 'none';
  if (victoryReplayStage) victoryReplayStage.style.display = 'none';
  if (victoryContinue)    victoryContinue.parentElement.classList.remove('visible');

  // Restore last-used hero name so the field isn't blank if they replay.
  try { _heroName = localStorage.getItem(HERO_NAME_KEY) || ''; } catch (e) { _heroName = ''; }
  if (victoryNameField) victoryNameField.value = _heroName;

  // Stage i — "THE UNBEATABLE HAS FALLEN".
  msg.innerHTML = 'THE UNBEATABLE<br>HAS FALLEN.';
  requestAnimationFrame(() => { msg.style.opacity = 1; });
  _victoryTimeouts.push(setTimeout(() => {
    msg.style.opacity = 0;
    _victoryTimeouts.push(setTimeout(_victoryStageNameInput, _VICTORY_FADE_MS));
  }, _VICTORY_HOLD_MS));
}

function _victoryStageNameInput() {
  // Stage ii — hero-name input. Hides the sentence, shows the input block.
  const msg = document.getElementById('victory-message');
  if (msg) msg.style.display = 'none';
  if (!victoryNameInput) { _victoryStageEternalStory(); return; }
  victoryNameInput.style.display = 'flex';
  if (victoryNameField) {
    setTimeout(() => victoryNameField.focus(), 50);
  }
}

function _victoryStageEternalStory() {
  // Stage iii — "AND THEIR ETERNAL STORY…" then replay.
  if (victoryNameInput) victoryNameInput.style.display = 'none';
  const msg = document.getElementById('victory-message');
  if (!msg) return;
  msg.style.display = 'block';
  msg.style.opacity = 0;
  const heroLine = _heroName
    ? `AND ${_heroName.toUpperCase()}'S<br>ETERNAL STORY…`
    : 'AND THEIR<br>ETERNAL STORY…';
  msg.innerHTML = heroLine;
  requestAnimationFrame(() => { msg.style.opacity = 1; });
  _victoryTimeouts.push(setTimeout(() => {
    msg.style.opacity = 0;
    _victoryTimeouts.push(setTimeout(_victoryStageReplay, _VICTORY_FADE_MS));
  }, _VICTORY_HOLD_MS));
}

function _victoryStageReplay() {
  // Stage iv — render the captured frames on the small canvas.
  const msg = document.getElementById('victory-message');
  if (msg) msg.style.display = 'none';
  if (!victoryReplayStage || !victoryReplayCanvas) {
    _victoryStageActions();
    return;
  }
  victoryReplayStage.style.display = 'flex';
  if (victoryReplayCaption) {
    victoryReplayCaption.textContent = _heroName
      ? `${_heroName.toUpperCase()}  ·  ${score1} – ${score2}`
      : `THE HERO  ·  ${score1} – ${score2}`;
  }
  if (!_victoryReplayRenderer) {
    _victoryReplayRenderer = createRenderer(victoryReplayCanvas);
  }
  const frames = _bossReplay;
  if (!frames.length) {
    // No frames captured — skip directly to the action buttons.
    _victoryStageActions();
    return;
  }
  let i = 0;
  const tick = () => {
    if (i >= frames.length) {
      _victoryTimeouts.push(setTimeout(_victoryStageActions, 600));
      return;
    }
    const f = frames[i++];
    _victoryReplayRenderer.draw(f.s1, f.s2, f.ball, [], [], []);
    _victoryTimeouts.push(setTimeout(tick, _REPLAY_FRAME_MS));
  };
  tick();
}

function _victoryStageActions() {
  if (victoryContinue && victoryContinue.parentElement) {
    victoryContinue.parentElement.classList.add('visible');
  }
}

// Skip everything: cancel pending fades/replay timers and jump straight to
// the final action buttons, hiding the per-stage UI in between.
function _skipBossVictoryToEnd() {
  _victoryTimeouts.forEach(clearTimeout);
  _victoryTimeouts = [];
  const msg = document.getElementById('victory-message');
  if (msg) {
    msg.style.display = 'none';
    msg.style.opacity = 0;
  }
  if (victoryNameInput)   victoryNameInput.style.display = 'none';
  if (victoryReplayStage) victoryReplayStage.style.display = 'none';
  // Persist whatever the user typed before skipping, if anything.
  if (victoryNameField && victoryNameField.value.trim()) {
    _heroName = victoryNameField.value.trim().slice(0, 32);
    try { localStorage.setItem(HERO_NAME_KEY, _heroName); } catch (e) {}
  }
  _victoryStageActions();
}

function _hideBossVictoryCelebration() {
  _victoryTimeouts.forEach(clearTimeout);
  _victoryTimeouts = [];
  const cel = document.getElementById('boss-victory');
  if (cel) cel.style.display = 'none';
  if (victoryReplayStage) victoryReplayStage.style.display = 'none';
  if (victoryNameInput)   victoryNameInput.style.display   = 'none';
  if (victoryContinue && victoryContinue.parentElement) {
    victoryContinue.parentElement.classList.remove('visible');
  }
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
  // performance is predictable. BOSS uses mult=3 (its training config); all
  // other tiers use mult=2.
  const targetMult = bossModeActive ? 3 : 2;
  lenSl.value = '4';        lenV.textContent  = '4';
  bSpdSl.value = '3';       bSpdV.textContent = '3';
  snakeMultiplier = targetMult;
  multGroup.querySelectorAll('.mult-btn').forEach(b =>
    b.classList.toggle('active', parseInt(b.dataset.mult) === targetMult));
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
  // `btn` may be null (e.g. when launched directly from the boss-celebration
  // "FACE THE BOSS" button, which isn't part of the variant group).
  aiVariantGroup.querySelectorAll('.ai-variant-btn').forEach(b => {
    b.classList.toggle('active', btn !== null && b === btn);
    b.classList.remove('loading');
  });
  const needReload = !aiLocal || aiLoadedVariant !== aiVariant;
  let originalText = '';
  if (needReload) {
    aiLoading = true;
    if (btn) {
      btn.classList.add('loading');
      originalText = btn.textContent;
      btn.textContent = 'LOADING…';
    }
    try {
      aiLocal = new LocalAI();
      await aiLocal.load(
        `models/snake-pong-${aiVariant}.onnx`,
        `models/snake-pong-${aiVariant}.json`,
        aiVariant,
      );
      aiLoadedVariant = aiVariant;
    } catch (e) {
      console.error('[ai-local] load failed:', e);
      if (btn) {
        btn.textContent = 'LOAD FAILED';
        btn.classList.remove('loading');
        btn.title = String(e && e.message || e);
      }
      ovMsg.textContent = `AI LOAD FAILED: ${String(e && e.message || e).slice(0, 200)}`;
      aiLoading = false;
      return;
    }
    aiLoading = false;
    if (btn) {
      btn.textContent = originalText;
      btn.classList.remove('loading');
    }
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
  // BOSS = mult=3, no win cap. Other tiers = normal mult=2 / standard win.
  bossModeActive = btn.dataset.variant === 'boss';
  _loadAndPlayVariant(btn.dataset.variant, btn);
});

// Celebration overlay buttons. "FIGHT THE BOSS" launches the BOSS game
// directly — boss is never offered in the regular tier picker.
if (celebrationContinue) {
  celebrationContinue.addEventListener('click', () => {
    _hideBossUnlockCelebration();
    bossModeActive = true;
    _loadAndPlayVariant('boss', null);
  });
}
if (celebrationBack) {
  celebrationBack.addEventListener('click', () => {
    _hideBossUnlockCelebration();
    // Restore the regular main menu.
    aiPanel.style.display = 'none';
    localBtn.textContent = 'LOCAL';
    localBtn.style.display = '';
    aiBtn.style.display = '';
    onlineBtn.style.display = '';
    mainMenu.style.display = 'flex';
  });
}

// BOSS-defeated celebration buttons.
function _commitHeroNameAndAdvance() {
  if (victoryNameField) {
    _heroName = victoryNameField.value.trim().slice(0, 32);
    if (_heroName) {
      try { localStorage.setItem(HERO_NAME_KEY, _heroName); } catch (e) {}
    }
  }
  _victoryStageEternalStory();
}
if (victoryNameSubmit) {
  victoryNameSubmit.addEventListener('click', _commitHeroNameAndAdvance);
}
if (victorySkip) {
  victorySkip.addEventListener('click', _skipBossVictoryToEnd);
}
if (victoryNameField) {
  victoryNameField.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); _commitHeroNameAndAdvance(); }
  });
}
if (victoryContinue) {
  victoryContinue.addEventListener('click', () => {
    _hideBossVictoryCelebration();
    bossModeActive = true;
    _loadAndPlayVariant('boss', null);
  });
}
if (victoryBack) {
  victoryBack.addEventListener('click', () => {
    _hideBossVictoryCelebration();
    bossModeActive = false;
    aiPanel.style.display = 'none';
    localBtn.textContent = 'LOCAL';
    localBtn.style.display = '';
    aiBtn.style.display = '';
    if (onlineBtn) onlineBtn.style.display = '';
    mainMenu.style.display = 'flex';
  });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
buildLegend();
startLoop();
