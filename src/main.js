import { createRenderer }                                from './renderer.js';
import { registerInput }                                  from './input.js';
import { createSnakes, createBall,
         stepSnake, snakeHitsDeath, snakesCollide, stepBall,
         getBallTps }                                     from './logic.js';
import { startBgm, stopBgm, pauseBgm, resumeBgm, setBgmStyle,
         sfxBallHit, sfxScore, sfxDeath, sfxWin,
         sfxPowerup, getAudioStream }                     from './audio.js';
import { POWERUP_DEFS, spawnPowerup,
         SPAWN_COOLDOWN_MS, SPAWN_EXPECTED_MS,
         FIELD_EXPIRE_MS }                                from './powerups.js';
import { WALL_L, WALL_R }                                 from './constants.js';
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
const p1Pts      = document.getElementById('p1-pts');
const p2Pts      = document.getElementById('p2-pts');
const settingsBtn   = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const multGroup     = document.getElementById('mult-group');
const puToggle      = document.getElementById('pu-toggle');
const puLegend      = document.getElementById('powerup-legend');

const mainMenu     = document.getElementById('main-menu');
const aiLocalBtn   = document.getElementById('ai-local-btn');
const aiVariantGroup = document.getElementById('ai-variant-group');
const aiBtn       = document.getElementById('ai-btn');
const aiPanel     = document.getElementById('ai-panel');
const aiBack      = document.getElementById('ai-back');
const celebrationContinue = document.getElementById('celebration-continue');
const celebrationBack     = document.getElementById('celebration-back');
const victoryActions      = document.getElementById('victory-actions');
const victoryBack         = document.getElementById('victory-back');
const victoryNameInput    = document.getElementById('victory-name-input');
const victoryNameField    = document.getElementById('victory-name-field');
const victoryNameSubmit   = document.getElementById('victory-name-submit');
const victoryReplayStage  = document.getElementById('victory-replay-stage');
const victoryReplayCanvas = document.getElementById('victory-replay-canvas');
const victoryReplayCaption = document.getElementById('victory-replay-caption');
const victoryReplayDownload = document.getElementById('victory-replay-download');

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

// ── Local AI state (ONNX model plays P2) ──────────────────────────────────────
let aiLocalMode      = false;
let aiLocal          = null;
let aiLoading        = false;
let aiInferring      = false;
let aiPendingDir     = null;
let aiVariant        = 'hard';
let aiLoadedVariant  = null;
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
    powerupsEnabled ? activeEffects   : [],
    []);
}

// ── Local AI helpers ──────────────────────────────────────────────────────────
function applyAIDirToS2() {
  if (!aiLocalMode || !aiPendingDir || !s2) return;
  const { dx, dy } = aiPendingDir;
  if (dx !== 0 && s2.dir.x === -dx) return;
  if (dy !== 0 && s2.dir.y === -dy) return;
  s2.nextDir = { x: dx, y: dy };
}

async function runAILocalInference() {
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
  _showBossUnlockCelebration();
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
  // Boss mode: P1 wins by reaching the win threshold; BOSS never wins by
  // score because it's endless from its side.
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
  // effect on the very first scheduled note.
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
    const tierColors = {
      easy:   '#2c2',
      medium: '#cc0',
      hard:   '#f72',
      master: '#c3c',
      insane: '#e22',
      boss:   '#fb0',
    };
    if (tierColors[aiLoadedVariant]) ns2.color = tierColors[aiLoadedVariant];
    if (aiLoadedVariant === 'insane') ns2.effect = 'insane-glow';
    else if (aiLoadedVariant === 'boss') ns2.effect = 'boss-shimmer';
    if (tierColors[aiLoadedVariant]) p2Pts.style.color = tierColors[aiLoadedVariant];
  } else {
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
  startLoop();
}

function endRound() {
  phase = 'roundend';
  setTimeout(() => { if (phase === 'roundend') startRound(); }, 1000);
}

function _showActionOnly(text) {
  // Single-button overlay for pause / gameover: hide the LOCAL/AI mode picker
  // plus the AI difficulty panel, keep only the action button visible.
  localBtn.textContent = text;
  localBtn.style.display = '';
  if (aiBtn) aiBtn.style.display = 'none';
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
  // Special case: human (P1) just beat INSANE in local-AI mode → unlock BOSS
  // and show a celebration overlay before the regular gameover screen.
  if (winner === 1 && aiLocalMode && aiLoadedVariant === 'insane' && !bossUnlocked) {
    bossUnlocked = true;
    try { localStorage.setItem('snakepong_boss_unlocked', '1'); } catch (e) {}
    _showBossUnlockCelebration();
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
    return;
  }
  ovTitle.textContent  = `PLAYER ${winner} WINS!`;
  ovMsg.textContent    = `Final score: ${score1} – ${score2}`;
  _showActionOnly('PLAY AGAIN');
  overlay.style.display = 'flex';
}

// Sentence-by-sentence reveal of the boss-unlock celebration. Each sentence
// fades in (0.9s), holds, then fades out and is replaced by the next. The
// final sentence stays on screen and reveals the action buttons below.
const _CELEBRATION_SENTENCES = [
  'CONGRATULATIONS',
  'NO HAND THAT SHAPED THIS WORLD<br>HAS WALKED THIS FAR',
  'A HIDDEN MONSTER<br>HAS BEEN UNLEASHED',
  'FEARLESS, RESTLESS, ENDLESS',
  'AT 3 TIMES SPEED',
];
const _CELEBRATION_HOLD_MS  = 4000;
const _CELEBRATION_FADE_MS  = 900;
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
  cel.style.display = 'flex';
  overlay.style.display = 'flex';

  function showSentence(idx) {
    if (idx >= _CELEBRATION_SENTENCES.length) {
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
const _RECORD_W = 3840;           // composite-recording canvas size — 4K UHD.
const _RECORD_H = 2160;           // (Composite chrome scales off W/720.)
let _victoryTimeouts = [];
let _victoryReplayRenderer = null;  // lazy-init cached renderer for the replay canvas
let _recordCanvas = null;           // offscreen canvas drawn each frame for the
                                    // downloadable video — has the gold frame
                                    // and hero name baked in.
let _mp4MuxerModulePromise = null;  // cached dynamic-import promise for mp4-muxer

// mp4-muxer ESM, loaded on demand the first time a player reaches the
// boss-victory replay. Cached so subsequent victories don't re-fetch.
// Tries several CDNs in order — esm.sh and jsdelivr have intermittently
// failed for the same package on different days.
function _loadMp4Muxer() {
  if (_mp4MuxerModulePromise) return _mp4MuxerModulePromise;
  const urls = [
    'https://esm.sh/mp4-muxer@5',
    'https://cdn.jsdelivr.net/npm/mp4-muxer@5/+esm',
    'https://cdn.jsdelivr.net/npm/mp4-muxer@5.1.4/build/mp4-muxer.mjs',
    'https://unpkg.com/mp4-muxer@5/build/mp4-muxer.mjs',
  ];
  _mp4MuxerModulePromise = (async () => {
    for (const url of urls) {
      try {
        const mod = await import(url);
        if (mod && (mod.Muxer || (mod.default && mod.default.Muxer))) {
          console.info('[boss-victory] mp4-muxer loaded from', url);
          return mod;
        }
      } catch (e) {
        console.warn('[boss-victory] mp4-muxer load failed for', url, e);
      }
    }
    return null;
  })();
  return _mp4MuxerModulePromise;
}

// Set up an MP4 encoder backed by WebCodecs + mp4-muxer. Returns a small
// controller `{ encodeFrame(ts), finalize() -> Promise<Blob> }`, or null if
// WebCodecs / the muxer aren't available — caller falls back to WebM.
async function _setupMp4Recorder(canvas, audioStream) {
  if (typeof VideoEncoder === 'undefined' || typeof VideoFrame === 'undefined') return null;
  const muxerMod = await _loadMp4Muxer();
  if (!muxerMod) return null;
  const Muxer = muxerMod.Muxer || (muxerMod.default && muxerMod.default.Muxer);
  const ArrayBufferTarget = muxerMod.ArrayBufferTarget
    || (muxerMod.default && muxerMod.default.ArrayBufferTarget);
  if (!Muxer || !ArrayBufferTarget) return null;

  // Match the audio config to the actual track (sample rate / channel count
  // mismatches are a common silent-fail mode for AudioEncoder.encode).
  const audioTrack = audioStream && audioStream.getAudioTracks && audioStream.getAudioTracks()[0];
  const trackSettings = audioTrack && audioTrack.getSettings ? audioTrack.getSettings() : {};
  const sampleRate  = trackSettings.sampleRate  || 48000;
  const numChannels = trackSettings.channelCount || 2;

  let mp4Failed = false;
  const fail = (where, e) => { console.warn('[boss-victory] mp4', where, 'failed:', e); mp4Failed = true; };

  let muxer;
  try {
    muxer = new Muxer({
      target: new ArrayBufferTarget(),
      video: { codec: 'avc', width: _RECORD_W, height: _RECORD_H },
      audio: { codec: 'aac', numberOfChannels: numChannels, sampleRate },
      fastStart: 'in-memory',
      firstTimestampBehavior: 'offset',
    });
  } catch (e) { fail('muxer-init', e); return null; }

  let videoEncoder;
  try {
    videoEncoder = new VideoEncoder({
      output: (chunk, meta) => {
        try { muxer.addVideoChunk(chunk, meta); } catch (e) { fail('addVideoChunk', e); }
      },
      error: (e) => fail('video-encoder', e),
    });
    videoEncoder.configure({
      // Main profile level 5.1 — supports up to 4096×2304, so 4K UHD
      // (3840×2160) fits. Baseline tops out at 4.2 (1920×1080) and
      // wouldn't carry this resolution. If the platform encoder doesn't
      // accept 5.1 we'll fall through to the MediaRecorder WebM path.
      codec: 'avc1.4D0033',
      width: _RECORD_W,
      height: _RECORD_H,
      bitrate: 20_000_000,        // ~20 Mbps — appropriate for 4K @ 22 fps
      framerate: 22,
      latencyMode: 'realtime',
    });
  } catch (e) { fail('video-configure', e); return null; }

  let audioEncoder = null;
  let audioReader = null;
  let audioAborted = false;
  if (audioTrack && typeof AudioEncoder !== 'undefined' && typeof MediaStreamTrackProcessor !== 'undefined') {
    try {
      audioEncoder = new AudioEncoder({
        output: (chunk, meta) => {
          try { muxer.addAudioChunk(chunk, meta); } catch (e) { fail('addAudioChunk', e); }
        },
        error: (e) => console.warn('[boss-victory] audio encode err:', e),
      });
      audioEncoder.configure({
        codec: 'mp4a.40.2',
        numberOfChannels: numChannels,
        sampleRate,
        bitrate: 128_000,
      });
      const proc = new MediaStreamTrackProcessor({ track: audioTrack });
      audioReader = proc.readable.getReader();
      (async () => {
        while (!audioAborted) {
          let res;
          try { res = await audioReader.read(); } catch (e) { break; }
          if (res.done) break;
          try { audioEncoder.encode(res.value); } catch (e) { /* ignore */ }
          res.value.close();
        }
      })();
    } catch (e) {
      console.warn('[boss-victory] audio configure failed (continuing without audio):', e);
      audioEncoder = null;
      audioReader = null;
    }
  }

  let frameCount = 0;
  return {
    encodeFrame(timestampUs) {
      if (mp4Failed) return false;
      let frame;
      try { frame = new VideoFrame(canvas, { timestamp: timestampUs }); }
      catch (e) { fail('VideoFrame', e); return false; }
      const isKey = frameCount % 22 === 0;
      try { videoEncoder.encode(frame, { keyFrame: isKey }); }
      catch (e) { fail('encode', e); }
      frame.close();
      frameCount++;
      return !mp4Failed;
    },
    failed() { return mp4Failed; },
    async finalize() {
      audioAborted = true;
      if (audioReader) { try { await audioReader.cancel(); } catch (e) {} }
      try { await videoEncoder.flush(); } catch (e) { fail('video-flush', e); }
      if (audioEncoder) { try { await audioEncoder.flush(); } catch (e) {} }
      if (mp4Failed) throw new Error('mp4 encoding failed');
      try { muxer.finalize(); }
      catch (e) { fail('muxer-finalize', e); throw e; }
      return new Blob([muxer.target.buffer], { type: 'video/mp4' });
    },
  };
}
let _heroName = '';

// Render one composite frame (background + gold ember frame + hero name +
// the live game canvas, scaled into the inner area) onto _recordCanvas.
// Called every replay tick alongside the on-screen draw.
function _drawCompositeRecordFrame() {
  if (!_recordCanvas || !victoryReplayCanvas) return;
  const ctx = _recordCanvas.getContext('2d');
  const W = _RECORD_W, H = _RECORD_H;
  // All chrome dimensions are tuned at the original 720×540 design and
  // scaled up. Bumping _RECORD_W/H proportionally keeps the look intact.
  const S = W / 720;

  // Background.
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, W, H);

  // Radial ember glow behind the frame, mirroring the CSS aurora-frame.
  const grad = ctx.createRadialGradient(W / 2, H / 2, 40 * S, W / 2, H / 2, W * 0.6);
  grad.addColorStop(0,   'rgba(40, 20, 0, 0.55)');
  grad.addColorStop(0.7, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Gold ember-style border.
  ctx.save();
  ctx.shadowColor = 'rgba(255, 200, 100, 0.55)';
  ctx.shadowBlur  = 18 * S;
  ctx.strokeStyle = 'rgba(255, 200, 100, 0.55)';
  ctx.lineWidth   = 1.5 * S;
  ctx.strokeRect(20 * S, 20 * S, W - 40 * S, H - 40 * S);
  ctx.restore();

  // Hero-name caption near the top of the frame.
  ctx.save();
  const heroText = (_heroName || 'THE HERO').toUpperCase();
  ctx.fillStyle    = '#ffd47a';
  ctx.font         = `600 ${Math.round(17 * S)}px "Courier New", monospace`;
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  ctx.shadowColor  = 'rgba(255, 200, 100, 0.7)';
  ctx.shadowBlur   = 10 * S;
  // Fake letter-spacing by drawing each char with a wider advance.
  const tracked = heroText.split('').join(' ');
  ctx.fillText(tracked, W / 2, 60 * S);
  ctx.restore();

  // Inner game viewport, preserving the engine's 936:676 aspect ratio.
  const margin   = 70  * S;
  const topGap   = 100 * S;
  const bottomGap = 60 * S;
  const maxW = W - 2 * margin;
  const maxH = H - topGap - bottomGap;
  const ratio = 936 / 676;
  let gW = maxW, gH = gW / ratio;
  if (gH > maxH) { gH = maxH; gW = gH * ratio; }
  const gX = (W - gW) / 2;
  const gY = topGap + (maxH - gH) / 2;

  // Subtle inner gold outline around the game.
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 200, 100, 0.35)';
  ctx.lineWidth   = 1 * S;
  ctx.strokeRect(gX - S, gY - S, gW + 2 * S, gH + 2 * S);
  ctx.restore();

  // The live game canvas was drawn this tick by _victoryReplayRenderer.
  ctx.drawImage(victoryReplayCanvas, gX, gY, gW, gH);
}

function _captureBossReplayFrame() {
  if (!s1 || !s2 || !ball) return;
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
  const msg = document.getElementById('victory-message');
  if (!cel || !msg) return;

  // Pre-fetch mp4-muxer in the background while the player reads the
  // sentence reveal — by the time we reach the replay stage, the import
  // is cached and recorder setup is essentially instant.
  _loadMp4Muxer();

  _victoryTimeouts.forEach(clearTimeout);
  _victoryTimeouts = [];

  if (mainMenu) mainMenu.style.display = 'none';
  if (aiPanel) aiPanel.style.display = 'none';
  const bossCelEl = document.getElementById('boss-celebration');
  if (bossCelEl) bossCelEl.style.display = 'none';
  cel.style.display = 'flex';
  overlay.style.display = 'flex';

  msg.style.opacity = 0;
  msg.style.display = 'block';
  if (victoryNameInput)   victoryNameInput.style.display   = 'none';
  if (victoryReplayStage) victoryReplayStage.style.display = 'none';
  if (victoryActions)     victoryActions.classList.remove('visible');

  try { _heroName = localStorage.getItem(HERO_NAME_KEY) || ''; } catch (e) { _heroName = ''; }
  if (victoryNameField) victoryNameField.value = _heroName;

  msg.innerHTML = 'THE UNBEATABLE HAS FALLEN';
  requestAnimationFrame(() => { msg.style.opacity = 1; });
  _victoryTimeouts.push(setTimeout(() => {
    msg.style.opacity = 0;
    _victoryTimeouts.push(setTimeout(_victoryStageNameInput, _VICTORY_FADE_MS));
  }, _VICTORY_HOLD_MS));
}

function _victoryStageNameInput() {
  const msg = document.getElementById('victory-message');
  if (msg) msg.style.display = 'none';
  if (!victoryNameInput) { _victoryStageEternalStory(); return; }
  victoryNameInput.style.display = 'flex';
  if (victoryNameField) setTimeout(() => victoryNameField.focus(), 50);
}

function _victoryStageEternalStory() {
  // Stage iii — "We shall remember your story" then the replay video.
  if (victoryNameInput) victoryNameInput.style.display = 'none';
  const msg = document.getElementById('victory-message');
  if (!msg) return;
  msg.style.display = 'block';
  msg.style.opacity = 0;
  // Don't include the hero's name on this page — the name lives on the
  // replay caption, this stage is the universal salute.
  msg.innerHTML = 'AND YOUR JOURNEY SHALL BE REMEMBERED';
  requestAnimationFrame(() => { msg.style.opacity = 1; });
  _victoryTimeouts.push(setTimeout(() => {
    msg.style.opacity = 0;
    _victoryTimeouts.push(setTimeout(_victoryStageReplay, _VICTORY_FADE_MS));
  }, _VICTORY_HOLD_MS));
}

function _victoryStageReplay() {
  // Final stage — replay the boss-victory match on a small canvas while
  // the BGM plays, looping forever. The first loop is captured to a
  // downloadable WebM (canvas video + master audio); after that the
  // canvas keeps looping for the player while the file stays available.
  const msg = document.getElementById('victory-message');
  if (msg) msg.style.display = 'none';
  if (!victoryReplayStage || !victoryReplayCanvas) {
    _victoryStageActions();
    return;
  }
  victoryReplayStage.style.display = 'flex';
  if (victoryReplayCaption) {
    victoryReplayCaption.textContent = _heroName ? _heroName.toUpperCase() : 'THE HERO';
  }
  if (!_victoryReplayRenderer) {
    _victoryReplayRenderer = createRenderer(victoryReplayCanvas);
  }
  // Always reveal the download button. It starts in a "preparing" state
  // (greyed-out, click-blocked) and becomes active once the recorder
  // finishes writing the first loop.
  if (victoryReplayDownload) {
    victoryReplayDownload.classList.add('preparing');
    if (victoryReplayDownload.href) {
      try { URL.revokeObjectURL(victoryReplayDownload.href); } catch (e) {}
      victoryReplayDownload.removeAttribute('href');
    }
  }
  // The "back to menu" button stays visible the whole time — there's no
  // need to wait for the loop to finish since the loop never finishes.
  _victoryStageActions();

  const frames = _bossReplay;
  if (!frames.length) return;

  // Build an offscreen canvas that has the gold ember frame, the hero's
  // name, and the live game scaled into the centre. The recording captures
  // THIS canvas so the downloaded video has the whole presentation baked
  // in (the on-screen view stays the bare game canvas + CSS chrome).
  if (!_recordCanvas) {
    _recordCanvas = document.createElement('canvas');
    _recordCanvas.width  = _RECORD_W;
    _recordCanvas.height = _RECORD_H;
  }

  // Two recording paths, picked at runtime:
  //
  //   1. WebCodecs + mp4-muxer → guaranteed .mp4 (Chrome 94+, Edge 94+,
  //      Safari 16.4+, Firefox 130+ etc.). Loaded on demand from CDN.
  //   2. MediaRecorder fallback → .webm (Firefox without WebCodecs, very
  //      old browsers).
  //
  // The on-screen loop is deferred until the recorder is ready — that
  // way the recording starts at frame 0 of the buffer rather than
  // wherever an async-racing tickPlay had already advanced to (the bug
  // that made saved videos appear to "begin from the ending").
  let mp4Encoder        = null;
  let mediaRecorder     = null;
  let mediaRecorderExt  = 'webm';
  const mediaChunks     = [];
  let recordingFinished = false;  // first full loop captured
  let recordingBaseTs   = 0;      // performance.now() when encoding started

  const audioStream = getAudioStream();

  function _setupMediaRecorderFallback() {
    if (typeof MediaRecorder === 'undefined' || !_recordCanvas.captureStream) return;
    const videoStream = _recordCanvas.captureStream(30);
    const combined = new MediaStream([
      ...videoStream.getVideoTracks(),
      ...audioStream.getAudioTracks(),
    ]);
    const candidates = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus',
      'video/webm',
    ];
    const mime = candidates.find(m => MediaRecorder.isTypeSupported(m)) || '';
    mediaRecorderExt = 'webm';
    mediaRecorder = new MediaRecorder(combined, {
      ...(mime ? { mimeType: mime } : {}),
      // ~20 Mbps to keep the 4K composite from compressing into mush in
      // the WebM fallback path.
      videoBitsPerSecond: 20_000_000,
      audioBitsPerSecond: 192_000,
    });
    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size > 0) mediaChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const blob = new Blob(mediaChunks, { type: mediaRecorder.mimeType || `video/${mediaRecorderExt}` });
      _surfaceDownload(blob, mediaRecorderExt);
    };
    // 100 ms timeslice ensures short loops (a few seconds) still produce
    // ondataavailable events; without it MediaRecorder can flush nothing
    // when stop() arrives before the default ~1 s collection window.
    mediaRecorder.start(100);
    recordingBaseTs = performance.now();
  }

  function _surfaceDownload(blob, ext) {
    if (!victoryReplayDownload) return;
    if (victoryReplayDownload.href) {
      try { URL.revokeObjectURL(victoryReplayDownload.href); } catch (e) {}
    }
    const url = URL.createObjectURL(blob);
    victoryReplayDownload.href = url;
    victoryReplayDownload.download =
      `${(_heroName || 'hero').toLowerCase().replace(/\s+/g, '-')}-snake-pong-victory.${ext}`;
    victoryReplayDownload.classList.remove('preparing');
  }

  function _startReplayLoop() {
    // Soundtrack the replay with the celebration BGM. It was stopped at
    // endGame; restart it here so the first loop is baked into the file
    // and subsequent loops still have music for the on-screen replay.
    setBgmStyle('celebration');
    startBgm();

    let recordStartFrameIdx = null;
    let i = 0;
    const tickPlay = () => {
      if (i >= frames.length) i = 0;

      const f = frames[i];
      _victoryReplayRenderer.draw(f.s1, f.s2, f.ball, [], [], []);
      _drawCompositeRecordFrame();

      // Mid-flight bail-out: if the WebCodecs path silently failed,
      // drop it, set up MediaRecorder, and rewind to frame 0 so the
      // recording still begins at the start of the match.
      if (mp4Encoder && mp4Encoder.failed && mp4Encoder.failed()) {
        console.warn('[boss-victory] mp4 path failed; switching to MediaRecorder');
        mp4Encoder = null;
        recordStartFrameIdx = null;
        try { _setupMediaRecorderFallback(); }
        catch (e) { console.warn('[boss-victory] fallback setup failed:', e); }
        i = 0;
        _victoryTimeouts.push(setTimeout(tickPlay, _REPLAY_FRAME_MS));
        return;
      }

      const recordingActive = !!(mp4Encoder || (mediaRecorder && mediaRecorder.state === 'recording'));
      if (recordingActive && !recordingFinished) {
        if (recordStartFrameIdx === null) {
          recordStartFrameIdx = i;
        } else if (i === recordStartFrameIdx) {
          // Full circle — exactly one loop is captured. Finalise.
          recordingFinished = true;
          if (mp4Encoder) {
            mp4Encoder.finalize()
              .then(blob => _surfaceDownload(blob, 'mp4'))
              .catch(e => {
                console.warn('[boss-victory] mp4 finalise failed; falling back to WebM:', e);
                recordingFinished = false;
                recordStartFrameIdx = null;
                mp4Encoder = null;
                try { _setupMediaRecorderFallback(); }
                catch (e2) { console.warn('[boss-victory] fallback setup failed:', e2); }
                i = 0;
              });
          } else if (mediaRecorder && mediaRecorder.state === 'recording') {
            try { mediaRecorder.stop(); } catch (e) {}
          }
        }
        if (mp4Encoder && !recordingFinished) {
          const tsUs = Math.max(0, (performance.now() - recordingBaseTs) * 1000) | 0;
          mp4Encoder.encodeFrame(tsUs);
        }
      }

      i++;
      _victoryTimeouts.push(setTimeout(tickPlay, _REPLAY_FRAME_MS));
    };
    tickPlay();
  }

  // Kick off WebCodecs setup. Only START the on-screen + recording loop
  // once the chosen path is ready — that way the recording captures from
  // frame 0 of _bossReplay rather than wherever a racing tickPlay had
  // already advanced to (the bug that made saved videos appear to start
  // from the ending of the match).
  _setupMp4Recorder(_recordCanvas, audioStream).then(rec => {
    if (rec) {
      mp4Encoder = rec;
      recordingBaseTs = performance.now();
    } else {
      try { _setupMediaRecorderFallback(); }
      catch (e) { console.warn('[boss-victory] recording unavailable:', e); }
    }
    _startReplayLoop();
  }).catch(e => {
    console.warn('[boss-victory] mp4 setup error:', e);
    try { _setupMediaRecorderFallback(); }
    catch (e2) { console.warn('[boss-victory] recording unavailable:', e2); }
    _startReplayLoop();
  });
}

function _victoryStageActions() {
  if (victoryActions) victoryActions.classList.add('visible');
}

function _hideBossVictoryCelebration() {
  _victoryTimeouts.forEach(clearTimeout);
  _victoryTimeouts = [];
  // If the user dismissed the overlay mid-replay, the BGM was still
  // running — kill it before returning to the menu / next match.
  stopBgm();
  const cel = document.getElementById('boss-victory');
  if (cel) cel.style.display = 'none';
  if (victoryReplayStage) victoryReplayStage.style.display = 'none';
  if (victoryNameInput)   victoryNameInput.style.display   = 'none';
  if (victoryActions) victoryActions.classList.remove('visible');
  if (victoryReplayDownload) {
    if (victoryReplayDownload.href) {
      try { URL.revokeObjectURL(victoryReplayDownload.href); } catch (e) {}
      victoryReplayDownload.removeAttribute('href');
    }
    victoryReplayDownload.classList.add('preparing');
  }
}

// ── Settings modal ────────────────────────────────────────────────────────────
function openSettings() {
  settingsModal.classList.add('open');
  if (phase === 'playing') pause();
}

function closeSettings() {
  settingsModal.classList.remove('open');
  if (phase === 'paused') resume();
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
    if (aiLocalMode) return;  // AI controls P2
    if (phase !== 'playing' || !s2) return;
    if (dx !== 0 && s2.dir.x === -dx) return;
    if (dy !== 0 && s2.dir.y === -dy) return;
    s2.nextDir = { x: dx, y: dy };
  },
});

// ── UI events ─────────────────────────────────────────────────────────────────
localBtn.addEventListener('click', () => {
  if (phase === 'paused') { resume(); return; }
  // From a fresh main-menu click: 2-human local game.
  // From a gameover replay: keep the current mode (AI mode persists).
  if (phase !== 'gameover') {
    aiLocalMode = false;
  }
  startGame();
});

aiBtn.addEventListener('click', () => {
  mainMenu.style.display = 'none';
  aiPanel.style.display = 'flex';
});
aiBack.addEventListener('click', () => {
  aiPanel.style.display = 'none';
  localBtn.textContent = 'LOCAL';
  localBtn.style.display = '';
  aiBtn.style.display = '';
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
  // `btn` may be null when launched from the boss-celebration "FIGHT THE
  // BOSS" button, which isn't part of the variant group.
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
  bossModeActive = btn.dataset.variant === 'boss';
  _loadAndPlayVariant(btn.dataset.variant, btn);
});

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
    aiPanel.style.display = 'none';
    localBtn.textContent = 'LOCAL';
    localBtn.style.display = '';
    aiBtn.style.display = '';
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
  // Hide the input immediately and let the frame settle for one fade
  // duration before the next sentence appears — gives a clear breath
  // between "tell me your name" and "we shall remember your story".
  if (victoryNameInput) victoryNameInput.style.display = 'none';
  _victoryTimeouts.push(setTimeout(_victoryStageEternalStory, _VICTORY_FADE_MS));
}
if (victoryNameSubmit) {
  victoryNameSubmit.addEventListener('click', _commitHeroNameAndAdvance);
}
if (victoryNameField) {
  victoryNameField.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); _commitHeroNameAndAdvance(); }
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
    mainMenu.style.display = 'flex';
  });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
buildLegend();
startLoop();
