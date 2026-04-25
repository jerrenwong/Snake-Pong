import { CELL, COLS, ROWS, W, H } from './constants.js';
import { POWERUP_DEFS } from './powerups.js';

function buildBackground(ctx) {
  ctx.fillStyle = '#0e0e0e';
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = '#161616';
  ctx.lineWidth = 1;
  for (let c = 0; c <= COLS; c++) {
    ctx.beginPath(); ctx.moveTo(c * CELL, 0); ctx.lineTo(c * CELL, H); ctx.stroke();
  }
  for (let r = 0; r <= ROWS; r++) {
    ctx.beginPath(); ctx.moveTo(0, r * CELL); ctx.lineTo(W, r * CELL); ctx.stroke();
  }
}

function drawSnake(ctx, snake) {
  const body = snake.body;
  const n    = body.length;
  ctx.fillStyle = snake.color;
  for (let i = 0; i < n; i++) {
    const t = i / n;
    ctx.globalAlpha = i === 0 ? 1 : Math.max(0.25, 1 - t * 0.75);
    const seg = body[i];
    ctx.fillRect(seg.x * CELL + 2, seg.y * CELL + 2, CELL - 4, CELL - 4);
  }
  ctx.globalAlpha = 1;

  // Eyes
  const h  = body[0];
  const cx = h.x * CELL + CELL / 2;
  const cy = h.y * CELL + CELL / 2;
  const dx = snake.dir.x, dy = snake.dir.y;
  ctx.fillStyle = '#000';
  [[-dy * 4 + dx * 3, dx * 4 + dy * 3], [dy * 4 + dx * 3, -dx * 4 + dy * 3]]
    .forEach(([ox, oy]) => {
      ctx.beginPath();
      ctx.arc(cx + ox, cy + oy, 2, 0, Math.PI * 2);
      ctx.fill();
    });
}

// ── Public factory ────────────────────────────────────────────────────────────

export function createRenderer(canvas) {
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  // Pre-render static background once
  const bgCanvas    = document.createElement('canvas');
  bgCanvas.width    = W;
  bgCanvas.height   = H;
  buildBackground(bgCanvas.getContext('2d'));

  function drawPowerups(powerups, now) {
    for (const p of powerups) {
      const def  = POWERUP_DEFS[p.type];
      const px   = p.x * CELL;
      const py   = p.y * CELL;
      const r    = CELL / 2 - 2;
      const cx   = px + CELL / 2;
      const cy   = py + CELL / 2;

      // Fade out in the last 3 seconds before field expiry
      const timeLeft = p.fieldExpiresAt - now;
      const FADE_START = 3_000;
      const baseAlpha = timeLeft < FADE_START
        ? Math.max(0.15, timeLeft / FADE_START) * (0.5 + 0.5 * Math.sin(now / 120))
        : 1;

      // Pulsing glow
      const pulse = 0.5 + 0.5 * Math.sin(now / 300);
      ctx.globalAlpha = baseAlpha * (0.3 + 0.25 * pulse);
      ctx.fillStyle   = def.glow;
      ctx.beginPath();
      ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = baseAlpha;

      // Filled circle
      ctx.fillStyle = def.color;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle  = '#000';
      ctx.font       = `bold ${Math.round(CELL * 0.45)}px monospace`;
      ctx.textAlign  = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(def.label, cx, cy);
      ctx.globalAlpha = 1;
    }
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  // effects: [{ type, player, expiresAt, totalDuration }]
  function drawEffects(effects, now) {
    const BAR_W = 60, BAR_H = 6, MARGIN = 6;
    const p1Effects = effects.filter(e => e.player === 1);
    const p2Effects = effects.filter(e => e.player === 2);

    function drawBar(eff, x, y) {
      const def     = POWERUP_DEFS[eff.type];
      const frac    = Math.max(0, (eff.expiresAt - now) / eff.totalDuration);
      ctx.fillStyle = '#222';
      ctx.fillRect(x, y, BAR_W, BAR_H);
      ctx.fillStyle = def.color;
      ctx.fillRect(x, y, BAR_W * frac, BAR_H);
      ctx.font         = `${BAR_H + 2}px monospace`;
      ctx.textBaseline = 'middle';
      ctx.fillStyle    = '#fff';
    }

    // P1 bars — top-left
    p1Effects.forEach((eff, i) => {
      drawBar(eff, MARGIN, MARGIN + i * (BAR_H + 4));
    });

    // P2 bars — top-right
    p2Effects.forEach((eff, i) => {
      drawBar(eff, W - MARGIN - BAR_W, MARGIN + i * (BAR_H + 4));
    });

    ctx.textBaseline = 'alphabetic';
  }

  function draw(s1, s2, ball, powerups = [], effects = [], wallCells = []) {
    const now = performance.now();
    ctx.drawImage(bgCanvas, 0, 0);
    if (!s1 || !s2 || !ball) return;

    // Draw map wall cells
    if (wallCells.length > 0) {
      ctx.fillStyle = '#1a1a2e';
      for (const { x, y } of wallCells)
        ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
      ctx.strokeStyle = '#2a2a50';
      ctx.lineWidth = 1;
      for (const { x, y } of wallCells)
        ctx.strokeRect(x * CELL + 0.5, y * CELL + 0.5, CELL - 1, CELL - 1);
    }

    drawPowerups(powerups, now);
    drawSnake(ctx, s1);
    drawSnake(ctx, s2);

    const bx = ball.x * CELL + CELL / 2;
    const by = ball.y * CELL + CELL / 2;
    ctx.fillStyle = 'rgba(255,255,255,0.18)';
    ctx.beginPath();
    ctx.arc(bx, by, CELL / 2 + 1, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(bx, by, CELL / 2 - 3, 0, Math.PI * 2);
    ctx.fill();

    if (effects.length > 0) drawEffects(effects, now);
  }

  return { draw };
}
