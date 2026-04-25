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

function drawSnake(ctx, snake, now = 0) {
  const body = snake.body;
  const n    = body.length;

  // Per-segment fill colour. For most snakes this is a constant; the
  // INSANE-tier and BOSS-tier AI use animated effects matching their button.
  let segFill = (i) => snake.color;

  if (snake.effect === 'insane-glow') {
    // Mirror the .tier-insane CSS animation (1.6s ease pulse). Pulses an
    // outer red shadow around every segment.
    const pulse = 0.5 - 0.5 * Math.cos((now / 1600) * 2 * Math.PI);
    ctx.shadowColor = '#f33';
    ctx.shadowBlur = pulse * 14;
  } else if (snake.effect === 'boss-shimmer') {
    // True metallic gold: a continuous horizontal gradient set on the canvas
    // context, scrolling 2 × W over a 3 s period so the highlight band sweeps
    // across the snake. All segments share the same gradient → smooth
    // interpolation between segments rather than the previous palette steps.
    const phase = (now / 3000) % 1;
    const offset = phase * W;
    const grad = ctx.createLinearGradient(-W + offset, 0, W + offset, 0);
    grad.addColorStop(0.00, '#3a2a00');
    grad.addColorStop(0.20, '#7a5800');
    grad.addColorStop(0.45, '#c89500');
    grad.addColorStop(0.55, '#ffe27a'); // peak highlight
    grad.addColorStop(0.65, '#c89500');
    grad.addColorStop(0.85, '#7a5800');
    grad.addColorStop(1.00, '#3a2a00');
    segFill = () => grad;
    // Pulsing gold halo.
    const pulse = 0.5 + 0.5 * Math.sin(now / 400);
    ctx.shadowColor = '#fc3';
    ctx.shadowBlur = 6 + pulse * 8;
  }

  for (let i = 0; i < n; i++) {
    const t = i / n;
    ctx.globalAlpha = i === 0 ? 1 : Math.max(0.25, 1 - t * 0.75);
    ctx.fillStyle = segFill(i);
    const seg = body[i];
    ctx.fillRect(seg.x * CELL + 2, seg.y * CELL + 2, CELL - 4, CELL - 4);
  }
  ctx.globalAlpha = 1;
  ctx.shadowBlur = 0;

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
    drawSnake(ctx, s1, now);
    drawSnake(ctx, s2, now);

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
