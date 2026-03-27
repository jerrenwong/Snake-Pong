import { CELL, COLS, ROWS, W, H, WALL_L, WALL_R } from './constants.js';

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

  ctx.fillStyle = '#1e1e1e';
  ctx.fillRect(WALL_L * CELL, 0, 2 * CELL, H);

  ctx.strokeStyle = '#2c2c2c';
  ctx.lineWidth = 1;
  for (let r = 0; r < ROWS; r++) {
    ctx.strokeRect(WALL_L * CELL + 1, r * CELL + 1, 2 * CELL - 2, CELL - 2);
  }

  ctx.fillStyle = 'rgba(0, 160, 255, 0.03)';
  ctx.fillRect(0, 0, WALL_L * CELL, H);

  ctx.fillStyle = 'rgba(255, 100, 0, 0.03)';
  ctx.fillRect(WALL_R * CELL, 0, W - WALL_R * CELL, H);
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

  function draw(s1, s2, ball) {
    ctx.drawImage(bgCanvas, 0, 0);
    if (!s1 || !s2 || !ball) return;

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
  }

  return { draw };
}
