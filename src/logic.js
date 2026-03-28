import { COLS, ROWS, WALL_L, WALL_R } from './constants.js';

// ── Entity factories ──────────────────────────────────────────────────────────

export function createSnakes(len) {
  const hy  = Math.floor(ROWS / 2);
  const h1x = Math.floor(WALL_L / 2);
  const h2x = WALL_R + Math.floor((COLS - WALL_R) / 2);

  const s1 = {
    body:    Array.from({ length: len }, (_, i) => ({ x: h1x - i, y: hy })),
    dir:     { x:  1, y: 0 },
    nextDir: { x:  1, y: 0 },
    color: '#2af',
  };

  const s2 = {
    body:    Array.from({ length: len }, (_, i) => ({ x: h2x + i, y: hy })),
    dir:     { x: -1, y: 0 },
    nextDir: { x: -1, y: 0 },
    color: '#f62',
  };

  return { s1, s2 };
}

export function createBall() {
  const side = Math.random() < 0.5 ? -1 : 1;
  const bx   = side < 0
    ? Math.floor(WALL_L / 2)
    : WALL_R + Math.floor((COLS - WALL_R) / 2);
  return {
    x:  bx,
    y:  Math.floor(ROWS / 2),
    vx: side,
    vy: Math.random() < 0.5 ? 1 : -1,
  };
}

// ── Snake logic ───────────────────────────────────────────────────────────────

export function stepSnake(snake) {
  snake.dir = snake.nextDir;
  const head = snake.body[0];
  snake.body.unshift({ x: head.x + snake.dir.x, y: head.y + snake.dir.y });
  snake.body.pop();
}

// wallSet: Set<"x,y"> of impassable map-wall cells (null/undefined = no walls)
export function snakeHitsDeath(snake, wallSet) {
  const h = snake.body[0];
  if (h.x < 0 || h.x >= COLS || h.y < 0 || h.y >= ROWS) return true;
  if (wallSet && wallSet.has(`${h.x},${h.y}`)) return true;
  return snake.body.slice(1).some(c => c.x === h.x && c.y === h.y);
}

export function snakesCollide(s1, s2) {
  const h1 = s1.body[0], h2 = s2.body[0];
  if (h1.x === h2.x && h1.y === h2.y) return 'both';
  if (s2.body.some(c => c.x === h1.x && c.y === h1.y)) return 's1';
  if (s1.body.some(c => c.x === h2.x && c.y === h2.y)) return 's2';
  return null;
}

// ── Ball logic ────────────────────────────────────────────────────────────────

// Returns null if no score, or 1/2 for the player who earned the point.
// Mutates ball position in place.
// wallSet: Set<"x,y"> of impassable map-wall cells (null/undefined = no walls)
export function stepBall(ball, s1, s2, wallSet) {
  let nx = ball.x + ball.vx;
  let ny = ball.y + ball.vy;

  // Top/bottom bounce
  if (ny < 0 || ny >= ROWS) {
    ball.vy = -ball.vy;
    ny = ball.y + ball.vy;
  }

  // Ball exits left → P1 missed → P2 scores
  if (nx < 0)     return 2;
  // Ball exits right → P2 missed → P1 scores
  if (nx >= COLS) return 1;

  // Obstacle check: snake bodies + map walls
  const w = (x, y) => wallSet ? wallSet.has(`${x},${y}`) : false;
  const allSegs = [...s1.body, ...s2.body];
  const hHit = w(nx, ball.y) || allSegs.some(c => c.x === nx    && c.y === ball.y);
  const vHit = w(ball.x, ny) || allSegs.some(c => c.x === ball.x && c.y === ny);
  const dHit = !hHit && !vHit && (w(nx, ny) || allSegs.some(c => c.x === nx && c.y === ny));

  if (hHit || vHit || dHit) {
    if (hHit) ball.vx = -ball.vx;
    if (vHit) ball.vy = -ball.vy;
    if (dHit) { ball.vx = -ball.vx; ball.vy = -ball.vy; }
    nx = ball.x + ball.vx;
    ny = Math.max(0, Math.min(ROWS - 1, ball.y + ball.vy));
  }

  ball.x = nx;
  ball.y = ny;
  return null;
}

// ── Speed helpers ─────────────────────────────────────────────────────────────

// Ball speed slider 1–10 → tps 1–12
// Snake tick interval is derived as: ballTickMs / snakeMultiplier (in main.js)
export function getBallTps(sliderValue) {
  return 1 + (sliderValue - 1) * (11 / 9);
}
