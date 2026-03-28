import { COLS, ROWS, WALL_L, WALL_R } from './constants.js'; // WALL_L/WALL_R used for half-boundary only

// ── Power-up definitions ──────────────────────────────────────────────────────
// Each entry: label, color, glow, duration (ms), description
export const POWERUP_DEFS = {
  length_boost: {
    label:       '+5',
    color:       '#44ff88',
    glow:        'rgba(68,255,136,0.45)',
    duration:    10_000,
    description: 'Snake grows 5 cells for 10s',
  },
  ball_speed: {
    label:       '⚡',
    color:       '#ffdd00',
    glow:        'rgba(255,221,0,0.45)',
    duration:    10_000,
    description: 'Ball 2× faster in opponent\'s half for 10s',
  },
  snake_speed: {
    label:       '>>',
    color:       '#ff55ff',
    glow:        'rgba(255,85,255,0.45)',
    duration:    10_000,
    description: 'Your snake moves 2× faster for 10s',
  },
};

export const POWERUP_IDS = Object.keys(POWERUP_DEFS);

export const SPAWN_COOLDOWN_MS = 5_000;   // rest period after each spawn (per player)
export const SPAWN_EXPECTED_MS = 10_000;  // expected wait after cooldown (geometric, per player)
export const FIELD_EXPIRE_MS   = 10_000;  // uncollected power-up disappears after this

let _uid = 0;

// Spawns a power-up in the given player's half (player = 1 or 2).
// Returns { uid, type, player, x, y, fieldExpiresAt } or null if no free cell.
// wallSet: Set<"x,y"> of impassable map-wall cells (null/undefined = no extra walls)
export function spawnPowerup(s1, s2, existing, player, now, wallSet = null) {
  const blocked = new Set();

  // Snake body cells
  for (const seg of s1.body) blocked.add(`${seg.x},${seg.y}`);
  for (const seg of s2.body) blocked.add(`${seg.x},${seg.y}`);

  // Map wall cells
  if (wallSet) for (const key of wallSet) blocked.add(key);

  // Existing power-ups on field
  for (const p of existing) blocked.add(`${p.x},${p.y}`);

  // Restrict to the player's own half
  const colMin = player === 1 ? 0       : WALL_R + 1;
  const colMax = player === 1 ? WALL_L  : COLS;

  const free = [];
  for (let c = colMin; c < colMax; c++) {
    for (let r = 0; r < ROWS; r++) {
      if (!blocked.has(`${c},${r}`)) free.push({ x: c, y: r });
    }
  }

  if (free.length === 0) return null;

  const cell = free[Math.floor(Math.random() * free.length)];
  const type = POWERUP_IDS[Math.floor(Math.random() * POWERUP_IDS.length)];
  return { uid: ++_uid, type, player, x: cell.x, y: cell.y, fieldExpiresAt: now + FIELD_EXPIRE_MS };
}
