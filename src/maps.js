// ── Map definitions ───────────────────────────────────────────────────────────
// Each map: { id, name, cells: [{x,y},...], walls: Set<"x,y"> }
// 'walls' is a Set for O(1) collision lookup; 'cells' is an array for rendering.

function makeWallSet(cells) {
  return new Set(cells.map(({ x, y }) => `${x},${y}`));
}

function defineMap(id, name, cells) {
  return { id, name, cells, walls: makeWallSet(cells) };
}

// Smiley face centred at (17, 12) — the grid centre of a 36×26 board.
function smileyCells() {
  const cx = 17, cy = 12;
  const cells = [];

  // Face outline: ring where 22 ≤ dx²+dy² ≤ 44  (approx radius 4.7–6.6 cells)
  for (let dy = -7; dy <= 7; dy++) {
    for (let dx = -7; dx <= 7; dx++) {
      const r2 = dx * dx + dy * dy;
      if (r2 >= 22 && r2 <= 44) cells.push({ x: cx + dx, y: cy + dy });
    }
  }

  // Left eye (2×2 block)
  for (let ey = 0; ey <= 1; ey++)
    for (let ex = 0; ex <= 1; ex++)
      cells.push({ x: cx - 3 + ex, y: cy - 2 + ey });

  // Right eye (2×2 block)
  for (let ey = 0; ey <= 1; ey++)
    for (let ex = 0; ex <= 1; ex++)
      cells.push({ x: cx + 2 + ex, y: cy - 2 + ey });

  // Smile arc
  [[-3, 2], [-2, 3], [-1, 3], [0, 3], [1, 3], [2, 3], [3, 2]]
    .forEach(([dx, dy]) => cells.push({ x: cx + dx, y: cy + dy }));

  return cells;
}

export const MAPS = [
  defineMap('open',   'Open',   []),
  defineMap('smiley', ':)',     smileyCells()),
];

export const DEFAULT_MAP = MAPS[0];
