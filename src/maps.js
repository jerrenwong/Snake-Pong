// ── Map definitions ───────────────────────────────────────────────────────────
// Each map: { id, name, cells: [{x,y},...], walls: Set<"x,y"> }
// 'walls' is a Set for O(1) collision lookup; 'cells' is an array for rendering.

function makeWallSet(cells) {
  return new Set(cells.map(({ x, y }) => `${x},${y}`));
}

function defineMap(id, name, cells) {
  return { id, name, cells, walls: makeWallSet(cells) };
}

// Smiley face centred at (17, 12) — eyes and smile only, no outer ring.
function smileyCells() {
  const cx = 17, cy = 12;
  const cells = [];

  // Left eye (3×3 block)
  for (let ey = 0; ey <= 2; ey++)
    for (let ex = 0; ex <= 2; ex++)
      cells.push({ x: cx - 5 + ex, y: cy - 3 + ey });

  // Right eye (3×3 block)
  for (let ey = 0; ey <= 2; ey++)
    for (let ex = 0; ex <= 2; ex++)
      cells.push({ x: cx + 3 + ex, y: cy - 3 + ey });

  // Smile: wide arc across the lower half of the face
  [
    [-5, 3], [-4, 4], [-3, 5], [-2, 5], [-1, 5],
    [0, 5],  [1, 5],  [2, 5],  [3, 4],  [4, 3],
  ].forEach(([dx, dy]) => cells.push({ x: cx + dx, y: cy + dy }));

  return cells;
}

export const MAPS = [
  defineMap('open',   'Open',   []),
  defineMap('smiley', ':)',     smileyCells()),
];

export const DEFAULT_MAP = MAPS[0];
