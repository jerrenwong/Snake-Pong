// Keyboard input — no game state here.
// Parses keys into semantic events and fires callbacks supplied by main.js.
// All guards (phase check, 180° check) live in the callbacks.

const P1_KEYS = {
  a: [-1,  0], A: [-1,  0],
  d: [ 1,  0], D: [ 1,  0],
  w: [ 0, -1], W: [ 0, -1],
  s: [ 0,  1], S: [ 0,  1],
};

const P2_KEYS = {
  ArrowLeft:  [-1,  0],
  ArrowRight: [ 1,  0],
  ArrowUp:    [ 0, -1],
  ArrowDown:  [ 0,  1],
};

const PREVENT_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ']);

export function registerInput({ onEscape, onDirectionP1, onDirectionP2 }) {
  document.addEventListener('keydown', e => {
    if (PREVENT_KEYS.has(e.key)) e.preventDefault();

    if (e.key === 'Escape') { onEscape(); return; }

    const d1 = P1_KEYS[e.key];
    if (d1) { onDirectionP1(d1[0], d1[1]); return; }

    const d2 = P2_KEYS[e.key];
    if (d2) { onDirectionP2(d2[0], d2[1]); }
  });

  // ── Touch swipe controls ─────────────────────────────────────────────────
  const MIN_SWIPE = 20; // px threshold
  let touchStartX = 0, touchStartY = 0;

  document.addEventListener('touchstart', e => {
    const t = e.touches[0];
    touchStartX = t.clientX;
    touchStartY = t.clientY;
  }, { passive: true });

  document.addEventListener('touchmove', e => {
    // Prevent scroll while playing
    if (!e.target.closest('#settings-modal, #settings-panel')) e.preventDefault();
  }, { passive: false });

  document.addEventListener('touchend', e => {
    const t = e.changedTouches[0];
    const dx = t.clientX - touchStartX;
    const dy = t.clientY - touchStartY;
    if (Math.abs(dx) < MIN_SWIPE && Math.abs(dy) < MIN_SWIPE) return;

    let sx, sy;
    if (Math.abs(dx) > Math.abs(dy)) { sx = dx > 0 ? 1 : -1; sy = 0; }
    else                              { sx = 0; sy = dy > 0 ? 1 : -1; }

    // Left half → P1, right half → P2
    if (touchStartX < window.innerWidth / 2) onDirectionP1(sx, sy);
    else                                      onDirectionP2(sx, sy);
  });
}
