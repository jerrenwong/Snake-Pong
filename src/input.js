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
}
