// Pure obs-building helper for the Snake-Pong ONNX agent (no ort dependency).
// Mirrors rl/ai_guest.py::_build_obs_from_state + rl/play_server.py phase tracking.
// Kept in its own file so Node-side parity tests can import it without pulling in
// onnxruntime-web (which targets browsers).

export const COLS = 36;
export const ROWS = 26;

// Egocentric action encoding: 0=up, 1=down, 2=left, 3=right.
export const ACTION_DELTAS = [[0, -1], [0, 1], [-1, 0], [1, 0]];

// For the right-side player (P2), mirror x-axis: {up,down} unchanged, left<->right.
export const MIRROR_ACTION = [0, 1, 3, 2];

// Build the flat egocentric obs vector from P2's POV.
//   s1, s2: { body: [{x,y}, ...] }
//   ball:   { x, y, vx, vy }
//   meta:   { snake_length, snake_multiplier, interp_ball_obs }
//   phase:  integer ∈ [0, snake_multiplier)
// Returns Float32Array of length 4*L + 4.
export function buildObsP2(s1, s2, ball, meta, phase) {
  const L = meta.snake_length;
  const mult = meta.snake_multiplier;
  const interp = meta.interp_ball_obs;
  const nx = COLS - 1;
  const ny = ROWS - 1;

  const obs = new Float32Array(4 * L + 4);

  const encodeBody = (body, off) => {
    const bd = body.length >= L
      ? body.slice(0, L)
      : body.concat(Array(L - body.length).fill(body[body.length - 1]));
    for (let i = 0; i < L; i++) {
      const c = bd[i];
      const xm = nx - c.x;  // mirror
      obs[off + 2 * i] = xm / nx;
      obs[off + 2 * i + 1] = c.y / ny;
    }
  };
  encodeBody(s2.body, 0);          // own
  encodeBody(s1.body, 2 * L);      // opp

  let bx = ball.x, by = ball.y, vx = ball.vx, vy = ball.vy;
  if (interp && mult > 1) {
    const frac = phase / mult;
    bx = bx + frac * vx;
    by = by + frac * vy;
    vx = vx / mult;
    vy = vy / mult;
  }
  obs[4 * L + 0] = (nx - bx) / nx;  // mirror x
  obs[4 * L + 1] = by / ny;
  obs[4 * L + 2] = -vx;             // mirror vx
  obs[4 * L + 3] = vy;
  return obs;
}

// Maintain the "phase" counter from a stream of states. Returns the updated
// { phase, lastBallKey }. Call before buildObsP2 on each AI inference.
export function updatePhase(phase, lastBallKey, ball, mult) {
  const key = `${ball.x},${ball.y}`;
  if (lastBallKey === null || key !== lastBallKey) {
    return { phase: 0, lastBallKey: key };
  }
  return {
    phase: Math.min(phase + 1, Math.max(1, mult) - 1),
    lastBallKey: key,
  };
}
