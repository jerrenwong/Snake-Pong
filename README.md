# Snake Pong

A two-player game combining Snake and Pong. Each player controls a snake that acts as their paddle — maneuver your snake to deflect the ball past your opponent.

> **Requires a local HTTP server** (ES modules don't work over `file://`).
> Run `python3 -m http.server` in the project directory and open `http://localhost:8000`.

## How to Play

### Controls

| Player | Left | Right | Up | Down |
|--------|------|-------|----|------|
| P1 (blue, left side) | `A` | `D` | `W` | `S` |
| P2 (orange, right side) | `←` | `→` | `↑` | `↓` |

**ESC** — pause / resume

### Objective

Deflect the ball so it exits through your opponent's side. First to reach the Win Score wins.

### Losing a point

- The ball exits through **your side** (you failed to deflect it)
- Your snake hits the **outer wall**, the **center divider**, or **itself**
- Your snake **collides head-on** with the opponent's snake

When a snake dies mid-rally the opposing player scores a point and a new round starts automatically.

## Settings (⚙ gear icon)

| Setting | Effect |
|---------|--------|
| **Snake Length** | Starting length of each snake (3–20 segments) |
| **Ball Speed** | Ball ticks per second (slider 1–10) |
| **Snake Speed** | Snake moves 1×, 2×, or 3× per ball tick |
| **Win Score** | Points needed to win the match (1–20) |
| **Power-ups** | Toggle optional power-up mode ON / OFF |

## Power-ups (optional)

Enable in Settings. When active, colored orbs spawn on the field every ~15 seconds on average (5s cooldown after each spawn, then geometric distribution with 10s expected wait). Collect a power-up by moving your snake's head onto it.

| Symbol | Color | Effect | Duration |
|--------|-------|--------|----------|
| **+5** | Green | Your snake grows 5 cells longer | 10s |
| **⚡** | Yellow | Ball moves 2× faster while in your opponent's half | 10s |
| **>>** | Pink | Your snake moves 2× faster | 10s |

A timer bar at the top corner of the canvas shows the remaining duration for each active effect. If you collect the same power-up type while it's already active, the timer resets.

## Rules

- Snakes cannot reverse 180° — only left/right turns relative to current direction.
- The ball bounces off the top/bottom walls and off any segment of either snake.
- The center divider is lethal to snakes; the ball passes through it freely.
- Power-up effects expire when the timer runs out; snake length returns to pre-boost size on expiry.

## Credits

Thanks to **Mehmet Can Bastemir** for suggestions for improvement.

## Project Structure

```
snake_pong/
├── index.html        — layout, CSS, HTML
└── src/
    ├── constants.js  — grid dimensions, wall positions
    ├── logic.js      — snake & ball physics, collision detection
    ├── renderer.js   — canvas drawing (background, snakes, ball, power-ups, effect bars)
    ├── input.js      — keyboard event → semantic callbacks
    ├── audio.js      — Web Audio API chiptune BGM + SFX (no external files)
    ├── powerups.js   — power-up definitions, spawn logic
    └── main.js       — game loop, state machine, UI wiring
```
