# Snake Pong

A two-player game combining Snake and Pong. Each player controls a snake that acts as their paddle — you have to maneuver your snake to deflect the ball past your opponent.

## How to Play

Open `index.html` in any modern browser. No install or server needed.

### Controls

| Player | Left | Right | Up | Down |
|--------|------|-------|----|------|
| P1 (blue, left side) | `A` | `D` | `W` | `S` |
| P2 (orange, right side) | `←` | `→` | `↑` | `↓` |

**ESC** — pause / resume

### Objective

Deflect the ball so it passes your opponent's snake and exits through their side of the board. First player to reach the Win Score wins.

### Losing a point

- The ball exits through **your side** of the board (you failed to deflect it)
- Your snake **hits the outer wall**, the **center divider**, or **itself**
- Your snake **collides head-on** with the other snake

When a snake dies mid-rally the opposing player scores a point and a new round starts.

## Configuration

All sliders are available before (and between) rounds:

| Slider | Effect |
|--------|--------|
| **Snake Length** | Starting length of each snake (3–20 segments) |
| **Snake Speed** | How fast the snakes move (ticks per second) |
| **Ball Speed** | How fast the ball moves, independent of snake speed |
| **Win Score** | Number of points needed to win the match (1–20) |

## Rules

- Snakes cannot reverse 180° — you can only turn left or right relative to your current direction.
- The ball bounces off the top and bottom walls and off any part of either snake.
- The center divider is deadly to snakes but the ball passes straight through it (wall only affects snakes).
