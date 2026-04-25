// server.js — Snake Pong relay server
// Serves static files AND handles WebSocket room relay on the same port.
// Usage: npm install && node server.js
// Both players open http://[this-machine-ip]:3000

const http = require('http');
const fs   = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');

// ── Static file server ────────────────────────────────────────────────────────

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript',
  '.mjs':  'application/javascript',
  '.css':  'text/css',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.wasm': 'application/wasm',
};

const ROOT = __dirname;

const server = http.createServer((req, res) => {
  // Strip query string and prevent directory traversal
  const urlPath = req.url.split('?')[0].replace(/\.\./g, '');
  const filePath = path.join(ROOT, urlPath === '/' ? 'index.html' : urlPath);

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not found');
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'text/plain' });
    res.end(data);
  });
});

// ── WebSocket relay ───────────────────────────────────────────────────────────

const wss = new WebSocketServer({ server });
const rooms = new Map(); // code → { host: ws, guest: ws | null }

const CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // no ambiguous chars

function genCode() {
  let code;
  do {
    code = Array.from({ length: 5 }, () =>
      CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)]
    ).join('');
  } while (rooms.has(code));
  return code;
}

wss.on('connection', (ws, req) => {
  // Disable Nagle's algorithm for minimum relay latency
  req.socket.setNoDelay(true);

  ws._code = null;
  ws._role = null;

  ws.on('message', raw => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    if (msg.type === 'host') {
      // Create a new room
      const code = genCode();
      rooms.set(code, { host: ws, guest: null });
      ws._code = code;
      ws._role = 'host';
      ws.send(JSON.stringify({ type: 'hosted', code }));
      console.log(`[${code}] Room created`);

    } else if (msg.type === 'join') {
      const code = (msg.code || '').toUpperCase().trim();
      const room = rooms.get(code);
      if (!room) {
        ws.send(JSON.stringify({ type: 'error', reason: 'Room not found' }));
        return;
      }
      if (room.guest) {
        ws.send(JSON.stringify({ type: 'error', reason: 'Room is full' }));
        return;
      }
      room.guest = ws;
      ws._code = code;
      ws._role = 'guest';
      ws.send(JSON.stringify({ type: 'joined' }));
      room.host.send(JSON.stringify({ type: 'guest_joined' }));
      console.log(`[${code}] Guest joined`);

    } else if (msg.type === 'relay') {
      // Forward the raw message to the other player
      const room = rooms.get(ws._code);
      if (!room) return;
      const target = ws._role === 'host' ? room.guest : room.host;
      if (target && target.readyState === 1) {
        target.send(raw.toString());
      }
    }
  });

  ws.on('close', () => {
    if (!ws._code) return;
    const room = rooms.get(ws._code);
    if (!room) return;
    const other = ws._role === 'host' ? room.guest : room.host;
    if (other && other.readyState === 1) {
      other.send(JSON.stringify({ type: 'opponent_left' }));
    }
    rooms.delete(ws._code);
    console.log(`[${ws._code}] Room closed (${ws._role} disconnected)`);
  });
});

// ── Start ─────────────────────────────────────────────────────────────────────

// ── Keepalive pings (prevent Railway / proxy idle-timeout disconnects) ────
const PING_INTERVAL = 15_000;
setInterval(() => {
  for (const ws of wss.clients) {
    if (ws.readyState === 1) ws.ping();
  }
}, PING_INTERVAL);

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Snake Pong running at http://localhost:${PORT}`);
  console.log('Share your LAN IP so others can connect, e.g. http://192.168.x.x:3000');
});
