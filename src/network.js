// src/network.js — thin WebSocket client wrapper
// on(type, fn) replaces the handler for that message type (not additive).

export function createNetwork(url) {
  let ws = null;
  const handlers = {};

  function connect() {
    return new Promise((resolve, reject) => {
      ws = new WebSocket(url);
      ws.onopen    = () => resolve();
      ws.onerror   = () => reject(new Error('WebSocket connection failed'));
      ws.onmessage = ({ data }) => {
        let msg;
        try { msg = JSON.parse(data); } catch { return; }
        handlers[msg.type]?.(msg);
      };
      ws.onclose = () => handlers.disconnect?.({});
    });
  }

  function send(obj) {
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function on(type, fn) { handlers[type] = fn; }

  function close() { ws?.close(); ws = null; }

  return { connect, send, on, close };
}
