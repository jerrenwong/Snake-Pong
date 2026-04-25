// Node-side half of the JS/Python obs parity test.
//
// Reads test cases from a JSON file whose path is given as argv[2], computes
// each one's obs with buildObsP2 from src/ai_local_obs.js, and writes the
// flat obs arrays (one per line, space-separated) to stdout. The Python side
// compares to its own reference and reports mismatches.
import { buildObsP2 } from '../../src/ai_local_obs.js';
import * as fs from 'fs';

const inputPath = process.argv[2];
if (!inputPath) {
  console.error('usage: node parity_js_obs.mjs <cases.json>');
  process.exit(2);
}
const cases = JSON.parse(fs.readFileSync(inputPath, 'utf8'));

for (const c of cases) {
  const obs = buildObsP2(c.s1, c.s2, c.ball, c.meta, c.phase);
  // Stream as whitespace-separated floats — round-trip through toPrecision(17).
  process.stdout.write(Array.from(obs).map(v => v.toPrecision(17)).join(' ') + '\n');
}
