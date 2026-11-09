#!/usr/bin/env node
'use strict';
/**
 * DSL Parallel Folder Sync — dependency-free local helper.
 * Runs only on 127.0.0.1. No compression/decompression, plugins, or external network calls.
 * Launch from repository root: node tools/fast-folder-sync-helper.cjs
 * Open: http://127.0.0.1:8765
 */
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');

const HOST = '127.0.0.1';
const PORT = 8765;
const ORIGIN = `http://${HOST}:${PORT}`;
const SESSION_TOKEN = crypto.randomBytes(32).toString('hex');
// The public HTML remains at repository root; this helper lives in /tools/.
const CLIENT_FILE = path.resolve(__dirname, '..', 'parallel-folder-sync.html');
let lastPlan = null;
let operation = null;
let progress = { state: 'idle' };

function respond(res, status, data) {
  const bytes = Buffer.from(JSON.stringify(data));
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': bytes.length, 'Cache-Control': 'no-store' });
  res.end(bytes);
}
function fail(message, code = 400) { const err = new Error(message); err.status = code; throw err; }
function within(parent, candidate) {
  const rel = path.relative(parent, candidate);
  return rel === '' || (rel !== '..' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel));
}
function rootCheck(a, b) {
  if (within(a, b) || within(b, a)) fail('Source and destination must be separate, non-nested directories.');
}
function isEligible(rel, filter) {
  return filter === 'all' || rel.toLowerCase().endsWith('.txt');
}
async function enumerate(root, filter) {
  const result = new Map();
  let ignoredLinks = 0;
  const pending = [{ dir: root, rel: '' }];
  while (pending.length) {
    const { dir, rel } = pending.pop();
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const subrel = rel ? path.join(rel, entry.name) : entry.name;
      if (entry.isSymbolicLink()) { ignoredLinks++; continue; }
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) pending.push({ dir: full, rel: subrel });
      else if (entry.isFile() && isEligible(subrel, filter)) result.set(subrel, { full, rel: subrel });
    }
  }
  return { files: result, ignoredLinks };
}
async function pool(items, max, fn) {
  let offset = 0;
  await Promise.all(Array.from({ length: Math.min(max, items.length) }, async () => {
    while (offset < items.length) await fn(items[offset++]);
  }));
}
async function checkRoots(source, destination) {
  if (typeof source !== 'string' || typeof destination !== 'string' || !source.trim() || !destination.trim()) fail('Provide both folder paths.');
  const src = await fs.realpath(source.trim());
  const dst = await fs.realpath(destination.trim());
  if (!(await fs.stat(src)).isDirectory() || !(await fs.stat(dst)).isDirectory()) fail('Both paths must be existing directories.');
  rootCheck(src, dst);
  return { src, dst };
}
async function createPlan(payload) {
  if (operation && ['copying', 'deleting'].includes(operation.state)) fail('A copy is already running.', 409);
  const threads = Number(payload.threads);
  if (!Number.isSafeInteger(threads) || threads < 1 || threads > 64) fail('Concurrency must be between 1 and 64.');
  if (payload.filter !== 'txt' && payload.filter !== 'all') fail('Unknown file filter.');
  if (typeof payload.mirror !== 'boolean') fail('Invalid mirror mode.');
  const roots = await checkRoots(payload.source, payload.destination);
  const [source, target] = await Promise.all([enumerate(roots.src, payload.filter), enumerate(roots.dst, payload.filter)]);
  const sourceFiles = [...source.files.values()];
  const targetFiles = target.files;
  const copy = [], skip = [], remove = [];
  // A differently capitalized filename may refer to the same file on NTFS / SMB.
  // Never plan a destructive deletion for a case-only path difference.
  const foldedSource = new Set([...source.files.keys()].map(name => name.toLowerCase()));
  let byteCount = 0;
  await pool(sourceFiles, 32, async item => {
    const original = await fs.lstat(item.full);
    if (!original.isFile()) fail(`Source entry changed since scan: ${item.rel}`);
    const currentTarget = targetFiles.get(item.rel);
    const existing = currentTarget ? await fs.lstat(currentTarget.full) : null;
    if (existing && !existing.isFile()) fail(`Destination entry is not a regular file: ${item.rel}`);
    // Two seconds accounts for coarse timestamps on some network shares.
    const same = existing && original.size === existing.size && Math.abs(original.mtimeMs - existing.mtimeMs) < 2000;
    if (same) skip.push(item.rel);
    else { copy.push({ ...item, dst: path.join(roots.dst, item.rel), size: original.size, mtimeMs: original.mtimeMs }); byteCount += original.size; }
  });
  if (payload.mirror) {
    await pool([...targetFiles.values()].filter(item => !source.files.has(item.rel) && !foldedSource.has(item.rel.toLowerCase())), 32, async item => {
      const meta = await fs.lstat(item.full);
      if (meta.isFile()) remove.push({ ...item, size: meta.size, mtimeMs: meta.mtimeMs });
    });
  }
  copy.sort((a,b) => a.rel.localeCompare(b.rel));
  remove.sort((a,b) => a.rel.localeCompare(b.rel));
  const plan = {
    id: crypto.randomBytes(18).toString('hex'), created: Date.now(), ...roots,
    threads, filter: payload.filter, mirror: payload.mirror, copy, remove, skipCount: skip.length,
    total: sourceFiles.length, bytes: byteCount,
    ignoredLinks: source.ignoredLinks + target.ignoredLinks
  };
  lastPlan = plan;
  progress = { state: 'previewed' };
  return {
    planId: plan.id, source: roots.src, destination: roots.dst, files: plan.total,
    copy: copy.length, skip: skip.length, delete: remove.length, bytes: byteCount,
    ignoredLinks: plan.ignoredLinks, filter: plan.filter, mirror: plan.mirror,
    sampleCopy: copy.slice(0, 6).map(item => item.rel),
    sampleDelete: remove.slice(0, 6).map(item => item.rel)
  };
}
async function ensureSafeDestination(dstRoot, full) {
  const rel = path.relative(dstRoot, full);
  if (!rel || rel.startsWith('..' + path.sep) || path.isAbsolute(rel)) fail('Invalid destination path.');
  const parts = rel.split(path.sep);
  let dir = dstRoot;
  for (const part of parts.slice(0, -1)) {
    dir = path.join(dir, part);
    try {
      const st = await fs.lstat(dir);
      if (!st.isDirectory() || st.isSymbolicLink()) fail(`Unsafe destination directory: ${dir}`);
    } catch (e) {
      if (e.code !== 'ENOENT') throw e;
      try { await fs.mkdir(dir); } catch (err) { if (err.code !== 'EEXIST') throw err; }
      const st = await fs.lstat(dir);
      if (!st.isDirectory() || st.isSymbolicLink()) fail(`Unsafe destination directory: ${dir}`);
    }
  }
  try {
    const st = await fs.lstat(full);
    if (!st.isFile() || st.isSymbolicLink()) fail(`Unsafe destination file: ${full}`);
  } catch (err) { if (err.code !== 'ENOENT') throw err; }
}
async function perform(plan) {
  const p = operation;
  p.state = 'copying';
  await pool(plan.copy, plan.threads, async item => {
    if (p.cancelled) return;
    try {
      // Re-check that the source is a regular file before copying.
      const st = await fs.lstat(item.full);
      if (!st.isFile() || st.isSymbolicLink()) throw new Error('Source is no longer a regular file');
      await ensureSafeDestination(plan.dst, item.dst);
      await fs.copyFile(item.full, item.dst);
      await fs.utimes(item.dst, st.atime, st.mtime);
      p.copied++;
      p.bytes += st.size;
    } catch (err) {
      p.failed++;
      if (p.errors.length < 30) p.errors.push({ file: item.rel, message: err.message });
    } finally { p.completed++; }
  });
  // Mirror deletions only happen after every copy finishes successfully, never after cancellation.
  if (!p.cancelled && p.failed === 0 && plan.mirror) {
    p.state = 'deleting';
    // Delete serially to make cancellation and file-change checks meaningful.
    for (const item of plan.remove) {
      if (p.cancelled) break;
      try {
        const st = await fs.lstat(item.full);
        if (!st.isFile() || st.isSymbolicLink() || st.size !== item.size || Math.abs(st.mtimeMs - item.mtimeMs) > 2000) {
          throw new Error('Destination changed after preview; not deleted');
        }
        await fs.unlink(item.full);
        p.deleted++;
      } catch (err) {
        p.failed++;
        if (p.errors.length < 30) p.errors.push({ file: item.rel, message: err.message });
      }
      p.completed++;
    }
  }
  p.state = p.cancelled ? 'cancelled' : p.failed ? 'finished-with-errors' : 'completed';
  p.finished = Date.now();
}
function startPlan(payload) {
  const plan = lastPlan;
  if (!plan || payload.planId !== plan.id || Date.now() - plan.created > 10 * 60 * 1000) fail('Preview expired; scan again.', 409);
  if (operation && ['copying', 'deleting'].includes(operation.state)) fail('A copy is already running.', 409);
  if (plan.mirror && payload.confirm !== 'MIRROR') fail('Type MIRROR to confirm deletion of extra destination files.');
  operation = {
    state: 'copying', copied: 0, deleted: 0, failed: 0,
    skipped: plan.skipCount, completed: 0, totalOperations: plan.copy.length + plan.remove.length,
    totalFiles: plan.total, bytes: 0, plannedBytes: plan.bytes,
    started: Date.now(), cancelled: false, errors: []
  };
  lastPlan = null;
  // Errors are recorded in the progress object; never leave a detached rejection.
  perform(plan).catch(err => {
    operation.failed++;
    operation.errors.push({ file: '(operation)', message: err.message });
    operation.state = 'finished-with-errors'; operation.finished = Date.now();
  });
  return { started: true };
}
async function readBody(req) {
  if (!/^application\/json(?:\s*;|$)/i.test(req.headers['content-type'] || '')) fail('Expected application/json.', 415);
  let body = '';
  for await (const chunk of req) {
    body += chunk.toString('utf8');
    if (body.length > 8192) fail('Request too large.', 413);
  }
  try { return JSON.parse(body); } catch { fail('Invalid JSON.'); }
}
function acceptHost(req) {
  return req.headers.host === `${HOST}:${PORT}`;
}
const server = http.createServer(async (req, res) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'");
  try {
    if (!acceptHost(req)) return respond(res, 403, { error: 'Invalid host.' });
    const url = new URL(req.url, ORIGIN);
    if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/parallel-folder-sync.html')) {
      const html = await fs.readFile(CLIENT_FILE);
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Content-Length': html.length, 'Cache-Control': 'no-store' });
      return res.end(html);
    }
    if (req.method === 'GET' && url.pathname === '/api/status') return respond(res, 200, { helper: true, token: SESSION_TOKEN, node: process.version, progress: operation || progress });
    if (req.method !== 'POST' || !['/api/plan', '/api/start', '/api/progress', '/api/cancel'].includes(url.pathname)) return respond(res, 404, { error: 'Not found.' });
    if (req.headers.origin !== ORIGIN || req.headers['x-dsl-token'] !== SESSION_TOKEN) return respond(res, 403, { error: 'Origin or token rejected.' });
    const body = await readBody(req);
    if (url.pathname === '/api/plan') return respond(res, 200, await createPlan(body));
    if (url.pathname === '/api/start') return respond(res, 200, startPlan(body));
    if (url.pathname === '/api/cancel') {
      if (operation && ['copying','deleting'].includes(operation.state)) operation.cancelled = true;
      return respond(res, 200, { cancellationRequested: true });
    }
    if (url.pathname === '/api/progress') return respond(res, 200, operation || progress);
  } catch (err) {
    const code = err.status || (['ENOENT','ENOTDIR'].includes(err.code) ? 400 : 500);
    if (code === 500) console.error('[helper]', err);
    return respond(res, code, { error: code === 500 ? 'Operation failed. See local server log.' : err.message });
  }
});

function isUpdatedNode() {
  const [major, minor, patch] = process.versions.node.split('.').map(Number);
  if (major === 22) return minor > 23 || (minor === 23 && patch >= 4);
  if (major === 24) return minor > 22 || (minor === 22 && patch >= 1);
  if (major === 26) return minor > 12 || (minor === 12 && patch >= 1);
  return major > 26;
}
if (!isUpdatedNode()) console.warn('\nSECURITY WARNING: Your Node.js runtime may be missing the October 2026 Brotli fixes. Upgrade to a supported patched Node.js release (e.g. 24.22.2+). This helper does not decode Brotli.\n');
server.listen(PORT, HOST, () => console.log(`DSL Parallel Folder Sync: ${ORIGIN}\nPrivate localhost helper. Press Ctrl+C to stop.`));
