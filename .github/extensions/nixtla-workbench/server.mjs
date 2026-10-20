import { spawn, execFile } from "node:child_process";
import { createServer } from "node:http";
import { randomBytes, timingSafeEqual } from "node:crypto";
import { readFile, access } from "node:fs/promises";
import { dirname, resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";

const directory = dirname(fileURLToPath(import.meta.url));
const repo = resolve(directory, "../../..");
const writes = new Set(["initialize", "profile", "import_run", "save_state", "save_note", "start_job", "resume_job", "cancel_job", "export_values", "export_evidence", "explore_save", "explore_export", "calendar_export", "judgment_save", "judgment_export", "recover_jobs", "prepare_pipeline", "prepare_forecast", "prepare_external"]);

export async function configuration() {
  let runtime;
  try {
    runtime = JSON.parse(await readFile(resolve(directory, "runtime.json"), "utf8"));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    runtime = { python: resolve(repo, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python") };
  }
  await access(runtime.python);
  if (runtime.packageVersion) {
    const installed = await new Promise((done, fail) => execFile(
      runtime.python, ["-c", "from importlib.metadata import version; print(version('nixtla-scaffold'))"],
      { windowsHide: true, timeout: 15000 }, (error, stdout) => error ? fail(error) : done(stdout.trim()),
    ));
    if (installed !== runtime.packageVersion) throw new Error("Canvas and Python package versions differ. Reinstall the canvas from the matching runtime.");
  }
  let contractPath = resolve(directory, "workbench_contract.json");
  try { await access(contractPath); }
  catch (error) {
    if (error.code !== "ENOENT") throw error;
    contractPath = resolve(repo, "src/nixtla_scaffold/workbench_contract.json");
  }
  return { runtime, contract: JSON.parse(await readFile(contractPath, "utf8")) };
}

export function createService({ workspace, python, contract, spawnWorker = spawn }) {
  const children = new Map();
  const settlements = new Set();
  let writer = Promise.resolve();
  let closed = false;
  let closing = false;
  let closingPromise;
  let reads = 0;
  const readQueue = [];
  let runtimeReceipt, runtimePending, runtimeChecked = 0;

  const workerArgs = (extra = []) => [
    "-c", "from nixtla_scaffold.cli import main; raise SystemExit(main())",
    "workbench", "worker", "--workspace", workspace, ...extra,
  ];

  function run(input, { extra = [], jobId, timeout = 60000 } = {}) {
    if (closed) return Promise.reject(new Error("Workbench is closed."));
    return new Promise((resolveResult, reject) => {
      const child = spawnWorker(python, workerArgs(extra), {
        cwd: workspace, shell: false, windowsHide: true,
        env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONDONTWRITEBYTECODE: "1", NIXTLA_WORKBENCH_OWNER_PID: String(process.pid) },
        stdio: ["pipe", "pipe", "pipe"],
      });
      const chunks = [];
      let bytes = 0, stopped = false, failure, settled = false;
      const key = jobId ?? Symbol("read");
      const finish = (error, result) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        children.delete(key);
        if (error) reject(error); else resolveResult(result);
      };
      const stop = async () => {
        if (stopped || child.exitCode !== null) return;
        stopped = true;
        if (!child.pid) return;
        if (process.platform === "win32") {
          await new Promise((done, fail) => execFile(
            "taskkill", ["/PID", String(child.pid), "/T", "/F"],
            { windowsHide: true }, (error) => error && child.exitCode === null ? fail(error) : done(),
          ));
        } else child.kill("SIGTERM");
      };
      children.set(key, { child, stop });
      const timer = setTimeout(() => {
        failure = new Error(jobId ? "The local analysis budget expired." : "Local operation timed out.");
        void stop().catch(error => finish(error));
      }, timeout);
      child.on("error", finish);
      child.stdin.on("error", error => {
        failure = error;
        void stop().catch(finish);
      });
      child.stdout.on("data", chunk => {
        bytes += chunk.length;
        if (bytes > 8 * 1024 * 1024) {
          failure = new Error("Response exceeds 8 MiB. Narrow the evidence scope.");
          void stop().catch(finish);
        } else chunks.push(chunk);
      });
      // Do not reflect arbitrary worker stderr (which can contain source values)
      // into the browser or the extension protocol.
      child.stderr.on("data", () => {});
      child.on("close", code => {
        if (failure || stopped) return finish(failure ?? new Error("Owned worker stopped."));
        try {
          const envelope = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          if (code !== 0 || envelope.error) {
            const error = new Error(envelope.error?.message ?? "Worker failed without a readable result.");
            error.code = envelope.error?.code ?? "worker_failed";
            finish(error);
          } else if (!Object.hasOwn(envelope, "result")) finish(new Error("Worker result is missing."));
          else finish(null, envelope.result);
        } catch { finish(new Error("Worker returned invalid JSON. No successful result was recorded.")); }
      });
      child.stdin.end(input ? JSON.stringify(input) : "");
    });
  }

  async function execute(operation, input) {
    if (!Object.hasOwn(contract.operations, operation)) throw new Error("Unknown operation.");
    if (operation === "runtime_capabilities") {
      if (!input?.forceRefresh && runtimeReceipt && Date.now() - runtimeChecked < 60000) {
        return { ...runtimeReceipt, cached: true };
      }
      if (!runtimePending) {
        runtimePending = run({ operation, input }).then(receipt => {
          runtimeReceipt = receipt;
          runtimeChecked = Date.now();
          return receipt;
        }).finally(() => { runtimePending = undefined; });
      }
      return runtimePending;
    }
    const result = await run({ operation, input });
    if (["start_job", "resume_job"].includes(operation) && result.created) {
      const jobId = result.jobId;
      if (closing) {
        await run(null, { extra: ["--job-id", jobId, "--terminal-status", "interrupted"] });
        return { ...result, status: "interrupted" };
      }
      // The parent owns the process for the whole session, not one HTTP request.
      const pending = run(null, { extra: ["--job-id", jobId], jobId, timeout: Math.min(900, result.budgetSeconds ?? 900) * 1000 }).catch(async error => {
        if (closed) return;
        try {
          const state = await run({ operation: "job", input: { jobId } });
          if (["queued", "running", "cancel_requested"].includes(state.status)) {
            await run(null, { extra: ["--job-id", jobId, "--terminal-status",
              state.status === "cancel_requested" ? "cancelled" : "interrupted"] });
          }
        } catch (settleError) {
          process.stderr.write(`Workbench job requires reconciliation: ${settleError.message}\n`);
        }
        process.stderr.write(`Workbench job stopped: ${error.message}\n`);
      });
      settlements.add(pending);
      void pending.finally(() => settlements.delete(pending));
    }
    if (operation === "cancel_job" && result.status === "cancel_requested") {
      const owned = children.get(input.jobId);
      if (!owned) {
        return { ...result, warning: "No worker is owned by this session. Reconcile the interrupted job; cancellation is not confirmed." };
      }
      await owned.stop();
    }
    return result;
  }

  function drainReads() {
    while (!closing && reads < 4 && readQueue.length) {
      const request = readQueue.shift();
      reads++;
      void execute(request.operation, request.input).then(request.resolve, request.reject).finally(() => {
        reads--;
        drainReads();
      });
    }
  }

  return {
    async call(operation, input = {}) {
      if (closing) throw new Error("Workbench is shutting down; no new operation was started.");
      if (writes.has(operation)) {
        const pending = writer.then(() => execute(operation, input));
        writer = pending.catch(() => undefined);
        return pending;
      }
      if (readQueue.length >= 64) throw new Error("The evidence queue is full. Collapse unused folders and retry after pending reads complete.");
      return new Promise((resolve, reject) => {
        readQueue.push({ operation, input, resolve, reject });
        drainReads();
      });
    },
    close() {
      if (closingPromise) return closingPromise;
      closing = true;
      for (const request of readQueue.splice(0)) request.reject(new Error("Workbench is shutting down; the queued read was not started."));
      closingPromise = (async () => {
        await writer;
        const jobs = [...children.keys()].filter(key => typeof key === "string");
        const results = await Promise.allSettled([...children.values()].map(entry => entry.stop()));
        for (const result of results) {
          if (result.status === "rejected") process.stderr.write(`Workbench worker cleanup failed: ${result.reason}\n`);
        }
        if (results.every(result => result.status === "fulfilled")) {
          await Promise.allSettled([...settlements]);
          for (const jobId of jobs) {
            try { await run(null, { extra: ["--job-id", jobId, "--terminal-status", "interrupted"] }); }
            catch (error) { process.stderr.write(`Workbench interruption receipt failed: ${error.message}\n`); }
          }
        }
        closed = true;
      })();
      return closingPromise;
    },
  };
}

export async function startServer({ workspace, service, handoff }) {
  const token = randomBytes(32).toString("hex");
  const root = resolve(directory, "ui/dist");
  const manifest = JSON.parse(await readFile(resolve(root, ".vite/manifest.json"), "utf8"));
  const assets = new Set(["/"]);
  for (const entry of Object.values(manifest)) {
    for (const file of [entry.file, ...(entry.css ?? []), ...(entry.assets ?? [])]) {
      if (file && !file.endsWith(".map")) assets.add("/" + file);
    }
  }
  let origin = "";
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, origin);
      if (req.method === "POST" && url.pathname === "/api") {
        const provided = String(req.headers["x-workbench-token"] ?? "");
        if (req.headers.origin !== origin || provided.length !== token.length ||
            !timingSafeEqual(Buffer.from(provided), Buffer.from(token))) {
          res.writeHead(403).end(); return;
        }
        const chunks = [];
        let bytes = 0;
        for await (const chunk of req) {
          bytes += chunk.length;
          if (bytes > 256 * 1024) { res.writeHead(413).end(); return; }
          chunks.push(chunk);
        }
        const { operation, input } = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        let result;
        if (operation === "handoff") {
          if (!handoff) throw new Error("Chat handoff is available only inside the Copilot canvas.");
          if (typeof input?.question !== "string" || !input.question.trim() || input.question.length > 4000) {
            throw new Error("Provide a question of at most 4,000 characters.");
          }
          const context = await service.call("context", { runId: input.runId, series: input.series, document: input.document });
          await handoff({ question: input.question, context });
          result = { sent: true };
        } else result = await service.call(operation, input);
        res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
        res.end(JSON.stringify({ result })); return;
      }
      if (req.method !== "GET" || !assets.has(url.pathname)) { res.writeHead(404).end(); return; }
      const file = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
      const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml" };
      res.writeHead(200, {
        "Content-Type": types[extname(file)] ?? "application/octet-stream",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
      });
      res.end(await readFile(resolve(root, file)));
    } catch (error) {
      if (res.headersSent) { res.end(); return; }
      res.writeHead(400, { "Content-Type": "application/json", "Cache-Control": "no-store" });
      res.end(JSON.stringify({ error: { code: error.code ?? "operation_failed", message: error.message } }));
    }
  });
  await new Promise((done, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", done);
  });
  origin = `http://127.0.0.1:${server.address().port}`;
  return {
    url: `${origin}/#${token}`, workspace,
    close: () => new Promise((done, reject) => server.close(error => error ? reject(error) : done())),
  };
}
