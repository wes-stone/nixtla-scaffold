import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import test from "node:test";
import { configuration, createService } from "./server.mjs";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";

const { runtime, contract } = await configuration();

function readFixture() {
  const workers = [];
  const service = createService({ workspace: ".", python: runtime.python, contract, spawnWorker() {
    const child = Object.assign(new EventEmitter(), {
      exitCode: null, stdin: new PassThrough(), stdout: new PassThrough(), stderr: new PassThrough(),
      finish() {
        this.exitCode = 0;
        this.stdout.emit("data", Buffer.from('{"result":{"initialized":true}}'));
        this.emit("close", 0);
      },
    });
    workers.push(child);
    return child;
  } });
  return { service, workers };
}

test("expanded-folder reads queue without exceeding four concurrent workers", async () => {
  const { service, workers } = readFixture();
  try {
    const result = Promise.all(Array.from({ length: 10 }, () => service.call("catalog")));
    assert.equal(workers.length, 4);
    for (const expected of [8, 10, 10]) {
      for (const child of workers.filter(child => child.exitCode === null)) child.finish();
      await new Promise(resolve => setImmediate(resolve));
      assert.equal(workers.length, expected);
      assert.ok(workers.filter(child => child.exitCode === null).length <= 4);
    }
    assert.equal((await result).length, 10);
  } finally { await service.close(); }
});

test("runtime probes coalesce, cache briefly, and refresh explicitly", async () => {
  const { service, workers } = readFixture();
  try {
    const a = service.call("runtime_capabilities");
    const b = service.call("runtime_capabilities");
    assert.equal(workers.length, 1);
    workers[0].finish();
    await Promise.all([a, b]);
    assert.equal((await service.call("runtime_capabilities")).cached, true);
    assert.equal(workers.length, 1);
    const refreshed = service.call("runtime_capabilities", { forceRefresh: true });
    assert.equal(workers.length, 2);
    workers[1].finish();
    await refreshed;
  } finally { await service.close(); }
});

test("read queue is bounded and shutdown rejects every unstarted read", async () => {
  const { service, workers } = readFixture();
  const pending = Promise.allSettled(Array.from({ length: 68 }, () => service.call("catalog")));
  await assert.rejects(service.call("catalog"), /queue is full/);
  const closing = service.close();
  for (const child of workers) child.finish();
  await closing;
  const results = await pending;
  assert.equal(workers.length, 4);
  assert.equal(results.filter(result => result.status === "fulfilled").length, 4);
  assert.equal(results.filter(result => result.status === "rejected").length, 64);
  await assert.rejects(service.call("catalog"), /shutting down/);
});

async function fixture(t) {
  const workspace = await mkdtemp(join(tmpdir(), "nixtla-service-test-"));
  const workers = [];
  const service = createService({
    workspace, python: runtime.python, contract,
    spawnWorker(command, args, options) {
      if (!args.includes("--job-id") || args.includes("--terminal-status")) return spawn(command, args, options);
      const child = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], options);
      workers.push(child);
      return child;
    },
  });
  t.after(async () => {
    await service.close();
    await rm(workspace, { recursive: true });
  });
  await service.call("initialize");
  const source = join(workspace, "source.csv");
  const rows = ["ds,y", ...Array.from({ length: 48 }, (_, i) =>
    `${2020 + Math.floor(i / 12)}-${String(i % 12 + 1).padStart(2, "0")}-01,${100 + i}`)];
  await writeFile(source, rows.join("\n"));
  const { sourceId } = await service.call("profile", { path: source, horizon: 3 });
  return { service, workspace, workers, request: {
    kind: "forecast", sourceId, requestKey: "lifecycle", spec: { horizon: 3, model_policy: "baseline" },
  } };
}

async function terminal(service, jobId) {
  for (let attempt = 0; attempt < 30; attempt++) {
    const job = await service.call("job", { jobId });
    if (["cancelled", "interrupted", "failed", "completed"].includes(job.status)) return job;
    await delay(100);
  }
  assert.fail("The owned worker stopped without a durable terminal receipt.");
}

test("Stop terminates its owned worker and preserves a cancelled receipt", { timeout: 60000 }, async t => {
  const { service, workers, request } = await fixture(t);
  const job = await service.call("start_job", request);
  const duplicate = await service.call("start_job", request);
  assert.equal(job.jobId, duplicate.jobId);
  assert.equal(workers.length, 1);
  assert.ok(workers[0].pid);
  await service.call("cancel_job", { jobId: job.jobId });
  assert.equal((await terminal(service, job.jobId)).status, "cancelled");
  assert.ok(workers[0].exitCode !== null || workers[0].signalCode !== null);
});

test("explicit resume launches one worker under the remaining original budget", { timeout: 60000 }, async t => {
  const { service, workers, request } = await fixture(t);
  const original = await service.call("start_job", { ...request, maxWallClockSeconds: 45 });
  await service.call("cancel_job", { jobId: original.jobId });
  const stopped = await terminal(service, original.jobId);
  const remaining = stopped.input.executionBudget.remainingSeconds;
  assert.ok(remaining > 0 && remaining < 45);
  const input = { jobId: original.jobId, requestKey: "explicit-resume" };
  const child = await service.call("resume_job", input);
  assert.ok(child.budgetSeconds <= remaining);
  assert.equal((await service.call("resume_job", input)).created, false);
  assert.equal(workers.length, 2);
  await service.call("cancel_job", { jobId: child.jobId });
  assert.equal((await terminal(service, child.jobId)).status, "cancelled");
});

test("startup consumes a smaller deadline and expiration never resets its budget", { timeout: 60000 }, async t => {
  const { service, workers, request } = await fixture(t);
  const job = await service.call("start_job", { ...request, maxWallClockSeconds: 0.5 });
  const stopped = await terminal(service, job.jobId);
  assert.equal(stopped.status, "interrupted");
  assert.equal(stopped.input.executionBudget.remainingSeconds, 0);
  assert.ok(workers[0].exitCode !== null || workers[0].signalCode !== null);
  await assert.rejects(service.call("resume_job", { jobId: job.jobId, requestKey: "no-reset" }), /No time remains/);
});

test("shutdown settles owned work and cannot launch an admitted queued request afterward", { timeout: 60000 }, async t => {
  const { service, workspace, workers, request } = await fixture(t);
  const first = await service.call("start_job", request);
  const admitted = service.call("start_job", { ...request, requestKey: "closing" });
  const closing = service.close();
  assert.equal(service.close(), closing);
  await assert.rejects(service.call("catalog"), /shutting down/);
  const second = await admitted;
  await closing;
  assert.equal(workers.length, 1);
  assert.ok(workers[0].exitCode !== null || workers[0].signalCode !== null);
  const reopened = createService({ workspace, python: runtime.python, contract });
  try {
    for (const jobId of [first.jobId, second.jobId]) {
      assert.equal((await reopened.call("job", { jobId })).status, "interrupted");
    }
  } finally { await reopened.close(); }
});

test("a killed service leaves recoverable ownership, not a silently replayed job", { timeout: 90000 }, async t => {
  const { service, workspace, request } = await fixture(t);
  const script = join(workspace, "owner.mjs");
  const ready = join(workspace, "job.json");
  await writeFile(script, `
    import { spawn } from "node:child_process";
    import { writeFile, appendFile } from "node:fs/promises";
    import { createService } from ${JSON.stringify(new URL("./server.mjs", import.meta.url).href)};
    const service = createService({
      workspace: ${JSON.stringify(workspace)}, python: ${JSON.stringify(runtime.python)}, contract: ${JSON.stringify(contract)},
      spawnWorker(command, args, options) {
        if (!args.includes("--job-id") || args.includes("--terminal-status")) return spawn(command,args,options);
        const id=args[args.indexOf("--job-id")+1];
        const child = spawn(command, ["-c",
          "import sys,time; from nixtla_scaffold.workbench_jobs import claim_worker; from nixtla_scaffold.workbench_store import WorkbenchStore; claim_worker(WorkbenchStore(sys.argv[1]),sys.argv[2]); time.sleep(120)",
          ${JSON.stringify(workspace)}, id], options);
        child.stderr.on("data", chunk => { void appendFile(${JSON.stringify(join(workspace, "worker-error.txt"))}, chunk); });
        return child;
      }
    });
    await writeFile(${JSON.stringify(ready)}, JSON.stringify(await service.call("start_job", ${JSON.stringify(request)})));
    setInterval(()=>{},1000);
  `);
  const owner = spawn(process.execPath, [script], { stdio: ["ignore", "ignore", "pipe"], windowsHide: true });
  let ownerErrors = "";
  owner.stderr.on("data", chunk => { ownerErrors += chunk.toString(); });
  let job;
  try {
    for (let attempt = 0; attempt < 120; attempt++) {
      try { job = JSON.parse(await readFile(ready, "utf8")); break; }
      catch (error) { if (error.code !== "ENOENT") throw error; }
      await delay(100);
    }
    assert.ok(job);
    for (let attempt = 0; attempt < 20; attempt++) {
      if ((await service.call("job", { jobId: job.jobId })).status === "running") break;
      await delay(100);
    }
    const running = await service.call("job", { jobId: job.jobId });
    if (running.status !== "running") {
      let workerErrors = "";
      try { workerErrors = await readFile(join(workspace, "worker-error.txt"), "utf8"); } catch (error) { if (error.code !== "ENOENT") throw error; }
      assert.fail(JSON.stringify({ running, ownerErrors, workerErrors, exitCode: owner.exitCode }));
    }
    const exited = new Promise(resolve => owner.once("exit", resolve));
    owner.kill("SIGKILL"); await exited;
    const recovered = await service.call("recover_jobs", { approved: true });
    assert.deepEqual(recovered.recovered, [job.jobId]);
    assert.equal((await service.call("job", { jobId: job.jobId })).status, "interrupted");
    assert.equal((await service.call("catalog")).jobs.length, 1);
  } finally {
    if (owner.exitCode === null && owner.signalCode === null) {
      const exited = new Promise(resolve => owner.once("exit", resolve)); owner.kill("SIGKILL"); await exited;
    }
    if (job) await service.call("recover_jobs", { approved: true });
  }
});
