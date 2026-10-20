import { useEffect, useState } from "react";
import { call, failure, runtimeSchema } from "./api";
import type { RuntimeCapabilities as Receipt } from "./api";

const labels = {
  not_installed: "Not installed", importable: "Import passed", import_failed: "Import failed",
  probe_failed: "Probe failed", timeout: "Probe timed out",
};

export function RuntimeCapabilities() {
  const [open, setOpen] = useState(false);
  const [revision, setRevision] = useState(0);
  const [receipt, setReceipt] = useState<Receipt>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setBusy(true);
    setError("");
    void call("runtime_capabilities", { forceRefresh: revision > 0 }, runtimeSchema, controller.signal)
      .then(result => { if (!controller.signal.aborted) setReceipt(result); })
      .catch(reason => { if (!controller.signal.aborted) setError(failure(reason)); })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [open, revision]);
  return <details className="runtime-capabilities" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Runtime capabilities</summary>
    <p>Separate from the models in this saved run. Checking imports never fits or installs anything.</p>
    {busy && <p role="status">Checking the selected environment…{receipt && " Previous result shown below."}</p>}
    {error && <p role="alert">{error} {receipt && "The previous result is stale."}</p>}
    {receipt && <>
      <p><strong>{receipt.registered_recipes} registered recipes</strong> · {receipt.importable_recipes} have importable dependencies. Not a count of successful fits.</p>
      <dl>
        <dt>Interpreter</dt><dd>{receipt.python}</dd>
        <dt>Versions</dt><dd>Python {receipt.python_version} · scaffold {receipt.package_version ?? "unknown"}</dd>
        <dt>Checked</dt><dd>{new Date(receipt.probed_at).toLocaleString()}</dd>
        <dt>Probe</dt><dd>{receipt.probe_seconds === undefined ? "Duration unavailable" : `${receipt.probe_seconds}s`}{receipt.cached && " · cached for up to 60 seconds; refresh to recheck"}</dd>
        <dt>Fingerprint</dt><dd>{receipt.fingerprint}</dd>
      </dl>
      {receipt.engines.map(engine => <details key={engine.package}>
        <summary>{engine.package} {engine.version ?? ""} · {labels[engine.status]}</summary>
        <p>{engine.compatibility === "version_matched" ? "Version matches the tested adapter; fit success still depends on the data."
          : engine.version ? "This version is not verified against the adapter." : "Not available in this interpreter."}</p>
        {engine.error && <p role="alert">{engine.error}</p>}
        {engine.limitations.map(limitation => <p key={limitation}>{limitation}</p>)}
        <p>{engine.recipes.length ? `${engine.recipes.length} registered recipes: ${engine.recipes.join(", ")}`
          : engine.integration === "custom_adapter" ? "Custom/research adapter only; not part of the built-in recipe count."
            : engine.integration === "experimental_intervals" ? "SplitConformal and NexCP are integrated experimental interval methods, not point models. Calibration support is evaluated per run."
              : engine.integration === "reconciliation_adapter" ? "MinTrace OLS and structural WLS reconciliation adapter; no additional point-model recipes."
                : "No built-in point-model recipes; see the integration status and limitations."}</p>
      </details>)}
      <p>Dataset eligibility is not evaluated here. Actual successes, partial results and failures belong to the saved run's evidence.</p>
    </>}
    <button className="quiet" disabled={busy} onClick={() => setRevision(value => value + 1)}>Refresh runtime status</button>
  </details>;
}
