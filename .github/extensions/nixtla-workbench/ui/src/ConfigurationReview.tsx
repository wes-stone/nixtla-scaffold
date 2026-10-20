import { useState } from "react";
import { number, text } from "./api";
import type { Row } from "./api";
import { AnalysisPlot } from "./AnalysisPlot";
import { ErrorNotice, Table } from "./components";
import { EvidenceStatus, EvidenceTable, useEvidence } from "./Evidence";
import type { AnalysisProps } from "./Evidence";
import { savedModel } from "./modelSelection";
import { AnomalyReview } from "./AnomalyReview";
import { NativeFittedReview } from "./NativeFittedReview";

const files = ["series", "seasonality", "phases", "models"].map(name => `audit/configuration_${name}.csv`);
const labels: Record<string, string> = {
  season_length: "Seasonal periods", fourier_order: "Fourier order", multiplicative: "Multiplicative requested",
  model: "Structure / search space", damped: "Damping", phi: "Damping parameter", test_size: "Inner validation horizon",
  n_windows: "Inner validation windows", step_size: "Inner spacing", metric: "Inner search loss",
  window_size: "Averaging window", seasonal_period: "Selected seasonal periods", seasonality_weights: "Recent seasonal weighting",
  smoother: "Exponential residual smoother", ma: "Residual moving average",
};
function object(value: unknown): Row {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Row : {};
}
function settings(value: unknown): { rows: Row[]; error: string } {
  if (value === undefined || value === null || value === "") return { rows: [], error: "" };
  try {
    const parsed: unknown = typeof value === "string" ? JSON.parse(value) : value;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Expected a configuration object");
    return { rows: Object.entries(parsed).map(([key, value]) => ({
      setting: labels[key] ?? key.replaceAll("_", " "),
      value: value === null ? "Native automatic/default" : typeof value === "object" ? JSON.stringify(value) : String(value),
    })), error: "" };
  } catch (error) {
    return { rows: [], error: `Saved configuration could not be read: ${error instanceof Error ? error.message : String(error)}` };
  }
}

export function ConfigurationReview({ document: doc, view, update, inspect, dark }: AnalysisProps) {
  const [refresh, setRefresh] = useState(0);
  const available = view.artifacts.includes(files[0]);
  const scoped = Boolean(doc.series) || view.series.length <= 1;
  const result = useEvidence(doc, available && scoped ? files : [], {}, refresh);
  const series = result.data[files[0]]?.rows[0];
  const models = result.data[files[3]]?.rows ?? [];
  const requested = doc.analysis.previewModel || savedModel(view.rows);
  const model = models.find(row => row.model === requested) ?? models.find(row => row.is_selected === true) ?? models[0];
  const constructor = settings(model?.settings), fitted = settings(model?.selected_native_config);
  const evaluation = object(view.spec.evaluation), transform = object(view.spec.transform), calendar = object(view.spec.calendar);
  const season = number(series?.primary_period) ?? number(view.profile.season_length);
  const phases = (result.data[files[2]]?.rows ?? []).filter(row => number(row.period) === season).map(row => ({
    ...row, position: season === 7 && view.profile.freq === "D"
      ? `${text(row.phase)} · ${new Date(text(row.example_date).slice(0, 10)).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" })}`
      : `Phase ${text(row.phase)}`,
  }));
  if (!scoped) return <section aria-label="Forecast configuration"><h2>Forecast configuration</h2>
    <p className="notice">Select one series using the series control above to inspect its own seasonal patterns and fitted settings. Different series are not pooled into one configuration diagnosis.</p></section>;
  return <section aria-label="Forecast configuration">
    <header className="section-heading"><div><h2>Forecast configuration</h2>
      <p>Separate the settings we supplied, the patterns observed, and the choices the native model actually made.</p></div>
      <button onClick={() => setRefresh(value => value + 1)}>Reload configuration evidence</button></header>
    <Table title="Resolved modeling decisions" rows={[
      { decision: "Observation calendar", setting: text(view.profile.freq), basis: view.spec.freq ? "Explicit calendar" : "Inferred timestamp cadence" },
      { decision: "Primary seasonal period", setting: season === null ? "Unavailable" : `${season} observations`,
        basis: series ? `${text(series.period_source).replaceAll("_", " ")}; ${text(series.primary_evidence).replaceAll("_", " ")}` : "Saved period; data support was not recorded here" },
      { decision: "Training history", setting: evaluation.training_window ? `${text(evaluation.training_window)} most recent observations at every origin` : "Full available history at each origin",
        basis: series ? `${text(series.history_rows)} observations in this diagnostic; ${text(series.available_history_rows)} available` : "Saved specification" },
      { decision: "External target transform", setting: text(transform.target) || "Not recorded", basis: "Separate from internal model transformations" },
      { decision: "Enhanced calendar features", setting: calendar.enabled === true ? "Enabled for native MLForecast" : "Not enabled",
        basis: `Holiday calendar: ${text(calendar.holiday_calendar) || "not recorded"}. No geography is inferred; no extra forecast multiplier.` },
      { decision: "Forecast selection objective", setting: text(object(view.spec.selection_policy).objective) || "Not recorded", basis: "Outer backtests; native inner search can optimize a different criterion" },
      { decision: "Configuration admission", setting: text(object(view.spec.selection_policy).configuration_guard) || "Legacy warning-only",
        basis: "Exclude prevents flagged candidates from becoming the saved selection; their original paths and errors remain available." },
    ]} columns={["decision", "setting", "basis"]} inspect={inspect}/>
    {!available ? <p className="notice" role="status">This saved version predates configuration evidence. Its original spec remains visible, but fitted settings and seasonal-shape diagnostics are unavailable. Opening this page does not reconstruct evidence, refit models or change the forecast.</p> : <>
      <EvidenceStatus result={result}/>
      {series && <>
        <h3>What the data says</h3>
        <p className={series.shape_changed === true ? "notice" : "footnote"}>{text(series.recommendation)}</p>
        <p className="footnote">CoreForecast detected period: {series.detected_period === null ? "unavailable" : number(series.detected_period) === 0 ? "none in the bounded search" : text(series.detected_period)}.
          This is a diagnostic, not an automatic override. Calendar priors and complete cycles alone do not prove seasonality.</p>
        <Table title="Seasonal-period evidence" rows={result.data[files[1]]?.rows ?? []}
          columns={["period", "complete_cycles", "status", "strength_proxy", "shape_drift", "drift_threshold", "shape_changed"]} inspect={inspect} scale={1}/>
        {!!phases.length && <AnalysisPlot title="Seasonal shape · early versus recent cycles" rows={phases} x="position"
          fields={["early_index", "recent_index"]} fieldLabels={{ early_index: "Early 8 cycles", recent_index: "Recent 8 cycles" }}
          kind="bar" dark={dark} scale={1}/>}
        <p className="footnote">Each cycle is divided by its own mean before comparing its phases. Early and recent blocks contain eight non-overlapping complete cycles each. Strength and drift are descriptive heuristics, not statistical significance tests.</p>
      </>}
      {!!models.length && <>
        <h3>Inside the model</h3>
        <label>Configuration model<select value={text(model?.model)} onChange={event => update({ analysis: { previewModel: event.target.value } })}>
          {models.map(row => <option key={text(row.model)} value={text(row.model)}>{text(row.model)}{row.is_selected === true ? " · saved selection" : ""}</option>)}
        </select></label>
        <p>{text(model?.estimator_class)} · {text(model?.fit_status).replaceAll("_", " ")}</p>
        {model?.review_required === true && <p className="notice" role="status">Configuration review: changing seasonal shape and a next-cycle change outside recent experience. {model.selection_eligible === false ? "Excluded from selection under the saved policy; original predictions are retained for inspection." : model.admission_policy === "warn" ? "Warning-only saved policy: this candidate was not excluded. Its forecast is not clipped." : "The delivered path requires review after forecast adjustments. Original admission and unmodified candidate predictions remain available."}</p>}
        <Table title="Model behavior and comparison" rows={[
          { observation: "Internal transform", value: text(model?.internal_transform).replaceAll("_", " ") || "Unavailable" },
          { observation: "Actual multiplicative fit", value: model?.fitted_multiplicative === null ? "Not reported" : text(model?.fitted_multiplicative) },
          { observation: "Fitted structure", value: text(model?.fitted_method) || "Not reported; do not infer it from the model name" },
          { observation: "Native search evidence", value: text(model?.native_config_status).replaceAll("_", " ") },
          { observation: "Next-cycle change", value: number(model?.next_cycle_change_pct) === null ? "Unavailable" : `${Number(model?.next_cycle_change_pct).toFixed(2)}%` },
          { observation: "Recent cycle-change range", value: number(model?.observed_change_p05_pct) === null ? "Unavailable" : `${Number(model?.observed_change_p05_pct).toFixed(2)}% to ${Number(model?.observed_change_p95_pct).toFixed(2)}% (5th–95th percentiles)` },
        ]} columns={["observation", "value"]} inspect={inspect}/>
        <p className="boundary-note">The recent change range is a plausibility comparator, not a prediction interval, growth cap or business constraint. Admission includes the no-change reference; it does not guarantee accuracy. Unknown units and events remain unknown.</p>
        {constructor.error && <ErrorNotice error={constructor.error}/>}
        {!!constructor.rows.length && <Table title="Native constructor settings" rows={constructor.rows} columns={["setting", "value"]} inspect={inspect}/>}
        {fitted.error && <ErrorNotice error={fitted.error}/>}
        {!!fitted.rows.length && <Table title="Selected native search configuration" rows={fitted.rows} columns={["setting", "value"]} inspect={inspect}/>}
        <NativeFittedReview document={doc} view={view} update={update} inspect={inspect} dark={dark} refresh={refresh} model={text(model?.model)}/>
      </>}
      <details><summary>Complete configuration evidence and exports</summary>
        <EvidenceTable document={doc} name={files[3]} title="Per-model configuration receipts" inspect={inspect} scale={1}/>
        <EvidenceTable document={doc} name={files[0]} title="Series configuration diagnostics" inspect={inspect} scale={1}/>
        <EvidenceTable document={doc} name={files[1]} title="Seasonal-period diagnostic receipts" inspect={inspect} scale={1}/>
        <EvidenceTable document={doc} name={files[2]} title="Seasonal phase values" inspect={inspect} scale={1}/>
      </details>
    </>}
    <AnomalyReview document={doc} view={view} update={update} inspect={inspect} dark={dark} refresh={refresh}/>
    <button onClick={() => inspect({ specification: view.spec, configuration_artifact: available ? "audit/configuration.json" : "unavailable" })}>Inspect original specification</button>
    <p className="footnote">Configuration changes are explicit, frozen experiment treatments. Historical comparisons do not authorize adoption or establish untouched confirmation.</p>
  </section>;
}
