import { useEffect, useState } from "react";
import { z } from "zod";
import { call, failure, rowSchema, text } from "./api";
import type { AnalysisProps } from "./Evidence";
import { ErrorNotice, Table } from "./components";
import { calendarReceiptSchema, judgmentDraftSchema, judgmentSummary } from "./judgment";
import { formatted, scales } from "./format";

const resultSchema = z.object({
  recipe: z.object({ baseRunId: z.string(), parentRunId: z.string(), series: z.string(), settings: judgmentDraftSchema,
    totals: z.record(z.string().nullable()), hashes: z.array(rowSchema), sourceHash: z.string(),
    calendarLayer: calendarReceiptSchema.nullable().optional(), method: z.string(),
  }).passthrough(),
  rows: z.array(rowSchema), verified: z.boolean(),
});

export function FinalizationPipeline({ document: doc, view, update, inspect }: AnalysisProps) {
  const [data, setData] = useState<z.infer<typeof resultSchema>>(), [error, setError] = useState(""), [busy, setBusy] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setData(undefined); setError("");
    void call("judgment_pipeline", { runId: doc.runId, series: doc.series }, resultSchema, controller.signal)
      .then(setData).catch(err => { if (!controller.signal.aborted) setError(failure(err)); });
    return () => controller.abort();
  }, [doc.runId, doc.series]);
  const replay = async () => {
    setBusy(true); setError("");
    try { setData(await call("judgment_pipeline", { runId: doc.runId, series: doc.series, replay: true }, resultSchema)); }
    catch (err) { setError(failure(err)); }
    finally { setBusy(false); }
  };
  const recipe = data?.recipe, layer = recipe?.calendarLayer, scale = scales[doc.analysis.scale];
  return <section className="finalization-pipeline" aria-label="Saved finalization pipeline">
    <header className="section-heading"><div><h2>Saved finalization pipeline</h2><p>Reproduce the same model, calendar assumptions and final judgment without fitting a model.</p></div></header>
    {error && <ErrorNotice error={error}/>}
    {!data && !error && <p role="status">Loading the saved recipe...</p>}
    {recipe && <>
      <p><strong>{recipe.settings.label}</strong> · {recipe.series} · {recipe.settings.reason}</p>
      <p className="footnote">This is the immutable saved version, not unsaved worksheet edits. Statistical scores and intervals do not transfer to these values.</p>
      <div className="pipeline">{[
        { label: "1. Chosen model", detail: recipe.settings.model || "Saved selected forecast", value: { baseRunId: recipe.baseRunId, hashes: recipe.hashes, sourceHash: recipe.sourceHash } },
        { label: "2. Calendar layer", detail: layer ? (layer.settings.holdTotal ? "Rephase · model total held" : "Activity-day overlay · total can change") : "Off · model unchanged", value: layer ?? { enabled: false } },
        { label: "3. Final judgment", detail: judgmentSummary(recipe.settings, text(view.spec.unit_label)), value: recipe.settings },
        { label: "4. Saved version", detail: recipe.settings.label, value: { runId: doc.runId, parentRunId: recipe.parentRunId, totals: recipe.totals } },
      ].map(stage => <button key={stage.label} onClick={() => inspect(stage.value)}><strong>{stage.label}</strong><small>{stage.detail}</small></button>)}</div>
      <dl className="judgment-totals">
        <div><dt>Model sum</dt><dd>{formatted(recipe.totals.model_yhat, "yhat", scale)}</dd></div>
        {layer && <div><dt>After calendar</dt><dd>{formatted(recipe.totals.calendar_yhat, "yhat", scale)}</dd></div>}
        <div><dt>Final sum</dt><dd>{formatted(recipe.totals.adjusted_yhat, "yhat", scale)}</dd></div>
      </dl>
      <div className="toolbar"><button disabled={busy} onClick={() => void replay()}>{busy ? "Replaying recipe..." : "Replay saved recipe"}</button>
        <button disabled={busy} onClick={() => update({ view: "Adjust" })}>Open adjustment worksheet</button></div>
      {data?.verified && <p role="status">Replay verified: source fingerprints and every saved comparison value match. No model was fitted or version changed.</p>}
      <p className="footnote">Replay uses the pinned original baseline. To use new actuals, refresh that baseline separately, then make an explicit new adjustment version; obsolete month overrides are never silently rolled forward.</p>
      <Table title="Finalization stage comparison" rows={data?.rows ?? []}
        columns={["ds", "model_yhat", ...(layer ? ["calendar_yhat"] : []), "adjusted_yhat", ...(layer ? ["calendar_delta", "judgment_delta"] : []), "delta"]}
        columnLabels={{ ds: "Period", model_yhat: "Chosen model", calendar_yhat: "After calendar", calendar_delta: "Calendar change",
          judgment_delta: "Final tweaks", adjusted_yhat: "Final forecast", delta: "Total change" }}
        inspect={inspect} scale={scale} stickyColumn="ds"/>
      {layer && <details><summary>Calendar days & factor bridge</summary>
        <p className="footnote">Month length × day mix × holiday effect gives the exposure ratio. Total scaling preserves the model sum when selected. Factors are shown to six decimals; raw copy and exports retain the saved precision.</p>
        <Table title="Calendar layer details" rows={data?.rows ?? []} stickyColumn="ds"
          columns={["ds", "calendar_days", "weekdays", "weekends", "holiday_loss", "effective_days", "month_length_factor", "weekday_weekend_factor", "holiday_factor", "calendar_factor", "calendar_scaling_factor"]}
        columnLabels={{ ds: "Period", model_yhat: "Chosen model", calendar_days: "Days", weekdays: "Weekdays", weekends: "Weekends",
          holiday_loss: "Holiday loss (days)", effective_days: "Effective days", calendar_factor: "Exposure ratio",
          month_length_factor: "Month length ×", weekday_weekend_factor: "Day mix ×", holiday_factor: "Holiday ×", calendar_scaling_factor: "Total scaling ×" }}
          inspect={inspect} decimalPlaces={6}/></details>}
      <details><summary>Complete saved recipe & source hashes</summary><p>{recipe.method}</p>{layer && <p>{layer.method}</p>}
        <pre className="file-preview" tabIndex={0}>{JSON.stringify(recipe, null, 2)}</pre></details>
    </>}
  </section>;
}
