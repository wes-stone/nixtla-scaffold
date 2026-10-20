import { useEffect, useState } from "react";
import { z } from "zod";
import { call, failure } from "./api";
import { EvidenceStatus, EvidenceTable, useEvidence } from "./Evidence";
import type { AnalysisProps } from "./Evidence";
import { ErrorNotice, Table } from "./components";
import { scales } from "./format";

const comparisonSchema = z.object({
  unique_id: z.string(), reference_model: z.string(),
  paired_rows: z.number().int().nonnegative(), planned_later_rows: z.number().int().nonnegative(),
  warmup_rows: z.number().int().nonnegative(), comparison_eligible: z.boolean(),
  metrics: z.array(z.object({ model: z.string(), rmse: z.number().finite().nonnegative().nullable(), mae: z.number().finite().nonnegative().nullable() })),
});
const artifactSchema = z.object({ text: z.string() });
export function EnsembleReview({ document: doc, view, update, inspect }: AnalysisProps) {
  const name = view.artifacts.find(item => item === "appendix/ensemble_comparisons.json" || item === "ensemble_comparisons.json");
  const [state, setState] = useState<{ key: string; rows: z.infer<typeof comparisonSchema>[]; error: string }>();
  const [weightsOpen, setWeightsOpen] = useState(false), [refresh, setRefresh] = useState(0);
  const forecasts = useEvidence(doc, ["appendix/forecast_long.csv"], { models: ["WeightedEnsemble"], limit: 1 });
  const hasForecast = (forecasts.data["appendix/forecast_long.csv"]?.total ?? 0) > 0;
  const key = JSON.stringify([doc.runId, name, refresh]);
  useEffect(() => {
    if (!name) return;
    const controller = new AbortController();
    void call("artifact", { runId: doc.runId, name }, artifactSchema, controller.signal)
      .then(result => {
        const rows = z.array(comparisonSchema).parse(JSON.parse(result.text));
        if (!controller.signal.aborted) setState({ key, rows, error: "" });
      }).catch(error => { if (!controller.signal.aborted) setState({ key, rows: [], error: failure(error) }); });
    return () => controller.abort();
  }, [key]);
  const current = state?.key === key ? state : undefined;
  const paired = current?.rows.find(row => row.unique_id === doc.series);
  const reference = paired?.metrics.find(row => row.model === paired.reference_model);
  const blend = paired?.metrics.find(row => row.model === "WeightedEnsemble");
  const complete = paired?.comparison_eligible === true && paired.paired_rows > 0 && paired.paired_rows === paired.planned_later_rows;
  const delta = complete && reference?.rmse != null && reference.rmse > 0 && blend?.rmse != null
    ? (blend.rmse / reference.rmse - 1) * 100 : null;
  const weightsName = view.artifacts.find(item => item === "audit/model_weights.csv" || item === "model_weights.csv");
  return <section className="ensemble-review" aria-label="Weighted ensemble evidence">
    <header className="section-heading"><h3>Weighted ensemble</h3>
      <button disabled={!hasForecast} onClick={() => update({ view: "Forecast", analysis: { previewModel: "WeightedEnsemble", modelMode: "focus" } })}>View saved blend forecast</button>
    </header>
    <EvidenceStatus result={forecasts} reportPartial={false}/>
    {!forecasts.loading && !forecasts.error && !hasForecast && <p className="empty-inline">No weighted-ensemble forecast path was saved for this series.</p>}
    <p>Separate-cohort candidate, not an automatic winner. Compare it with the saved winner on the same later periods.</p>
    {name && !current && <p role="status">Reading saved paired ensemble results...</p>}
    {current?.error && <><ErrorNotice error={current.error}/><button onClick={() => setRefresh(value => value + 1)}>Retry ensemble evidence</button></>}
    {(!name || (current && !current.error && !paired)) && <p className="empty-inline">No paired weighted-ensemble result was saved for this series. A future path alone is not accuracy evidence. Build and backtest a recipe in Compare.</p>}
    {paired && <>
      {delta !== null && <p className="ensemble-verdict"><strong>{Math.abs(delta).toFixed(1)}% {delta > 0 ? "higher" : delta < 0 ? "lower" : "difference in"} RMSE{delta === 0 ? " versus" : " than"} {paired.reference_model} on matched periods.</strong></p>}
      <p className="footnote">{paired.paired_rows} matched predictions · {paired.warmup_rows} warm-up rows excluded.
        {!complete && " Incomplete paired support; no accuracy ranking is justified."} Retrospective evidence, not independent confirmation.</p>
      <Table title="Same-period ensemble comparison" rows={paired.metrics} columns={["model", "rmse", "mae"]} inspect={inspect} scale={scales[doc.analysis.scale]}/>
    </>}
    <details onToggle={event => setWeightsOpen(event.currentTarget.open)}><summary>Saved component weights</summary>
      <p>Final weights describe the future blend. Historical accuracy must use weights learned from earlier observable outcomes, not these final weights.</p>
      {weightsOpen && (weightsName ? <EvidenceTable document={doc} name={weightsName} title="Weighted ensemble components" inspect={inspect} decimalPlaces={4}/>
        : <p className="empty-inline">Component weights were not saved in this version.</p>)}
    </details>
  </section>;
}
