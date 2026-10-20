import { AnalysisPlot } from "./AnalysisPlot";
import { Table } from "./components";
import { EvidenceStatus, EvidenceTable, useEvidence } from "./Evidence";
import type { AnalysisProps } from "./Evidence";
import { scales } from "./format";

const summaryFile = "audit/anomaly_series.csv", pointFile = "audit/anomaly_points.csv";

export function AnomalyReview({ document: doc, view, inspect, dark, refresh }: AnalysisProps & { refresh: number }) {
  const available = view.artifacts.includes(summaryFile) && view.artifacts.includes(pointFile);
  const evidence = useEvidence(doc, available ? [summaryFile, pointFile] : [], {}, refresh);
  const summaries = (evidence.data[summaryFile]?.rows ?? []).filter(row => row.stage === "future");
  const points = (evidence.data[pointFile]?.rows ?? []).filter(row => row.stage === "future");
  const flagged = points.filter(row => row.flagged === true);
  return <section aria-label="Calendar and anomaly review">
    <h3>Calendar and anomaly review</h3>
    <p>Unusual usage is not necessarily bad data. Raw actuals are preserved; only a separately evaluated training recipe can use a proposed replacement.</p>
    {!available ? <p className="empty-inline">Anomaly evidence was not recorded for this saved version. Opening it does not run detection or change its forecast.</p> : <>
      <EvidenceStatus result={evidence}/>
      {!evidence.loading && !evidence.error && !summaries.length && <p className="empty-inline">Detection was off for this saved run; no anomaly assessment is available.</p>}
      {!!summaries.length && <Table title="Training review at the forecast cutoff" rows={summaries}
        columns={["cutoff", "status", "mode", "review_rows", "flagged", "eligible", "adjusted", "annual_effect_scope"]} inspect={inspect} scale={1}/>}
      {!!points.length && <AnalysisPlot title="Observed usage and seasonal reference" rows={points} x="ds"
        fields={["observed_y", "expected_y", "modeling_y"]}
        fieldLabels={{ observed_y: "Observed usage", expected_y: "Seasonal reference", modeling_y: "Training copy" }}
        kind="line" dark={dark} scale={scales[doc.analysis.scale]}/>}
      <p className="boundary-note">The seasonal reference is a robust diagnostic, not a forecast interval or a verified correction. The latest week, declared holidays and adjacent days, sustained episodes, and repeated weekday deviations are protected. Unknown holidays and outage causes remain unknown.</p>
      {!!flagged.length && <Table title="Flagged observations and decisions" rows={flagged}
        columns={["ds", "observed_y", "proposed_y", "score", "reason", "eligible", "applied"]} inspect={inspect} scale={1}/>}
      {!!points.length && !flagged.length && <p className="footnote">No observations crossed both declared detection thresholds in this review window. This does not certify that the source is error-free.</p>}
      <details><summary>Complete anomaly evidence and exports</summary>
        <EvidenceTable document={doc} name={summaryFile} title="Origin-local anomaly receipts" inspect={inspect} scale={1}/>
        <EvidenceTable document={doc} name={pointFile} title="Anomaly decisions in saved reporting units" inspect={inspect} scale={1}/>
      </details>
    </>}
    <p className="footnote">Enhanced calendar features are learned inside native MLForecast recipes, not applied as extra multipliers to other models. Holiday geography is never inferred. Adjusted recipes need intervals calibrated on unchanged outer actuals, not cleaned residuals.</p>
    <button onClick={() => inspect({ anomaly: view.spec.anomaly ?? "Not recorded", calendar: view.spec.calendar ?? "Not recorded" })}>Inspect calendar and anomaly policy</button>
  </section>;
}
