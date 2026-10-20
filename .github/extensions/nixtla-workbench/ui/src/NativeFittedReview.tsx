import { AnalysisPlot } from "./AnalysisPlot";
import { Table } from "./components";
import { EvidenceStatus, EvidenceTable, useEvidence } from "./Evidence";
import type { AnalysisProps } from "./Evidence";
import { scales } from "./format";
import { text } from "./api";
import type { Row } from "./api";

const summaryFile = "audit/native_fitted_summary.csv", pointFile = "audit/native_fitted_points.csv";

export function NativeFittedReview({ document: doc, view, inspect, dark, refresh, model }: AnalysisProps & { refresh: number; model: string }) {
  const available = view.artifacts.includes(summaryFile) && view.artifacts.includes(pointFile);
  const evidence = useEvidence(doc, available && model ? [summaryFile, pointFile] : [], { models: [model] }, refresh);
  const summary = (evidence.data[summaryFile]?.rows ?? []).filter(row => row.stage === "future");
  const points: Row[] = (evidence.data[pointFile]?.rows ?? []).map(row => ({
    ...row, flagged_y: row.flagged === true ? row.training_y : null,
  }));
  return <section aria-label="Native fitted-model anomalies">
    <h3>Native fitted-model anomalies</h3>
    <p>{model || "Selected model"} · Nixtla in-sample fitted values and 99% fitted bands. These are descriptive training diagnostics, not calibrated future intervals or a reason to rewrite actuals.</p>
    {!available ? <p className="empty-inline">Native fitted evidence was not saved for this version. Opening it does not refit models.</p> : <>
      <EvidenceStatus result={evidence}/>
      {!evidence.loading && !evidence.error && !summary.length && <p className="empty-inline">This model adapter did not record native fitted intervals. Select a supported native model above.</p>}
      {!!summary.length && <Table title="Native fitted diagnostic scope" rows={summary}
        columns={["model", "status", "level", "training_rows", "review_rows", "finite_rows", "flagged"]} inspect={inspect} scale={1}/>}
      {summary.filter(row => row.status !== "available").map(row => <p className="empty-inline" key={text(row.model)}>{text(row.reason)}</p>)}
      {!!points.length && <AnalysisPlot title="Native training fit and flagged observations" rows={points} x="ds"
        fields={["training_y", "fitted_y", "lower_99", "upper_99", "flagged_y"]}
        fieldLabels={{ training_y: "Training target", fitted_y: "Native fitted value", lower_99: "99% fitted lower", upper_99: "99% fitted upper", flagged_y: "Outside fitted band" }}
        dashedFields={["lower_99", "upper_99"]} pointFields={["flagged_y"]} dark={dark} scale={scales[doc.analysis.scale]}/>}
      {!!points.length && <Table title="Native fitted anomaly flags" rows={points.filter(row => row.flagged === true)}
        columns={["ds", "training_y", "fitted_y", "lower_99", "upper_99", "flagged"]} inspect={inspect} scale={1}/>}
      <p className="boundary-note">The chart shows up to 182 observations from this fitted model in saved reporting units. Training targets can include explicitly declared modeling-copy adjustments. In-sample fitting uses the whole training prefix; these flags are not historical live alerts, anomaly probabilities, or out-of-sample accuracy evidence.</p>
      <details><summary>Native fitted evidence and exports</summary>
        <EvidenceTable document={doc} name={pointFile} title="Native fitted values" models={[model]} inspect={inspect} scale={1}/>
        <EvidenceTable document={doc} name={summaryFile} title="Native fitted origin receipts" models={[model]} inspect={inspect} scale={1}/>
      </details>
    </>}
  </section>;
}
