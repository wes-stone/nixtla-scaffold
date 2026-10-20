import { useState } from "react";
import { text } from "./api";
import type { AnalysisProps } from "./Evidence";
import { EvidenceStatus, EvidenceTable, useEvidence } from "./Evidence";
import { Table } from "./components";
import { scales } from "./format";
import { intervalComparison, intervalScoreView, legacyIntervalWarning } from "./intervalScores";

const comparison = intervalComparison, support = "appendix/interval_support.csv";

export function IntervalReview({ document: doc, view, inspect }: AnalysisProps) {
  const [refresh, setRefresh] = useState(0);
  const present = view.artifacts.includes(comparison);
  const result = useEvidence(doc, present ? [comparison, support] : [], {}, refresh);
  if (!present) return null;
  const display = intervalScoreView((result.data[comparison]?.rows ?? []).filter(row => row.scope === "overall"));
  const overall = display.rows;
  const current = result.data[support]?.rows ?? [];
  const selected = view.spec.interval_calibration;
  const method = typeof selected === "object" && selected !== null && "method" in selected ? text(selected.method) : "Recorded in support";
  return <section className="interval-review" aria-label="Experimental interval evidence">
    <header className="section-heading"><h3>Experimental intervals</h3><button disabled={result.loading} onClick={() => setRefresh(value => value + 1)}>Reload interval evidence</button></header>
    <p>Saved backend: <strong>{method}</strong>. Point forecasts are unchanged. Coverage is a fraction on common scored rows; width and interval scores must be read together. These marginal bands do not certify month totals or planning readiness.</p>
    <EvidenceStatus result={result}/>
    {display.legacy && <p className="notice" role="status">{legacyIntervalWarning}</p>}
    {!result.loading && !result.error && !overall.length && <p className="empty-inline">No paired interval scores are available for this series. Review the support and cold-start reasons below; no replacement band has been substituted.</p>}
    {!!overall.length && <Table title="Matched interval comparison (bounded preview)" rows={overall} columns={["model", "method", "level", "observations", "coverage", "mean_width", "mean_interval_score", "weighted_interval_score", "wis_observations", "compared_methods"]} inspect={inspect} scale={scales[doc.analysis.scale]}/>}
    {!!current.length && <Table title="Current calibration support" rows={current} columns={["model", "method", "level", "status", "calibration_windows", "calibration_age_days", "reason"]} inspect={inspect}/>}
    <details><summary>Coverage and scores by horizon and origin</summary>
      <EvidenceTable document={doc} name={comparison} title="Saved interval evaluation" inspect={inspect}/>
    </details>
  </section>;
}
