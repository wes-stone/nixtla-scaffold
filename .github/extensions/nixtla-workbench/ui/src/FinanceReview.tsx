import { z } from "zod";
import type { AnalysisProps } from "./Evidence";
import { EvidenceTable } from "./Evidence";

const policy = z.object({
  selection_policy: z.object({ objective: z.string() }).optional(),
  interval_calibration: z.object({ aggregate_totals: z.boolean().optional() }).optional(),
});

export function FinanceReview({ document, view, inspect }: AnalysisProps) {
  const spec = policy.safeParse(view.spec);
  const landing = spec.success && spec.data.selection_policy?.objective === "monthly_landing";
  const intervals = spec.success && spec.data.interval_calibration?.aggregate_totals;
  if (!landing && !intervals) return null;
  return <section className="interval-review" aria-label="Finance totals evidence">
    <h3>Finance totals</h3>
    {landing && <>
      <p>Selection objective: <strong>monthly Landing MAPE</strong>, then Next28 MAPE and date-averaged WMAPE. Current and next month totals include actuals through each origin. Missing or zero-denominator windows cannot become favorable scores.</p>
      <EvidenceTable document={document} name="appendix/finance_scorecard.csv" title="Declared finance scorecard" inspect={inspect}/>
      <details><summary>Inspect origin and closed-month windows</summary>
        <EvidenceTable document={document} name="appendix/finance_windows.csv" title="Finance evaluation windows" inspect={inspect}/>
      </details>
    </>}
    {intervals && <>
      <p>Month and quarter bands use historical <strong>total errors</strong>, never summed daily bounds. At least 20 disjoint, fully landed windows with matching duration and current/next-period offset are required. Experimental bands assume exchangeable errors; they do not grant planning readiness or simultaneous coverage.</p>
      <EvidenceTable document={document} name="appendix/aggregate_intervals.csv" title="Aggregate uncertainty and support" inspect={inspect}/>
    </>}
  </section>;
}
