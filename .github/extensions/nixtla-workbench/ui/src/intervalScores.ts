import type { Row } from "./api";

export const intervalComparison = "appendix/interval_comparison.csv";
export const legacyIntervalWarning = "Legacy overall and horizon Winkler scores are hidden because their aggregation is incorrect. Re-score this saved evidence to restore them. Window scores, coverage, width and WIS are unchanged; raw exports retain the original values.";

export function intervalScoreView(rows: Row[]) {
  let legacy = false;
  const display = rows.map(row => {
    const aggregate = row.scope === "overall" || row.scope === "horizon";
    const corrected = (row.scoring_version === 2 || row.scoring_version === "2") && row.aggregation === "observation_mean";
    if (!aggregate || corrected) return row;
    legacy = true;
    return { ...row, mean_interval_score: null };
  });
  return { rows: display, legacy };
}
