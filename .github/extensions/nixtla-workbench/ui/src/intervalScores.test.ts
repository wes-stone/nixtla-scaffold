import { describe, expect, it } from "vitest";
import { intervalScoreView } from "./intervalScores";

describe("interval score provenance", () => {
  it("hides legacy aggregates without changing window scores, other metrics or source rows", () => {
    const source = ["overall", "horizon", "window"].map(scope => ({
      scope, mean_interval_score: 2, weighted_interval_score: 49.8, coverage: .5, mean_width: 2,
    }));
    const result = intervalScoreView(source);
    expect(result.legacy).toBe(true);
    expect(result.rows.map(row => row.mean_interval_score)).toEqual([null, null, 2]);
    expect(source.map(row => row.mean_interval_score)).toEqual([2, 2, 2]);
    expect(result.rows.every(row => row.weighted_interval_score === 49.8 && row.coverage === .5 && row.mean_width === 2)).toBe(true);
  });

  it.each([2, "2"])("shows corrected version %s alongside masked legacy rows", scoring_version => {
    const corrected = { scope: "overall", mean_interval_score: 497, scoring_version, aggregation: "observation_mean" };
    const result = intervalScoreView([corrected, { scope: "horizon", mean_interval_score: 2 }]);
    expect(result.rows[0]).toBe(corrected);
    expect(result.rows[1].mean_interval_score).toBeNull();
    expect(intervalScoreView([corrected]).legacy).toBe(false);
  });

  it("does not accept an unsupported aggregation or version as corrected", () => {
    for (const fields of [{ scoring_version: 2 }, { scoring_version: 2, aggregation: "cutoff_mean" }, { scoring_version: 3, aggregation: "observation_mean" }]) {
      expect(intervalScoreView([{ scope: "overall", mean_interval_score: 2, ...fields }]).rows[0].mean_interval_score).toBeNull();
    }
    expect(intervalScoreView([])).toEqual({ rows: [], legacy: false });
  });
});
