import { describe, expect, it } from "vitest";
import { documentSchema } from "./api";
import { compareAccuracy, evidenceLabel, finalForecast, modelRanking, modelSelection, tournamentWinner } from "./modelSelection";
import { chartTooltip, modelColor } from "./chartPresentation";
import { buildTimeline } from "./timeline";

describe("focused model exploration", () => {
  const models = ["AutoARIMA", "AutoARIMA_MSTLFeatures", "MSTL", "Naive", "Theta", "Ridge"];
  it("migrates legacy all-model displays to a single saved selection without changing the recipe", () => {
    const doc = documentSchema.parse({ id: "legacy", analysis: { models: ["*"] } });
    expect(modelSelection(doc, models, "Theta").shown).toEqual(["Theta"]);
    expect(doc.analysis.models).toEqual(["*"]);
  });

  describe("saved accuracy ranking", () => {
    const audit = [
      { model: "AutoARIMA", rmse: 10, mae: 8, bias: -0.1, selection_eligible: true },
      { model: "Huber", rmse: 2, mae: 1, bias: -0.3, selection_eligible: true, is_selected_model: true },
      { model: "WeightedEnsemble", rmse: 0, family: "ensemble", selection_eligible: false },
      { model: "Partial", rmse: 0.1, selection_eligible: false },
      { model: "Legacy", rmse: 0.01 },
    ];
    it("sorts comparable components by accuracy, not alphabetically or forecast endpoint", () => {
      const ranked = modelRanking(audit.map(row => row.model), audit);
      expect(ranked.slice(0, 2).map(row => [row.model, row.rank])).toEqual([["Huber", 1], ["AutoARIMA", 2]]);
      expect(ranked.filter(row => !row.comparable).every(row => row.rank === null)).toBe(true);
      expect(ranked.at(-1)?.model).toBe("WeightedEnsemble");
      expect(tournamentWinner(audit)).toBe("Huber");
      expect(tournamentWinner([{ model: "Huber", selected_model: "Huber", is_selected_model: "True", selection_eligible: true }])).toBe("Huber");
    });
    it("orders bias by magnitude without changing the persisted winner", () => {
      const ranking = modelRanking(audit.map(row => row.model), audit, "bias");
      expect(ranking[0]).toMatchObject({ model: "AutoARIMA", score: .1, rank: 1 });
      expect(tournamentWinner(audit)).toBe("Huber");
    });
    it("uses the declared finance tie-breakers without replacing saved selection", () => {
      const first = { model: "A", landing_mape_pct: 1, next28_mape_pct: 3, daily_wmape_pct: 1 };
      const second = { model: "B", landing_mape_pct: 1, next28_mape_pct: 2, daily_wmape_pct: 99 };
      expect(compareAccuracy(first, second, "landing_mape_pct")).toBeGreaterThan(0);
      expect(compareAccuracy({ ...first, next28_mape_pct: 2 }, second, "landing_mape_pct")).toBeLessThan(0);
      expect(tournamentWinner(audit)).toBe("Huber");
    });
    it("keeps zero valid, unknown scores unranked, and filters independent of ranks", () => {
      const rows = [{ model: "Zero", rmse: 0, selection_eligible: true }, { model: "Null", rmse: null, selection_eligible: true }];
      const ranking = modelRanking(["Null", "Zero", "Missing", "Zero"], rows);
      expect(ranking.map(row => row.rank)).toEqual([1, null, null]);
      expect(ranking[0].model).toBe("Zero");
      expect(modelRanking(audit.map(row => row.model), audit).filter(row => row.model === "AutoARIMA")[0].rank).toBe(2);
    });
    it("breaks ties by MAE, absolute bias, then name; never promotes unknown support", () => {
      expect(compareAccuracy({ model: "Z", rmse: 1, mae: 1, bias: -.1 }, { model: "A", rmse: 1, mae: 1, bias: -.9 }, "rmse")).toBeLessThan(0);
      expect(compareAccuracy({ model: "A", rmse: 1 }, { model: "Z", rmse: 1 }, "rmse")).toBeLessThan(0);
      expect(tournamentWinner([{ model: "Legacy", is_selected_model: true }])).toBe("");
      expect(tournamentWinner([{ model: "Fallback", is_selected_model: true, selection_eligible: false }])).toBe("");
      expect(tournamentWinner([...audit, { model: "Ambiguous", is_selected_model: true, selection_eligible: true }])).toBe("");
    });
  });
  it("keeps focus and comparison separate, deduplicates and caps old large lists", () => {
    const doc = documentSchema.parse({ id: "test", analysis: { previewModel: "MSTL", models: [...models, ...models], modelMode: "compare" } });
    expect(modelSelection(doc, models).shown).toEqual(models.slice(0, 5));
    doc.analysis.modelMode = "focus";
    expect(modelSelection(doc, models).shown).toEqual(["MSTL"]);
  });
  it("does not mix unavailable models across runs and keeps the focused comparison visible", () => {
    const doc = documentSchema.parse({ id: "test", analysis: { previewModel: "Old", models: ["Old", "Naive"], modelMode: "compare" } });
    expect(modelSelection(doc, models, "Theta")).toMatchObject({ focus: "Naive", shown: ["Naive"], unavailable: true });
  });
  it("keeps null endpoints and missing scores distinct from zero or validated evidence", () => {
    expect(finalForecast([{ model: "A", ds: "2026-01", yhat: 0 }, { model: "A", ds: "2026-02", yhat: null }], "A")).toEqual({ date: "2026-02", value: null });
    expect(evidenceLabel()).toBe("No saved tournament score");
    expect(evidenceLabel({ cv_windows: 1, selection_eligible: false })).toContain("partial support");
    expect(evidenceLabel({ cv_windows: 0 })).toContain("No full-horizon");
  });
  it("distinguishes configuration exclusion from missing backtest support without erasing scores", () => {
    const evidence = { model: "AutoMFLES", cv_windows: 20, rmse: 0.2,
      selection_eligible: false, configuration_eligible: false };
    expect(evidenceLabel(evidence)).toBe("20 origins · configuration excluded");
    expect(modelRanking(["AutoMFLES"], [evidence])[0]).toMatchObject({ score: 0.2, rank: null });
    expect(evidenceLabel({ ...evidence, configuration_eligible: true })).toContain("partial support");
  });
  it("preserves a model's color across scope and ordering changes", () => {
    const first = buildTimeline({ rows: [{ ds: "2026-10", AutoARIMA: 1, MSTL: 2 }], fields: ["AutoARIMA", "MSTL"], dark: false });
    const next = buildTimeline({ rows: [{ ds: "2026-10", MSTL: 2 }], fields: ["MSTL"], dark: false });
    expect(first.legend.find(item => item.name === "MSTL")?.color).toBe(next.legend[0].color);
    expect(next.legend[0].color).toBe(modelColor("MSTL"));
  });
  it("omits absent historical predictions, retains zero and safely renders long names", () => {
    const tooltip = chartTooltip([
      { axisValueLabel: "Sep", seriesName: "Observed actuals", value: 0 },
      { axisValueLabel: "Sep", seriesName: "AutoARIMA", value: null },
      { axisValueLabel: "Sep", seriesName: "<script>bad</script>", value: 2 },
    ]);
    expect(tooltip).toContain("0.00");
    expect(tooltip).not.toContain("AutoARIMA");
    expect(tooltip).not.toContain("<script>");
    expect(tooltip).toContain("&lt;script&gt;");
  });
  it("bounds diagnostic tooltip content while prioritizing the focused model", () => {
    const tooltip = chartTooltip(Array.from({ length: 43 }, (_, i) => ({ seriesName: `Model ${i}`, value: i })), 1, "Model 42");
    expect(tooltip.match(/chart-tooltip-row/g)).toHaveLength(8);
    expect(tooltip.indexOf("Model 42")).toBeLessThan(tooltip.indexOf("Model 0"));
    expect(tooltip).toContain("35 more series");
  });
});
