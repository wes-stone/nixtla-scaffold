import { describe, expect, it } from "vitest";
import { autocorrelation, histogram, paths, residuals, seasonalRows } from "./analysis";
import { documentSchema, stateSchema } from "./api";
import { updateDocument } from "./domain";
import { formatted } from "./format";

describe("saved analytical views", () => {
  it("keeps old tabs compatible and merges independent controls", () => {
    const doc = documentSchema.parse({ id: "old", view: "Forecast" });
    expect(doc.analysis.models).toEqual(["*"]);
    const state = stateSchema.parse({ documents: [doc], activeId: "old" });
    const next = updateDocument(state, "old", { analysis: { metric: "mae", scale: "billions" } });
    expect(next.documents[0].analysis.fixedAxes).toBe(true);
    expect(next.documents[0].analysis.metric).toBe("mae");
  });
  it("preserves gaps and refuses ambiguous duplicate paths", () => {
    const result = paths([{ ds: "2026-01", model: "A", yhat: null }, { ds: "2026-02", model: "B", yhat: 3 }]);
    expect(result.rows).toEqual([{ ds: "2026-01", A: null }, { ds: "2026-02", B: 3 }]);
    expect(() => paths([{ ds: "d", model: "A", yhat: 1 }, { ds: "d", model: "A", yhat: 2 }])).toThrow("Ambiguous");
  });
  it("aligns a fiscal year without moving the original dates", () => {
    const history = [{ ds: "2025-07-01", y: 10 }, { ds: "2026-01-01", y: 12 }];
    expect(seasonalRows(history, [], 7).rows).toEqual([{ ds: "01", "2025 Actual": 10 }, { ds: "07", "2025 Actual": 12 }]);
    expect(history[0].ds).toBe("2025-07-01");
  });
  it("uses actual minus predicted and only one-step support for ACF", () => {
    const rows = residuals([{ model: "A", y_actual: 10, yhat: 7, horizon_step: 2 }], "A");
    expect(rows[0].residual).toBe(3);
    expect(autocorrelation(rows)).toEqual([]);
    expect(histogram([{ e: -1 }, { e: 0 }, { e: 1 }], "e", 2).map(row => row.count)).toEqual([1, 2]);
  });
  it("never scales statistical tests or dimensionless scores", () => {
    expect(formatted(2.941, "bias_t_stat", 1e9)).toBe("2.94");
    expect(formatted(.25, "child_share_of_parent", 1e9)).toBe("0.25");
    expect(formatted(.895, "importance", 1e9)).toBe("0.90");
  });
});
