import { describe, expect, it } from "vitest";
import { chartNumber, columnLabel, formatted } from "./format";

describe("presentation never changes evidence precision", () => {
  it("renders constructor metadata as a label, never an inherited object member", () => {
    expect(columnLabel("constructor")).toBe("constructor");
    expect(columnLabel("__proto__")).toBe("  proto  ");
    expect(columnLabel("toString")).toBe("toString");
    expect(columnLabel("rmse")).toBe("RMSE");
  });
  it("formats measurements to two decimals with separators", () => {
    expect(formatted(1234567890.123456, "yhat")).toBe("1,234,567,890.12");
    expect(formatted("1234567890.123456", "yhat")).toBe("1,234,567,890.12");
    expect(formatted(12, "rmse")).toBe("12.00");
    expect(formatted(-.001, "bias")).toBe("0.00");
    expect(formatted(1234567890.123456, "yhat", 1e9)).toBe("1.23");
    expect(formatted(.405, "wape", 1e9)).toBe("0.41");
    expect(chartNumber(1234567890.123456)).toBe("1.23B");
  });
  it("scales interval losses but not their coverage or observation counts", () => {
    for (const column of ["mean_width", "mean_interval_score", "weighted_interval_score"]) {
      expect(formatted(1250, column, 1e3)).toBe("1.25");
    }
    expect(formatted(.85, "coverage", 1e3)).toBe("0.85");
    expect(formatted(20, "observations", 1e3)).toBe("20");
  });
  it("scales finance totals and daily errors, never percentage metrics", () => {
    for (const name of ["point_total", "lower", "upper", "prediction", "daily_rmse", "daily_mae", "daily_bias"]) {
      expect(formatted(1250, name, 1e3)).toBe("1.25");
    }
    for (const name of ["landing_mape_pct", "landing_bias_pct", "landing_worst_pct", "next28_mape_pct", "daily_wmape_pct"]) {
      expect(formatted(12.5, name, 1e3)).toBe("12.50");
    }
  });
  it("preserves identifiers, counts, missing values and exact oversized integers", () => {
    expect(formatted("001", "unique_id")).toBe("001");
    expect(formatted("001", "account")).toBe("001");
    expect(formatted("9007199254740993", "yhat")).toBe("9007199254740993");
    expect(formatted("2026-09-01", "ds")).toBe("2026-09-01");
    expect(formatted("NA", "unique_id")).toBe("NA");
    expect(formatted(6, "horizon_step")).toBe("6");
    expect(formatted(null, "yhat")).toBe("");
  });
  it("scales calendar measurements while leaving day counts and factors unchanged", () => {
    for (const column of ["observed", "normalized_actual", "saved_forecast", "calendar_scenario", "scenario_minus_saved"]) {
      expect(formatted("3100", column, 1e3)).toBe("3.10");
    }
    expect(formatted("25.5", "exposure", 1e3)).toBe("25.50");
    expect(formatted("31", "days", 1e3)).toBe("31.00");
    expect(formatted("1.034482758620689655", "calendar_factor", 1e3, 6)).toBe("1.034483");
    expect(formatted("-0.000000001", "calendar_factor", 1e3, 6)).toBe("0.000000");
    expect(formatted(null, "calendar_factor", 1e3, 6)).toBe("");
  });
  it("scales judgment comparisons but preserves original-unit inputs and percentages", () => {
    for (const column of ["model_yhat", "selected_yhat", "calendar_yhat", "calendar_unscaled_yhat", "adjusted_yhat", "delta", "cumulative_delta", "calendar_delta", "judgment_delta"]) {
      expect(formatted("1200000000", column, 1e9)).toBe("1.20");
    }
    for (const column of ["amount", "override", "percent", "delta_pct"]) {
      expect(formatted("-10", column, 1e9)).toBe("-10.00");
    }
    for (const column of ["effective_days", "holiday_loss", "calendar_factor", "calendar_scaling_factor", "month_length_factor", "weekday_weekend_factor", "holiday_factor"]) {
      expect(formatted("1.25", column, 1e9)).toBe("1.25");
    }
  });
});
