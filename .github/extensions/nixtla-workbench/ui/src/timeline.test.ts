import { describe, expect, it } from "vitest";
import { buildTimeline } from "./timeline";

describe("actuals and forecast windows", () => {
  const historyRows = [{ ds: "2025-01-01", y: 0 }, { ds: "2025-02-01", y: 2 }];
  const rows = [{ ds: "2025-03-01", yhat: 3 }, { ds: "2025-04-01", yhat: null }];

  it("draws real actuals only before the forecast, keeping missing values missing", () => {
    const chart = buildTimeline({ historyRows, rows, dark: false });
    expect(chart.option.series).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: "Observed actuals", data: [0, 2, null, null], connectNulls: false }),
      expect.objectContaining({ name: "Forecast", data: [null, null, 3, null], lineStyle: expect.objectContaining({ type: "dashed" }) }),
    ]));
    expect(chart.actual.has("2025-03-01")).toBe(false);
  });

  it("shades the exact forecast interval independently of model visibility", () => {
    const chart = buildTimeline({ historyRows, rows, dark: false });
    expect(chart.window).toEqual({ origin: "2025-02-01", start: "2025-03-01", end: "2025-04-01" });
    expect(chart.option.series).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: "Forecast window", data: [], silent: true,
        markArea: expect.objectContaining({
          itemStyle: { color: "rgba(100,116,139,.12)" },
          data: [[{ name: "Forecast window", xAxis: "2025-02-01" }, { xAxis: "2025-04-01" }]],
        }),
      }),
    ]));
    expect(chart.option.legend).toMatchObject({ data: ["Observed actuals", "Forecast"] });
    expect(chart.option.dataZoom).toBeUndefined();
  });

  it("focuses on twelve preceding periods and the forecast without blank future space", () => {
    const date = (i: number) => new Date(Date.UTC(2020, i, 1)).toISOString().slice(0, 10);
    const historical = Array.from({ length: 24 }, (_, i) => ({ ds: date(i), y: i }));
    const future = Array.from({ length: 6 }, (_, i) => ({ ds: date(i + 24), yhat: i + 25 }));
    const chart = buildTimeline({ historyRows: historical, rows: future, dark: true, range: "forecast",
      bounds: { min: 0, max: 1000000 }, domainDates: Array.from({ length: 36 }, (_, i) => date(i)) });
    expect(chart.option.xAxis).toMatchObject({ data: Array.from({ length: 18 }, (_, i) => date(i + 12)) });
    expect(chart.option.yAxis).not.toHaveProperty("max");
    expect(chart.dates).toHaveLength(30);
  });

  it("does not label actual-only charts as forecast windows", () => {
    const chart = buildTimeline({ rows: historyRows, fields: ["y"], dark: false });
    expect(chart.window).toBeNull();
    expect(chart.option.series).toHaveLength(1);
    expect(chart.option.series).toEqual(expect.arrayContaining([expect.objectContaining({ name: "Actuals" })]));
  });

  it("keeps path identity separate from silent window and interval layers as models change", () => {
    const chart = buildTimeline({ historyRows, rows: [{ ds: "2025-03-01", A: 3 }], fields: ["A"], dark: false, intervalLevel: 80 });
    expect(chart.option.series).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "path:A", silent: false, tooltip: { show: true } }),
      expect.objectContaining({ id: "forecast-window", silent: true, tooltip: { show: false } }),
    ]));
    expect(chart.legend.some(item => item.name.includes("interval"))).toBe(false);
  });

  it("separates held-out actuals from model inputs and emphasizes the chosen model", () => {
    const chart = buildTimeline({ historyRows, rows: [{ ds: "2025-03-01", y: 5, A: 3, B: 4 }],
      fields: ["y", "A", "B"], focusField: "A", origin: "2025-02-01", dark: true });
    expect(chart.option.series).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: "Actuals observed afterward", data: [null, 2, 5] }),
      expect.objectContaining({ name: "A", lineStyle: expect.objectContaining({ opacity: 1, width: 3 }) }),
      expect.objectContaining({ name: "B", lineStyle: expect.objectContaining({ opacity: .45 }) }),
    ]));
  });

  it("keeps the complete date axis and actual trend when replay origins change", () => {
    const historical = Array.from({ length: 6 }, (_, i) => ({ ds: `2025-0${i + 1}-01`, y: i }));
    const common = { historyRows: historical, dark: false, fields: ["y", "A"], fullHistoryOnly: true,
      range: "forecast" as const, domainDates: historical.map(row => row.ds) };
    const early = buildTimeline({ ...common, origin: "2025-02-01", rows: [
      { ds: "2025-03-01", y: 2, A: 10 }, { ds: "2025-04-01", y: null, A: 11 },
    ] });
    const later = buildTimeline({ ...common, origin: "2025-04-01", rows: [
      { ds: "2025-05-01", y: 4, A: 12 }, { ds: "2025-06-01", y: 5, A: 13 },
    ], bounds: { min: 0, max: 20 } });
    expect(early.option.xAxis).toMatchObject({ data: later.axisDates });
    expect(early.axisDates).toEqual(historical.map(row => row.ds));
    expect(early.option.series).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: "Observed actuals", data: [0, 1, null, null, null, null] }),
      expect.objectContaining({ name: "Actuals observed afterward", data: [null, 1, 2, null, 4, 5], connectNulls: false }),
      expect.objectContaining({ name: "A", data: [null, null, 10, 11, null, null] }),
    ]));
    expect(early.window?.end).toBe("2025-04-01");
    expect(later.window?.end).toBe("2025-06-01");
    expect(later.option.yAxis).toMatchObject({ min: 0, max: 20 });
    expect(early.forecast.has("2025-06-01")).toBe(false);
  });
});
