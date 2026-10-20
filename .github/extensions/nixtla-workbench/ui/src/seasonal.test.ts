import { describe, expect, it } from "vitest";
import { buildSeasonalOverlay, seasonalChart, seasonalKind, seasonalTooltip } from "./seasonal";
import type { Row } from "./api";

const daily = {
  frequency: "D",
  history: [
    { ds: "2026-08-01", y: 0 }, { ds: "2026-08-31", y: 80 },
    { ds: "2026-09-01", y: 90 }, { ds: "2026-09-03", y: null }, { ds: "2026-09-04", y: 94 },
    { ds: "2026-10-15", y: 115 }, { ds: "2026-10-16", y: "123.456789012345" },
  ],
  future: [
    { ds: "2026-10-17", yhat: 150, model: "AutoMFLES" },
    { ds: "2026-10-18", yhat: null, model: "AutoMFLES" },
    { ds: "2026-10-20", yhat: 170, model: "AutoMFLES" },
    { ds: "2026-11-01", yhat: 180, model: "AutoMFLES" },
  ],
};

describe("calendar projections, not refits or aggregations", () => {
  it("matches the supplied Monday-aligned formula for every day of three months", () => {
    const history = Array.from({ length: 92 }, (_, i) => ({
      ds: new Date(Date.UTC(2026, 7, 1 + i)).toISOString().slice(0, 10), y: i,
    }));
    const overlay = buildSeasonalOverlay({ history, future: [], frequency: "D" });
    for (const point of overlay.points) {
      const date = new Date(`${point.date}T00:00:00Z`);
      const first = new Date(`${point.date.slice(0, 7)}-01T00:00:00Z`);
      expect(point.slot).toBe(date.getUTCDate() - 1 + (first.getUTCDay() + 6) % 7);
      expect(point.slot % 7).toBe((date.getUTCDay() + 6) % 7);
    }
    expect(overlay.points).toHaveLength(92);
  });

  it("defaults to the first forecast month and its two prior months, not the last horizon month", () => {
    const overlay = buildSeasonalOverlay(daily);
    expect(overlay.focus).toBe("2026-10");
    expect(overlay.shown).toEqual(["2026-08", "2026-09", "2026-10"]);
    expect(overlay.labels).toHaveLength(36);
    expect(overlay.points.find(point => point.date === "2026-08-01")).toMatchObject({ slot: 5, value: 0 });
    expect(overlay.points.find(point => point.date === "2026-10-17")).toMatchObject({ slot: 19, phase: "Forecast", value: 150 });
    expect(overlay.points.some(point => point.date === "2026-11-01")).toBe(false);
    const november = buildSeasonalOverlay({ ...daily, focusPeriod: "2026-11" });
    expect(november.shown).toEqual(["2026-09", "2026-10", "2026-11"]);
    expect(november.points.find(point => point.date === "2026-11-01")?.slot).toBe(6);
  });

  it("preserves absent dates, null values, zeroes, and full calendar space beyond the horizon", () => {
    const overlay = buildSeasonalOverlay(daily);
    const september = overlay.lines.find(line => line.period === "2026-09")!;
    expect(september.points[1]?.value).toBe(90);
    expect(september.points[2]).toBeNull();
    expect(september.points[3]?.value).toBeNull();
    const future = overlay.lines.find(line => line.phase === "Forecast")!;
    expect(future.points[18]).toMatchObject({ date: "2026-10-16", bridge: true, phase: "Actual", value: "123.456789012345" });
    expect(future.points[20]?.value).toBeNull();
    expect(future.points[21]).toBeNull();
    expect(future.points[33]).toBeNull();
    const options = seasonalChart(overlay, true, 1e6, "AutoMFLES");
    expect(options.option.series).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: "2026-08 · actual", connectNulls: false, lineStyle: expect.objectContaining({ type: "solid" }),
        data: expect.arrayContaining([expect.objectContaining({ value: [5, 0] })]) }),
      expect.objectContaining({ name: "2026-10 · AutoMFLES forecast", lineStyle: expect.objectContaining({ type: "dashed" }),
        data: expect.arrayContaining([expect.objectContaining({ value: [19, 150] })]) }),
    ]));
    expect(options.legend.find(line => line.name === "2026-10 · actual")?.color).toBe(options.legend.find(line => line.phase === "Forecast")?.color);
  });

  it("never bridges a missing observation, another month, or hidden actuals", () => {
    for (const history of [[{ ds: "2026-10-15", y: 1 }], [{ ds: "2026-10-16", y: null }], [{ ds: "2026-09-30", y: 1 }]]) {
      const overlay = buildSeasonalOverlay({ ...daily, history });
      expect(overlay.lines.find(line => line.phase === "Forecast")?.points.some(point => point?.bridge)).toBe(false);
    }
    const hidden = buildSeasonalOverlay({ ...daily, includeActuals: false });
    expect(hidden.lines.every(line => line.phase === "Forecast")).toBe(true);
    expect(hidden.lines[0].points.some(point => point?.bridge)).toBe(false);
  });

  it("retains leap dates and changes only the coordinate when switching daily alignment", () => {
    const input = { frequency: "D", history: [{ ds: "2024-02-29", y: 11 }], future: [{ ds: "2024-03-01", yhat: 12 }] };
    const weekdays = buildSeasonalOverlay(input), dates = buildSeasonalOverlay({ ...input, alignment: "date" });
    expect(weekdays.points[0].slot).toBe(31);
    expect(dates.points[0].slot).toBe(28);
    expect(dates.points.map(point => point.value)).toEqual(weekdays.points.map(point => point.value));
    expect(dates.points[0].original).toBe(input.history[0]);
    expect(input.history).toEqual([{ ds: "2024-02-29", y: 11 }]);
  });

  it("uses a full 12-position year, keeps missing months blank, and honors a fiscal year start", () => {
    const input = { frequency: "ME", history: [{ ds: "2025-12-31", y: 12 }, { ds: "2026-06-30", y: 6 }],
      future: [{ ds: "2026-07-31", yhat: 7 }, { ds: "2026-09-30", yhat: 9 }] };
    const calendar = buildSeasonalOverlay(input);
    expect(calendar.labels).toHaveLength(12);
    expect(calendar.lines.find(line => line.phase === "Forecast")?.points[7]).toBeNull();
    const fiscal = buildSeasonalOverlay({ ...input, yearStart: 7 });
    expect(fiscal.labels[0]).toBe("Jul");
    expect(fiscal.periods).toEqual([{ key: "2025", label: "2025/26" }, { key: "2026", label: "2026/27" }]);
    expect(fiscal.points.find(point => point.date === "2026-06-30")).toMatchObject({ period: "2025", slot: 11 });
    expect(fiscal.points.find(point => point.date === "2026-07-31")).toMatchObject({ period: "2026", slot: 0 });
    expect(fiscal.lines.find(line => line.period === "2026")?.points.filter(Boolean)).toHaveLength(2);
  });

  it("keeps ISO week-year boundaries and week 53 distinct", () => {
    const overlay = buildSeasonalOverlay({ frequency: "W-SUN", history: [{ ds: "2021-01-03", y: 1 }],
      future: [{ ds: "2021-01-10", yhat: 2 }] });
    expect(overlay.points[0]).toMatchObject({ period: "2020", slot: 52 });
    expect(overlay.points[1]).toMatchObject({ period: "2021", slot: 0 });
    expect(overlay.labels.at(-1)).toBe("W53");
  });

  it("uses quarterly saved periods without expanding them into invented months", () => {
    const overlay = buildSeasonalOverlay({ frequency: "QE-DEC", yearStart: 7, history: [{ ds: "2025-12-31", y: 4 }],
      future: [{ ds: "2026-03-31", yhat: 1 }, { ds: "2026-09-30", yhat: 3 }] });
    expect(overlay.labels).toEqual(["Q1", "Q2", "Q3", "Q4"]);
    expect(overlay.periods).toEqual([{ key: "2025", label: "2025" }, { key: "2026", label: "2026" }]);
    expect(overlay.points).toHaveLength(3);
    expect(overlay.lines.find(line => line.phase === "Forecast")?.points[1]).toBeNull();
  });

  it("supports multiple explicit model paths without conflating their values", () => {
    const overlay = buildSeasonalOverlay({ ...daily, future: [...daily.future, { ds: "2026-10-17", model: "Ridge", yhat: 999 }] });
    expect(overlay.lines.filter(line => line.phase === "Forecast")).toHaveLength(2);
    expect(overlay.lines.find(line => line.model === "Ridge")?.points[19]?.value).toBe(999);
  });

  it("fails visibly for duplicate coordinates, pooled series, invalid dates and unsupported frequencies", () => {
    expect(() => buildSeasonalOverlay({ ...daily, history: [...daily.history, daily.history[0]] })).toThrow("Multiple actual values");
    expect(() => buildSeasonalOverlay({ ...daily, future: [...daily.future, daily.future[0]] })).toThrow("Multiple forecast values");
    expect(() => buildSeasonalOverlay({ ...daily, history: [{ ds: "2026-02-30", y: 1 }] })).toThrow("Invalid saved period");
    expect(() => buildSeasonalOverlay({ ...daily, history: daily.history.map((row, i) => ({ ...row, unique_id: String(i) })) })).toThrow("Select one series");
    for (const frequency of ["h", "ms", "2D", "2MS", "QS-MAR", "YE", ""]) {
      expect(seasonalKind(frequency)).toBeNull();
      expect(() => buildSeasonalOverlay({ ...daily, frequency })).toThrow("no periods have been resampled");
    }
  });

  it("shows unavailable historical periods instead of filling them", () => {
    const overlay = buildSeasonalOverlay({ frequency: "MS", history: [], future: [{ ds: "2026-10-01", yhat: 1 }] });
    expect(overlay.missingPeriods).toEqual(["2024", "2025"]);
    expect(overlay.lines).toHaveLength(1);
    expect(overlay.lines[0].points.filter(Boolean)).toHaveLength(1);
  });

  it("has a safe empty state and ignores stale saved focus periods", () => {
    expect(buildSeasonalOverlay({ history: [], future: [], frequency: "D" }).lines).toEqual([]);
    expect(buildSeasonalOverlay({ ...daily, focusPeriod: "1999-01" }).focus).toBe("2026-10");
  });
});

it("shows original dates in escaped tooltips, without mislabeling an observed bridge as forecast", () => {
  const params: Row[] = [
    { seriesName: "<img src=x>", data: { date: "2026-10-17", rawValue: 12345 } },
    { seriesName: "Observed anchor", data: { date: "2026-10-16", rawValue: 1, bridge: true } },
  ];
  const html = seasonalTooltip(params, 1000);
  expect(html).toContain("2026-10-17");
  expect(html).toContain("12.35");
  expect(html).toContain("&lt;img src=x&gt;");
  expect(html).not.toContain("<img");
  expect(html).not.toContain("Observed anchor");
});
