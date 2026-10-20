import { useEffect, useRef, useState } from "react";
import { z } from "zod";
import { call, failure, rowSchema } from "./api";
import { calendarSettingsSchema, neutralCalendar } from "./calendarSettings";
import type { CalendarSettings } from "./calendarSettings";
import type { AnalysisProps } from "./Evidence";
import { ErrorNotice, Table } from "./components";
import { AnalysisPlot } from "./AnalysisPlot";
import { scales } from "./format";
import { neutralCalendarLayer, neutralJudgment } from "./judgment";

const resultSchema = z.object({
  rows: z.array(rowSchema), recipe: z.object({ runId: z.string(), series: z.string(), settings: calendarSettingsSchema }),
  referencePeriod: z.string(), referenceActual: z.string(), referenceExposure: z.string(), historyPeriods: z.array(z.string()),
  method: z.string(), warnings: z.array(z.string()), claim: z.string(), forecastChanged: z.literal(false),
  holidayDates: z.array(z.object({ ds: z.string(), name: z.string() })),
});
const receiptSchema = z.object({
  id: z.string(), path: z.string(), queryPath: z.string(), rows: z.number(), sha256: z.string(), formulaEscapedCells: z.number(),
});
type Result = z.infer<typeof resultSchema>;
const exposureColumns = ["ds", "days", "weekdays", "weekends", "holiday_weekdays", "holiday_weekends", "exposure"];
const factorColumns = ["ds", "month_length_factor", "weekday_weekend_factor", "holiday_factor", "calendar_factor"];
const labels = {
  ds: "Month", days: "Calendar days", weekdays: "Weekdays (incl. holidays)", weekends: "Weekends (incl. holidays)",
  holiday_weekdays: "Weekday holidays", holiday_weekends: "Weekend holidays", exposure: "Weighted day equivalents",
  month_length_factor: "Month length ×", weekday_weekend_factor: "Day mix ×", holiday_factor: "Holiday ×", calendar_factor: "Combined ×",
  observed: "Observed total", normalized_actual: "Reference-month equivalent", saved_forecast: "Saved forecast (unchanged)",
  calendar_scenario: "Calendar-only scenario", scenario_minus_saved: "Scenario − saved",
};

export function WeightField({ label, value, change, valid }: { label: string; value: number; change: (value: number) => void; valid: (value: boolean) => void }) {
  const [input, setInput] = useState(String(value));
  useEffect(() => { if (Number(input) !== value) setInput(String(value)); }, [value]);
  const number = Number(input), invalid = input.trim() === "" || !Number.isFinite(number) || number < 0 || number > 3;
  return <label>{label}<input type="number" aria-label={label} min="0" max="3" step="0.05" value={input} aria-invalid={invalid}
    onChange={event => {
      const text = event.target.value, next = event.target.valueAsNumber, okay = Number.isFinite(next) && next >= 0 && next <= 3;
      setInput(text); valid(okay); if (okay) change(next);
    }}/><small>1 = ordinary weekday; 0 = no usage</small></label>;
}

export function CalendarAdjuster({ document: doc, view, update, dark, inspect }: AnalysisProps) {
  const settings = doc.calendar ?? neutralCalendar();
  const [result, setResult] = useState<Result>(), [resultKey, setResultKey] = useState("");
  const [busy, setBusy] = useState(""), [error, setError] = useState("");
  const [receipt, setReceipt] = useState<z.infer<typeof receiptSchema>>();
  const [scope, setScope] = useState<"forecast" | "all">("forecast"), [tab, setTab] = useState<"exposure" | "factors" | "values">("exposure");
  const [invalidFields, setInvalidFields] = useState<string[]>([]);
  const [reset, setReset] = useState(0);
  const request = { runId: doc.runId, series: doc.series, settings };
  const key = JSON.stringify(request), current = useRef(key); current.current = key;
  const pending = useRef(false), controller = useRef<AbortController | undefined>(undefined);
  const stale = resultKey !== key || invalidFields.length > 0;
  const change = (patch: Partial<CalendarSettings>) => { update({ calendar: { ...settings, ...patch } }); setReceipt(undefined); };
  const run = async () => {
    if (pending.current || invalidFields.length) return;
    const signal = new AbortController(); controller.current = signal;
    pending.current = true; setBusy("preview"); setError(""); setReceipt(undefined);
    try {
      const next = await call("calendar_preview", request, resultSchema, signal.signal);
      if (!signal.signal.aborted && current.current === key) { setResult(next); setResultKey(key); }
    } catch (err) { if (!signal.signal.aborted) setError(failure(err)); }
    finally { if (!signal.signal.aborted) { pending.current = false; setBusy(""); } }
  };
  useEffect(() => {
    setResult(undefined); setResultKey(""); setError(""); pending.current = false;
    void run();
    return () => { controller.current?.abort(); };
  }, [doc.runId, doc.series]);
  const exportPreview = async () => {
    if (!result || stale || pending.current) return;
    pending.current = true; setBusy("export"); setError("");
    try {
      const value = await call("calendar_export", result.recipe, receiptSchema);
      if (current.current === key) setReceipt(value);
    } catch (err) { setError(failure(err)); }
    finally { pending.current = false; setBusy(""); }
  };
  const download = async (kind: "csv" | "query") => {
    if (!receipt || pending.current) return;
    pending.current = true; setBusy("download"); setError("");
    try {
      const file = await call("explore_download", { id: receipt.id, kind }, z.object({ name: z.string(), base64: z.string(), mediaType: z.string() }));
      const blob = new Blob([Uint8Array.from(atob(file.base64), char => char.charCodeAt(0))], { type: file.mediaType });
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url; link.download = file.name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) { setError(failure(err)); }
    finally { pending.current = false; setBusy(""); }
  };
  const weight = (field: "weekendWeight" | "holidayWeight", label: string) => <WeightField key={`${reset}:${field}`} label={label} value={settings[field]}
    change={value => change({ [field]: value })} valid={okay => setInvalidFields(old => okay ? old.filter(item => item !== field) : [...new Set([...old, field])])}/>;
  const rows = result?.rows.filter(row => scope === "all" || row.scope === "forecast") ?? [];
  const normalized = result?.rows.filter(row => row.scope === "history") ?? [];
  return <section className="calendar-adjuster" aria-label="Calendar adjustment">
    <header className="section-heading"><div><h3>Calendar adjustment</h3><p>Separate month length, weekday/weekend mix and holiday effects from underlying usage.</p></div><span className="calendar-status">Scenario · not fitted</span></header>
    <p className="notice">For complete monthly flow totals only. Defaults give every day equal weight; holiday effects are off. The saved forecast is never adjusted or overwritten.</p>
    <fieldset disabled={!!busy} className="calendar-controls"><legend className="visually-hidden">Calendar scenario assumptions</legend>
      <label>Reference month<select aria-label="Calendar reference month" value={settings.referencePeriod} onChange={event => change({ referencePeriod: event.target.value })}>
        <option value="">Latest observed month{result && !settings.referencePeriod ? ` · ${result.referencePeriod.slice(0, 7)}` : ""}</option>
        {result?.historyPeriods.map(period => <option key={period} value={period}>{period.slice(0, 7)}</option>)}
      </select><small>Anchors the underlying daily rate</small></label>
      {weight("weekendWeight", "Weekend relative weight")}
      <label>Holiday calendar<select aria-label="Holiday calendar" value={settings.holidayCalendar}
        onChange={event => change({ holidayCalendar: calendarSettingsSchema.shape.holidayCalendar.parse(event.target.value) })}>
        <option value="none">None · holiday impact not modeled</option><option value="us_federal">US federal · observed dates</option><option value="custom">Custom dates</option>
      </select><small>Choose coverage explicitly; no global default</small></label>
      {weight("holidayWeight", "Holiday relative weight")}
      {settings.holidayCalendar === "custom" && <label className="calendar-custom">Holiday dates<textarea aria-label="Custom holiday dates" value={settings.customDates} rows={4} maxLength={12000}
        placeholder={"2026-12-25\n2027-01-01"} onChange={event => change({ customDates: event.target.value })}/><small>One YYYY-MM-DD per line. Use actual or observed dates deliberately; duplicates count once.</small></label>}
    </fieldset>
    <p className="footnote">Ordinary weekdays have weight 1. Holiday weight replaces the weekday or weekend weight on that date, so a weekend holiday is counted once.</p>
    {!!invalidFields.length && <ErrorNotice error="Enter relative weights between 0 and 3. Blank or invalid weights are not applied."/>}
    {error && <ErrorNotice error={error}/>}
    <div className="calendar-actions"><button className="primary" disabled={!!busy || !!invalidFields.length} onClick={() => void run()}>{busy === "preview" ? "Calculating calendar..." : "Update calendar preview"}</button>
      <button disabled={!!busy || stale || !result} onClick={() => {
        if (view.run.metadata.purpose === "judgment_scenario") update({ view: "Adjust" });
        else update({ view: "Adjust", judgment: { ...(doc.judgment ?? { ...neutralJudgment(), model: doc.analysis.previewModel }),
          calendar: { ...neutralCalendarLayer(), ...settings, enabled: true } } });
      }}>{view.run.metadata.purpose === "judgment_scenario" ? "Open saved adjustment recipe" : "Use these assumptions in Adjust"}</button>
      <button disabled={!!busy || stale || !result} onClick={() => void exportPreview()}>{busy === "export" ? "Saving..." : "Export calendar & assumptions"}</button>
      <button disabled={!!busy} onClick={() => { update({ calendar: neutralCalendar() }); setInvalidFields([]); setReset(reset + 1); setReceipt(undefined); }}>Reset to equal days</button></div>
    {result && <>
      <p className="footnote">Adjust can apply these day-weight assumptions to your chosen model, then layer final edits. This diagnostic's constant-rate projection is not copied into that model.</p>
      {stale && <p role="status" className="notice">Assumptions changed. The result below is the previous preview; update it before exporting.</p>}
      <p className="calendar-reference">Reference: <strong>{result.referencePeriod.slice(0, 7)}</strong> · {Number(result.referenceActual).toLocaleString(undefined, { maximumFractionDigits: 2 })} observed units / {Number(result.referenceExposure).toLocaleString(undefined, { maximumFractionDigits: 4 })} weighted days.
        The calendar-only projection holds this daily rate constant; it does not inherit the model's growth.</p>
      <AnalysisPlot title="Saved forecast vs calendar-only scenario" rows={result.rows} x="ds" fields={["observed", "saved_forecast", "calendar_scenario"]} dashedFields={["calendar_scenario"]} dark={dark} scale={scales[doc.analysis.scale]}/>
      <p className="footnote">Solid lines: observed totals and saved forecast. Dashed line with diamond markers: the separate calendar-only scenario. Compare exact values in the Values table.</p>
      <div className="calendar-table-tools"><div role="group" aria-label="Calendar table view">
        <button aria-pressed={tab === "exposure"} onClick={() => setTab("exposure")}>Days & holidays</button>
        <button aria-pressed={tab === "factors"} onClick={() => setTab("factors")}>Factor bridge</button>
        <button aria-pressed={tab === "values"} onClick={() => setTab("values")}>Values</button>
      </div><label>Periods<select aria-label="Calendar periods" value={scope} onChange={event => setScope(event.target.value === "all" ? "all" : "forecast")}><option value="forecast">Forecast months</option><option value="all">History + forecast</option></select></label></div>
      {tab === "factors" && <p className="footnote">Month length × weekday/weekend mix × holiday effect = combined factor, each relative to the reference month. A factor of 1 means unchanged exposure. Shown to six decimals; raw copy and exports retain full precision.</p>}
      <Table title={tab === "exposure" ? "Calendar day counts" : tab === "factors" ? "Calendar factor bridge" : "Calendar scenario values"}
        rows={rows} columns={tab === "exposure" ? exposureColumns : tab === "factors" ? factorColumns : ["ds", "observed", "normalized_actual", "saved_forecast", "calendar_scenario", "scenario_minus_saved"]}
        columnLabels={labels} inspect={inspect} scale={scales[doc.analysis.scale]} decimalPlaces={tab === "factors" ? 6 : 2}/>
      <details><summary>Normalized history & method</summary>
        <p className="footnote">{result.method}</p>
        <AnalysisPlot title="Observed vs reference-month-equivalent history" rows={normalized} x="ds" fields={["observed", "normalized_actual"]} dark={dark} scale={scales[doc.analysis.scale]}/>
        <Table title="Calendar-normalized actuals" rows={normalized} columns={["ds", "observed", "normalized_actual", "exposure"]} columnLabels={labels} inspect={inspect} scale={scales[doc.analysis.scale]}/>
      </details>
      <details><summary>Holiday dates & coverage · {result.holidayDates.length} dates</summary>
        <p className="footnote">US federal uses observed dates only. Regional, company-specific and international closures require explicit custom dates.</p>
        <Table title="Applied holiday calendar" rows={result.holidayDates} columns={["ds", "name"]}/>
      </details>
      <details><summary>Assumptions, limitations & saved recipe</summary>{result.warnings.map(message => <p className="footnote" key={message}>{message}</p>)}
        <pre className="file-preview" tabIndex={0}>{JSON.stringify(result.recipe, null, 2)}</pre></details>
    </>}
    {receipt && <section className="calendar-export" aria-label="Calendar export receipt"><h4>Calendar export saved · {receipt.rows} rows</h4>
      <p>Full history and forecast exposure, normalized values, separate scenario and its pinned assumptions. No model was run.</p>
      <div className="toolbar"><button disabled={!!busy} onClick={() => void download("csv")}>Download calendar CSV</button><button disabled={!!busy} onClick={() => void download("query")}>Download assumptions JSON</button></div>
      <details><summary>Local export paths</summary><p>{receipt.path}</p><p>{receipt.queryPath}</p></details></section>}
  </section>;
}
