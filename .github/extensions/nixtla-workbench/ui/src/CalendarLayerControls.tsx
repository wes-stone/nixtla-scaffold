import { useEffect, useState } from "react";
import type { z } from "zod";
import { WeightField } from "./CalendarAdjuster";
import { calendarLayerSchema, calendarReceiptSchema, neutralCalendarLayer } from "./judgment";
import type { CalendarLayer } from "./judgment";
import { ErrorNotice } from "./components";

function HolidayLoss({ period, value, change, valid, disabled }: {
  period: string; value?: number; change: (value?: number) => void; valid: (valid: boolean) => void; disabled: boolean;
}) {
  const [text, setText] = useState(value === undefined ? "" : String(value));
  useEffect(() => { setText(value === undefined ? "" : String(value)); }, [value]);
  const invalid = text !== "" && (!Number.isFinite(Number(text)) || Number(text) < 0 || Number(text) > 31);
  return <input type="number" min="0" max="31" step="0.25" disabled={disabled} aria-label={`Holiday loss ${period.slice(0, 7)}`}
    value={text} placeholder="Calendar default" aria-invalid={invalid} onChange={event => {
      const input = event.target.value, number = event.target.valueAsNumber;
      const okay = input === "" || (Number.isFinite(number) && number >= 0 && number <= 31);
      setText(input); valid(okay); if (okay) change(input === "" ? undefined : number);
    }}/>;
}

export function CalendarLayerControls({ settings: supplied, change, periods, history, receipt, busy, stale, loading, calculate, valid }: {
  settings?: CalendarLayer; change: (settings: CalendarLayer) => void; periods: string[]; history: string[];
  receipt?: z.infer<typeof calendarReceiptSchema> | null; busy: boolean; stale: boolean; loading: boolean;
  calculate: () => void; valid: (value: boolean) => void;
}) {
  const settings = supplied ?? neutralCalendarLayer();
  const [invalid, setInvalid] = useState<string[]>([]);
  useEffect(() => { valid(true); }, []);
  useEffect(() => { if (!settings.enabled) { setInvalid([]); valid(true); } }, [settings.enabled]);
  const validate = (field: string, okay: boolean) => {
    const next = okay ? invalid.filter(name => name !== field) : [...new Set([...invalid, field])];
    setInvalid(next); valid(next.length === 0);
  };
  const patch = (input: Partial<CalendarLayer>) => change({ ...settings, ...input });
  const reference = settings.referencePeriod || history.at(-1) || "";
  const months = [...new Set([reference, ...periods.map(period => period.slice(0, 7) + "-01")])].filter(Boolean).sort();
  return <section className="calendar-layer" aria-label="Calendar forecast layer">
    <h3>2. Layer calendar effects</h3>
    <label className="calendar-choice"><input type="checkbox" aria-label="Apply calendar layer" checked={settings.enabled}
      disabled={busy} onChange={event => patch({ enabled: event.target.checked })}/>Apply an activity-day overlay to this monthly consumption model</label>
    {!settings.enabled && <p className="footnote">Off: the chosen model flows directly to final adjustments. Actuals and saved model evidence never change.</p>}
    {settings.enabled && <>
      <p className="notice">Optional judgment, not a fitted correction. MSTL/ARIMA can already contain calendar seasonality; adding it again can double-count. Use complete monthly flow totals only.</p>
      <fieldset disabled={busy} className="calendar-controls"><legend className="visually-hidden">Calendar layer assumptions</legend>
        <label>Reference month<select aria-label="Layer reference month" value={settings.referencePeriod} onChange={event => patch({ referencePeriod: event.target.value })}>
          <option value="">Latest observed month{reference ? ` · ${reference.slice(0, 7)}` : ""}</option>
          {history.map(period => <option key={period} value={period}>{period.slice(0, 7)}</option>)}
        </select><small>Exposure reference only; the observed total stays unchanged</small></label>
        <WeightField label="Layer weekend weight" value={settings.weekendWeight} change={value => patch({ weekendWeight: value })} valid={okay => validate("weekend", okay)}/>
        <label>Holiday calendar<select aria-label="Layer holiday calendar" value={settings.holidayCalendar}
          onChange={event => patch({ holidayCalendar: calendarLayerSchema.shape.holidayCalendar.parse(event.target.value) })}>
          <option value="none">None · no dated holiday effect</option><option value="us_federal">US federal · observed dates</option><option value="custom">Custom dates</option>
        </select><small>Explicit coverage, not a global holiday assumption</small></label>
        <WeightField label="Layer holiday weight" value={settings.holidayWeight} change={value => patch({ holidayWeight: value })} valid={okay => validate("holiday", okay)}/>
        {settings.holidayCalendar === "custom" && <label className="calendar-custom">Holiday dates<textarea aria-label="Layer custom holiday dates"
          rows={3} maxLength={12000} value={settings.customDates} onChange={event => patch({ customDates: event.target.value })}/>
          <small>One YYYY-MM-DD per line; a weekend holiday counts once.</small></label>}
      </fieldset>
      <label className="calendar-choice"><input type="checkbox" aria-label="Hold model horizon total" checked={settings.holdTotal} disabled={busy}
        onChange={event => patch({ holdTotal: event.target.checked })}/>Hold the chosen model's horizon total before final tweaks</label>
      <p className="footnote">{settings.holdTotal ? "Rephases months to the same model total. Final haircuts and overrides can still change it." :
        "Total can change: each model month is multiplied by its effective activity days / reference activity days."} Equal day weights still reflect different month lengths.</p>
      <details className="calendar-losses"><summary>Monthly holiday losses</summary>
        <p className="footnote">Weekday-equivalent activity lost beyond normal weekends. An override replaces that month's dated holiday effect; it does not add a second loss. Blank uses the calendar above; with no calendar, blank means zero/unmodeled. January recovery comes only from reduced loss, not an extra rebound.</p>
        <div className="table-scroll" role="region" aria-label="Monthly holiday assumptions" tabIndex={0}><table><thead><tr><th>Month</th><th>Holiday loss (equivalent days)</th></tr></thead><tbody>
          {months.map(period => <tr key={period}><th scope="row">{period.slice(0, 7)}{period === reference ? " · reference" : ""}</th><td>
            <HolidayLoss period={period} disabled={busy} value={settings.holidayLosses.find(item => item.ds === period)?.days}
              valid={okay => validate(period, okay)} change={days => patch({ holidayLosses: [
                ...settings.holidayLosses.filter(item => item.ds !== period), ...(days === undefined ? [] : [{ ds: period, days }]),
              ] })}/></td></tr>)}
        </tbody></table></div>
      </details>
      {!!invalid.length && <ErrorNotice error="Correct invalid day weights (0–3) or holiday losses (0–31) before saving or exporting."/>}
      <button className="primary" disabled={busy || loading || invalid.length > 0 || !stale} onClick={calculate}>
        {loading ? "Updating calendar layer..." : "Update calendar layer"}</button>
      {stale && <p role="status">{loading ? "Calculating the calendar layer." : "Assumptions changed. Update the calendar layer to see its values."} Saving and export are disabled until this layer is current.</p>}
      {receipt && !stale && <details><summary>Calendar calculation & assumptions</summary><p>{receipt.method}</p>
        <p className="footnote">Reference {receipt.referencePeriod.slice(0, 7)}: {receipt.referenceExposure} effective days.
          Total scaling factor: {receipt.scalingFactor}. Exact rounding residual: {receipt.roundingResidual}.</p>
        {receipt.warnings.map(warning => <p className="footnote" key={warning}>{warning}</p>)}
      </details>}
    </>}
  </section>;
}
