import { useState } from "react";
import { ChevronLeft, ChevronRight, Search, X } from "lucide-react";
import { text } from "./api";
import type { Row } from "./api";
import type { AnalysisProps } from "./Evidence";
import { chartNumber, measurementScale, scales } from "./format";
import { modelColor } from "./chartPresentation";
import { RuntimeCapabilities } from "./RuntimeCapabilities";
import { accuracyMetrics, evidenceLabel, finalForecast, MAX_COMPARE_MODELS, metricLabel, modelRanking, modelSelection, savedModel, tournamentWinner } from "./modelSelection";

export function ModelBrowser({ document: doc, update, view, dark, models, audit, forecasts = [] }: AnalysisProps & {
  models: string[]; audit?: Row[]; forecasts?: Row[];
}) {
  const [query, setQuery] = useState("");
  const [family, setFamily] = useState("");
  const [sort, setSort] = useState("accuracy");
  const selected = savedModel(view.rows);
  const { focus, shortlist, shown, unavailable } = modelSelection(doc, models, selected);
  const byModel = new Map((audit ?? []).map(row => [text(row.model), row]));
  const families = [...new Set((audit ?? []).map(row => text(row.family)).filter(Boolean))].sort();
  const label = (row?: Row) => audit ? evidenceLabel(row) : "Loading evidence…";
  const metric = accuracyMetrics.find(item => item === doc.analysis.metric) ?? "rmse";
  const ranking = modelRanking(models, audit, metric);
  const winner = view.run.metadata.purpose === "judgment_scenario" ? "" : tournamentWinner(audit);
  const filtered = ranking.filter(item => item.model.toLowerCase().includes(query.trim().toLowerCase())
    && (!family || item.evidence?.family === family));
  if (sort === "name") filtered.sort((a, b) => a.model.localeCompare(b.model));
  const comparison = doc.analysis.modelMode === "compare";
  const focusOnly = (model: string) => update({ analysis: { previewModel: model, modelMode: "focus" } });
  const toggle = (model: string) => {
    const next = shortlist.includes(model) ? shortlist.filter(item => item !== model) : [...shortlist, model];
    if (!next.length || next.length > MAX_COMPARE_MODELS) return;
    update({ analysis: { models: next, modelMode: "compare", previewModel: next.includes(focus) ? focus : next[0] } });
  };
  const navigation = comparison ? shortlist : filtered.map(item => item.model);
  const index = navigation.indexOf(focus);
  const move = (model: string) => update({ analysis: { previewModel: model } });
  const latest = forecasts.map(row => text(row.ds)).sort().at(-1);
  const endpoint = finalForecast(forecasts, focus);
  const scale = scales[doc.analysis.scale];
  const unit = doc.analysis.scale === "raw" ? text(view.spec.unit_label) || "original units" : doc.analysis.scale;
  return <section className="model-browser" aria-label="Model browser">
    <header><h3>Explore models</h3><span>{models.length} in this run</span></header>
    <p className="model-run-scope">{view.spec.season_length === 1 ? "Nonseasonal" : `Season length ${text(view.spec.season_length)}`} · {text(view.spec.horizon)} forecast periods</p>
    {winner && <div className="model-winner" aria-label="Saved tournament winner">
      <strong>Winner: {winner}</strong>
      <span>Saved tournament selection · {label(byModel.get(winner))}</span>
      <button className="link" onClick={() => focusOnly(winner)}>Show winner</button>
    </div>}
    <div className="model-ensemble-access">
      {models.includes("WeightedEnsemble") ? <button className="link" onClick={() => focusOnly("WeightedEnsemble")}>Show weighted ensemble</button>
        : <span>Weighted ensemble: no saved path</span>}
      <button className="link" onClick={() => update({ view: "Compare", comparison: "models", blendId: "" })}>Weights & paired backtest</button>
    </div>
    <div className="model-mode" role="group" aria-label="Model display mode">
      <button aria-pressed={!comparison} onClick={() => update({ analysis: { modelMode: "focus" } })}>Focus one</button>
      <button aria-pressed={comparison} onClick={() => update({ analysis: { modelMode: "compare", models: shortlist } })}>Compare · {shortlist.length}/{MAX_COMPARE_MODELS}</button>
    </div>
    <div className="focused-model">
      <span>Focused model</span><strong>{focus || "No model available"}</strong>
      {endpoint && endpoint.date === latest && endpoint.value !== null &&
        <p>{chartNumber(endpoint.value, scale)} {unit} <span>· {endpoint.date.slice(0, 7)}</span></p>}
      <small>{label(byModel.get(focus))}{focus === winner ? " · Winner" : focus === selected ? " · Saved selection" : ""}</small>
    </div>
    <div className="model-stepper">
      <button aria-label="Previous model" disabled={index <= 0} onClick={() => move(navigation[index - 1])}><ChevronLeft size={16}/>Previous</button>
      <span>{index < 0 ? "—" : index + 1} / {navigation.length}</span>
      <button aria-label="Next model" disabled={index < 0 || index >= navigation.length - 1} onClick={() => move(navigation[index + 1])}>Next<ChevronRight size={16}/></button>
    </div>
    {comparison && <div className="model-shortlist" aria-label="Comparison shortlist">
      {shortlist.map(model => <div key={model}>
        <button aria-label={`Emphasize ${model}`} aria-pressed={model === focus} onClick={() => move(model)}>
          <i aria-hidden="true" style={{ background: modelColor(model, dark) }}/>{model}</button>
        <button className="icon" aria-label={`Remove ${model} from comparison`} disabled={shortlist.length === 1} onClick={() => toggle(model)}><X size={13}/></button>
      </div>)}
      <small>Up to five paths. Click a name to emphasize it.</small>
    </div>}
    <label className="model-search"><span>Find a model</span><div><Search size={15} aria-hidden="true"/>
      <input type="search" aria-label="Find a model" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search ARIMA, MSTL, Ridge…"/>
    </div></label>
    {families.length > 1 && <label className="model-family">Family<select aria-label="Browse model family" value={family} onChange={event => setFamily(event.target.value)}>
      <option value="">All families</option>{families.map(name => <option key={name}>{name}</option>)}</select></label>}
    <div className="model-sort-controls">
      <label>Sort models<select aria-label="Sort models" value={sort} onChange={event => setSort(event.target.value)}>
        <option value="accuracy">Best accuracy first</option><option value="name">Model name</option>
      </select></label>
      <label>Accuracy metric<select aria-label="Accuracy metric" value={metric} onChange={event => update({ analysis: { metric: event.target.value } })}>
        {accuracyMetrics.map(item => <option key={item} value={item}>{metricLabel(item)}</option>)}
      </select></label>
    </div>
    <div className="model-list-heading"><span>{filtered.length} matching</span><span>{metricLabel(metric)} · lower is better</span></div>
    <div className="model-roster" role="group" aria-label="Available models">
      {filtered.map(({ model, evidence, rank, score, ensemble }) => {
        return <div className="model-roster-row" data-focused={focus === model} data-winner={model === winner} key={model}>
          <button className="model-focus" aria-label={`Focus ${model}`} aria-pressed={focus === model && !comparison} onClick={() => focusOnly(model)}>
            <span className="model-name">{rank !== null && <span className="model-rank">#{rank} </span>}<i aria-hidden="true" style={{ background: modelColor(model, dark) }}/>{model}
              {model === winner && <span className="winner-badge">Winner</span>}</span>
            <small>{label(evidence)}{model === selected && model !== winner ? " · Saved selection" : ""}</small>
          </button>
          <span className="model-endpoint" title={ensemble ? "Separate cohort; see paired backtest, not the component ranking" : `${metricLabel(metric)}: ${score ?? "Not available"}`}>
            {score === null || ensemble ? "—" : chartNumber(score, measurementScale(metric, scale))}
          </span>
          <label className="model-compare"><input type="checkbox" aria-label={`Compare ${model}`}
            checked={shortlist.includes(model)} disabled={shortlist.includes(model) ? shortlist.length === 1 : shortlist.length >= MAX_COMPARE_MODELS}
            onChange={() => toggle(model)}/><span>Compare</span></label>
        </div>;
      })}
      {!filtered.length && <p className="model-empty">No matching model in this run. Clear the search or choose another saved run; model availability is run-specific.</p>}
    </div>
    <p className="model-display-status">Ranks use comparable, selection-eligible backtests only. Configuration-excluded candidates, unverified candidates and ensembles stay unranked; saved scores remain visible. Sorting never changes the saved winner.</p>
    <p className="model-display-status" role="status">{shown.length} {shown.length === 1 ? "model" : "models"} on chart · {comparison ? "comparison" : "focus"} only. No refit or adoption.</p>
    {unavailable && <p className="notice">The previously focused model is not in this run. Showing {focus || "no model"} instead.</p>}
    <button className="quiet" onClick={() => update({ analysis: { previewModel: "", modelMode: "focus" } })}>Return to saved selection</button>
    <RuntimeCapabilities/>
  </section>;
}
