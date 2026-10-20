import { expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ConfigurationReview } from "./ConfigurationReview";
import { documentSchema, viewSchema } from "./api";

function render(series: string[], selected = "") {
  const document = documentSchema.parse({ id: "configuration-test", series: selected });
  const view = viewSchema.parse({
    run: { id: "synthetic-fixture", name: "Synthetic configuration fixture", metadata: {} },
    rows: [], columns: [], total: 0, nextOffset: null, spec: { freq: "D", season_length: 7 },
    profile: { freq: "D", season_length: 7 }, series, seriesTruncated: false,
    models: [], artifacts: [], gate: {}, blends: [], notes: [],
  });
  return renderToStaticMarkup(<ConfigurationReview document={document} view={view}
    dark={false} update={() => {}} inspect={() => {}}/>);
}

it("requires a series rather than presenting one series as the whole panel", () => {
  const html = render(["001", "002"]);
  expect(html).toContain("Select one series");
  expect(html).not.toContain("Resolved modeling decisions");
});

it("keeps legacy evidence explicitly unavailable without inventing fitted settings", () => {
  const html = render(["001", "002"], "001");
  expect(html).toContain("7 observations");
  expect(html).toContain("This saved version predates configuration evidence");
  expect(html).not.toContain("Configuration model");
});
