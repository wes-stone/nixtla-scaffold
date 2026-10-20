import { expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Table } from "./components";

it("renders constructor evidence even with an empty column-label override", () => {
  const html = renderToStaticMarkup(<Table rows={[{ constructor: "Naive", toString: "saved" }]} />);
  expect(html).toContain('>constructor</th>');
  expect(html).toContain('>toString</th>');
  expect(html).toContain("Naive");
});
