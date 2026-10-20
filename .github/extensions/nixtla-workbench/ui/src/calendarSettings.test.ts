import { expect, it } from "vitest";
import { documentSchema, stateSchema } from "./api";
import { closeDocument, openDocument, reopenDocument, updateDocument } from "./domain";
import { calendarSettingsSchema, neutralCalendar } from "./calendarSettings";

it("leaves legacy documents unchanged and uses neutral, explicit calendar defaults", () => {
  expect(documentSchema.parse({ id: "legacy" }).calendar).toBeUndefined();
  expect(neutralCalendar()).toEqual({ referencePeriod: "", weekendWeight: 1, holidayWeight: 1, holidayCalendar: "none", customDates: "" });
  expect(calendarSettingsSchema.safeParse({ weekendWeight: -1 }).success).toBe(false);
  expect(calendarSettingsSchema.safeParse({ holidayWeight: NaN }).success).toBe(false);
});

it("preserves calendar assumptions independently of the saved model through recovery", () => {
  let state = openDocument(stateSchema.parse({}), documentSchema.parse({ id: "one", runId: "saved", calendar: { weekendWeight: 0.5 },
    analysis: { previewModel: "MSTL" } }), true);
  state = updateDocument(state, "one", { note: "Calendar scenario is separate" });
  state = reopenDocument(stateSchema.parse(JSON.parse(JSON.stringify(closeDocument(state, "one")))), "one");
  expect(state.documents[0].calendar?.weekendWeight).toBe(0.5);
  expect(state.documents[0].analysis.previewModel).toBe("MSTL");
});
