import { z } from "zod";

export const calendarSettingsSchema = z.object({
  referencePeriod: z.string().default(""),
  weekendWeight: z.number().finite().min(0).max(3).default(1),
  holidayWeight: z.number().finite().min(0).max(3).default(1),
  holidayCalendar: z.enum(["none", "us_federal", "custom"]).default("none"),
  customDates: z.string().max(12000).default(""),
});
export const neutralCalendar = () => calendarSettingsSchema.parse({});
export type CalendarSettings = z.infer<typeof calendarSettingsSchema>;
