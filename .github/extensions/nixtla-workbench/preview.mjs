import { resolve } from "node:path";
import { configuration, createService, startServer } from "./server.mjs";
const { runtime, contract } = await configuration();
const workspace = resolve(process.argv[2] ?? process.cwd());
const service = createService({ workspace, python: runtime.python, contract });
const server = await startServer({ workspace, service });
console.log(server.url);
process.once("SIGINT", async () => { await service.close(); await server.close(); process.exit(0); });
process.once("SIGTERM", async () => { await service.close(); await server.close(); process.exit(0); });
