/* eslint-disable @typescript-eslint/no-require-imports */
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const webDir = path.resolve(__dirname, "..");
const backendDir = path.resolve(__dirname, "..", "..", "backend");
const venvPython = path.join(backendDir, ".venv", "Scripts", "python.exe");
const pythonCmd = fs.existsSync(venvPython) ? `"${venvPython}"` : "python";

const backendCmd = `${pythonCmd} -m uvicorn app:app --reload --port 8000 --host 0.0.0.0`;
const frontendCmd = "npx next dev -p 5600";

const backend = spawn(backendCmd, { cwd: backendDir, stdio: "inherit", shell: true });
const frontend = spawn(frontendCmd, { cwd: webDir, stdio: "inherit", shell: true });

const shutdown = () => {
  if (!backend.killed) backend.kill("SIGINT");
  if (!frontend.killed) frontend.kill("SIGINT");
};

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
process.on("exit", shutdown);
