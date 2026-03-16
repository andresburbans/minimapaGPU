/* eslint-disable @typescript-eslint/no-require-imports */
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const webDir = path.resolve(__dirname, "..");
const backendDir = path.resolve(__dirname, "..", "..", "backend");

// Detectar entorno virtual de Python
const venvPythonWin = path.join(backendDir, ".venv", "Scripts", "python.exe");
const pythonExec = fs.existsSync(venvPythonWin) ? `"${venvPythonWin}"` : "python";

// Comandos de inicio
const backendCmd = `${pythonExec} -m uvicorn app:app --reload --port 8000 --host 0.0.0.0 --no-access-log`;
const frontendCmd = "npx next dev -p 5600";

console.log("==================================================");
console.log("[DEV] Iniciando entorno de desarrollo MinimapaGPU");
console.log(`[DEV] Backend Dir: ${backendDir}`);
console.log(`[DEV] Python: ${pythonExec}`);
console.log("==================================================");

// Iniciar Backend
console.log(`[DEV] Ejecutando: ${backendCmd} (Logs de acceso desactivados)`);
const backend = spawn(backendCmd, {
  cwd: backendDir,
  shell: true
});

// Iniciar Frontend
console.log(`[DEV] Ejecutando: ${frontendCmd} (Filtrando logs ruidosos)`);
const frontend = spawn(frontendCmd, {
  cwd: webDir,
  shell: true
});

// Función para filtrar y mostrar logs importantes
const setupFilter = (proc, name) => {
  const filter = (data) => {
    const lines = data.toString().split("\n");
    lines.forEach(line => {
      const cleanLine = line.trim();
      if (!cleanLine) return;

      // Ignorar logs de acceso estándar (GET, POST, 200, 304, etc)
      const isAccessLog = /" (GET|POST|PUT|DELETE|OPTIONS|HEAD) .* " (200|304|201|204)/.test(cleanLine) ||
        / (GET|POST) \/.* (200|304) in /.test(cleanLine); // Next.js format

      if (!isAccessLog) {
        process.stdout.write(`${line}\n`);
      }
    });
  };

  proc.stdout.on("data", filter);
  proc.stderr.on("data", filter);
};

setupFilter(backend, "BACKEND");
setupFilter(frontend, "FRONTEND");

// Manejo de limpieza de procesos
const cleanup = () => {
  console.log("\n[DEV] Deteniendo servicios...");

  if (process.platform === "win32") {
    try {
      // En Windows, matar el árbol de procesos (/t) forzadamente (/f)
      if (backend.pid) spawn("taskkill", ["/pid", backend.pid, "/f", "/t"]);
      if (frontend.pid) spawn("taskkill", ["/pid", frontend.pid, "/f", "/t"]);
    } catch (e) {
      // Ignorar si ya están muertos
    }
  } else {
    if (!backend.killed) backend.kill();
    if (!frontend.killed) frontend.kill();
  }
};

// Capturar señales de terminación
process.on("SIGINT", () => { cleanup(); process.exit(); });
process.on("SIGTERM", () => { cleanup(); process.exit(); });
process.on("exit", cleanup);

// Monitorear cierres inesperados
backend.on("close", (code) => {
  if (code !== 0 && code !== null) {
    console.error(`\n[DEV_ERROR] El Backend se cerró inesperadamente (Código: ${code})`);
    console.error("[DEV_ERROR] Verifica que no haya otro proceso usando el puerto 8000.");
  }
});

frontend.on("close", (code) => {
  if (code !== 0 && code !== null) {
    console.error(`\n[DEV_ERROR] El Frontend se cerró inesperadamente (Código: ${code})`);
  }
});
