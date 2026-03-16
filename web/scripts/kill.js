const { execSync } = require("child_process");

const PORTS = [8000, 5600];

console.log("[KILL] Buscando procesos en puertos:", PORTS.join(", "));

try {
    const stdout = execSync("netstat -ano").toString();
    const lines = stdout.split("\n");
    const pids = new Set();

    lines.forEach(line => {
        PORTS.forEach(port => {
            if (line.includes(`:${port}`)) {
                const parts = line.trim().split(/\s+/);
                const pid = parts[parts.length - 1];
                if (pid && /^\d+$/.test(pid) && pid !== "0") {
                    pids.add(pid);
                }
            }
        });
    });

    if (pids.size === 0) {
        console.log("[KILL] No se encontraron procesos activos en estos puertos.");
    } else {
        console.log(`[KILL] Terminando ${pids.size} procesos encontrados: ${Array.from(pids).join(", ")}`);
        pids.forEach(pid => {
            try {
                execSync(`taskkill /F /PID ${pid}`);
                console.log(`  ✓ PID ${pid} terminado.`);
            } catch (e) {
                console.log(`  ⚠ No se pudo terminar PID ${pid} (¿ya cerrado?).`);
            }
        });
    }
} catch (error) {
    console.error("[KILL] Error al ejecutar limpieza:", error.message);
}
