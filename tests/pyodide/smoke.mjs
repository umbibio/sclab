// Pyodide smoke test: build the sclab wheel, install it in Pyodide (node),
// then run driver.py. Needs network (the Pyodide CDN and PyPI). See README.md.
//
//   node tests/pyodide/smoke.mjs
//
// Env: PYODIDE_JS  path to pyodide.mjs (default: the "pyodide" npm package)
//      PYODIDE_CACHE  directory for downloaded Pyodide packages (optional)
//      SCLAB_WHEEL  a prebuilt sclab wheel (default: built with `uv build`)
import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..");

let wheel = process.env.SCLAB_WHEEL;
if (!wheel) {
  const out = mkdtempSync(join(tmpdir(), "sclab-wheel-"));
  execFileSync("uv", ["build", "--wheel", "--out-dir", out, root], { stdio: "inherit" });
  wheel = join(out, readdirSync(out).find((f) => f.endsWith(".whl")));
}

const pyodideJs = process.env.PYODIDE_JS;
const { loadPyodide } = await import(pyodideJs ? pathToFileURL(pyodideJs).href : "pyodide");
const options = process.env.PYODIDE_CACHE ? { packageCacheDir: process.env.PYODIDE_CACHE } : {};
const py = await loadPyodide(options);

py.FS.mkdirTree("/wheels");
py.FS.writeFile(`/wheels/${basename(wheel)}`, readFileSync(wheel));
await py.loadPackage("micropip", { messageCallback: () => {} });

const t0 = Date.now();
try {
  await py.runPythonAsync(`
import micropip
# anndata needs typing-extensions>=4.16, newer than Pyodide's
await micropip.install("typing-extensions>=4.16", reinstall=True)
await micropip.install("emfs:/wheels/${basename(wheel)}")
await micropip.install("pylimma")  # the 'de' extra
# the 'de-edger' extra without numba (no wasm build): sclab swaps in NumPy kernels
await micropip.install(["statsmodels", "matplotlib"])
await micropip.install("edgepython", deps=False)
`);
  console.log(`[OK] installed sclab and its dependencies (${Date.now() - t0} ms)`);
  await py.runPythonAsync(readFileSync(join(here, "driver.py"), "utf8"));
} catch (e) {
  console.error(String(e.message ?? e));
  console.error("SMOKE TEST FAILED");
  process.exit(1);
}
