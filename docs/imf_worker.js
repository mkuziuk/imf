/* Web worker: hosts the Pyodide runtime and the canonical NumPy module. */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const ready = (async () => {
  postMessage({ type: "boot", stage: "Loading Python runtime (~15 MB on first visit)…" });
  const py = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
  });
  postMessage({ type: "boot", stage: "Loading NumPy…" });
  await py.loadPackage("numpy");
  postMessage({ type: "boot", stage: "Loading imf_core.py…" });
  const resp = await fetch("imf_core.py");
  if (!resp.ok) throw new Error(`could not fetch imf_core.py (${resp.status})`);
  py.runPython(await resp.text());
  postMessage({ type: "ready" });
  return py;
})().catch((err) => {
  postMessage({ type: "boot-error", message: String(err) });
  throw err;
});

onmessage = async (event) => {
  const { type, runId, params } = event.data;
  if (type !== "run") return;
  let py;
  try {
    py = await ready;
  } catch {
    return; // boot error already reported
  }
  try {
    py.globals.set("PARAMS_JSON", JSON.stringify(params));
    const out = py.runPython("run_demo(PARAMS_JSON)");
    postMessage({ type: "result", runId, payload: JSON.parse(out) });
  } catch (err) {
    postMessage({ type: "error", runId, message: String(err) });
  }
};
