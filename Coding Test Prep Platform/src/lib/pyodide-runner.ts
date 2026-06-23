/* eslint-disable @typescript-eslint/no-explicit-any */

let pyodideInstance: any = null;
let loadingPromise: Promise<any> | null = null;

declare global {
  interface Window {
    loadPyodide: (config?: any) => Promise<any>;
  }
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.onload = () => resolve();
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

export async function getPyodide(): Promise<any> {
  if (pyodideInstance) return pyodideInstance;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    await loadScript("https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js");
    pyodideInstance = await window.loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/",
    });
    return pyodideInstance;
  })();

  return loadingPromise;
}

export interface PythonResult {
  stdout: string;
  stderr: string;
  error: string | null;
}

export async function runPython(
  code: string,
  timeoutMs: number = 10000
): Promise<PythonResult> {
  const pyodide = await getPyodide();

  // 타임아웃 처리
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    // stdout/stderr 캡처 설정
    pyodide.runPython(`
import sys, io
__stdout_capture = io.StringIO()
__stderr_capture = io.StringIO()
sys.stdout = __stdout_capture
sys.stderr = __stderr_capture
`);

    const runPromise = (async () => {
      try {
        await pyodide.runPythonAsync(code);
        return null;
      } catch (err: any) {
        return err.message || String(err);
      }
    })();

    // 타임아웃 레이스
    const error = await Promise.race([
      runPromise,
      new Promise<string>((_, reject) => {
        controller.signal.addEventListener("abort", () =>
          reject(new Error("시간 초과 (10초)"))
        );
      }),
    ]).catch((err) => err.message || String(err));

    const stdout: string = pyodide.runPython("__stdout_capture.getvalue()");
    const stderr: string = pyodide.runPython("__stderr_capture.getvalue()");

    // stdout/stderr 복원
    pyodide.runPython(`
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
`);

    return {
      stdout: stdout.trim(),
      stderr: stderr.trim(),
      error: typeof error === "string" ? error : null,
    };
  } finally {
    clearTimeout(timer);
  }
}
