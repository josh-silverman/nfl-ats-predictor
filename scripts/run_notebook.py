"""Execute a notebook top to bottom in a fresh kernel and save its outputs in place.

Usage (from the repo root):
    conda run -n nfl-ats python scripts/run_notebook.py notebooks/10-win-model.ipynb

Uses only jupyter_client (installed with ipykernel), so no new packages are needed. Stops at the first
error and still saves what ran, so a broken notebook is visible.
"""

import json
import sys
from pathlib import Path

from jupyter_client.manager import start_new_kernel

TIMEOUT = 900  # seconds per cell


def run(path: Path) -> int:
    nb = json.loads(path.read_text())
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(path.parent))
    failed = 0
    try:
        for count, cell in enumerate((c for c in nb["cells"] if c["cell_type"] == "code"), start=1):
            msg_id = kc.execute("".join(cell["source"]))
            outputs = []
            while True:
                msg = kc.get_iopub_msg(timeout=TIMEOUT)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "status" and content["execution_state"] == "idle":
                    break
                if kind == "stream":
                    outputs.append({"output_type": "stream", "name": content["name"],
                                    "text": content["text"].splitlines(keepends=True)})
                elif kind in ("execute_result", "display_data"):
                    out = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        out["execution_count"] = count
                    outputs.append(out)
                elif kind == "error":
                    outputs.append({"output_type": "error", "ename": content["ename"],
                                    "evalue": content["evalue"], "traceback": content["traceback"]})
                    failed = count
            cell["outputs"], cell["execution_count"] = outputs, count
            if failed:
                print(f"Error in code cell {count}: see saved outputs", file=sys.stderr)
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    path.write_text(json.dumps(nb, indent=1) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1])))
