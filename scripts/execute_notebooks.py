"""Execute every notebook from a fresh kernel and archive the executed copies.

Usage::

    python scripts/execute_notebooks.py
    python scripts/execute_notebooks.py --profile paper

Each notebook is run in its own kernel, in notebook order but with no state
carried between them: a notebook that only works because an earlier one left a
variable behind will fail here.  Source notebooks under ``notebooks/`` are left
without stored outputs; the executed copies are written to
``results/<profile>/executed_notebooks/``.

The kernel used is the interpreter running this script, so no named kernel needs
to be registered and no ``JUPYTER_PATH`` has to be set.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def main() -> int:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--timeout", type=int, default=1800)
    arguments = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from boundary_aware_dynamics.config import load_config

    config = load_config(ROOT / "configs" / f"{arguments.profile}.yaml")
    destination = ROOT / config.output_root / "executed_notebooks"
    destination.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {NOTEBOOK_DIR}.")
        return 1

    # The notebooks read their profile from this variable, so an archived copy
    # cannot be stamped with one profile's config hash while having run another.
    environment = {**os.environ, "BAD_PROFILE": config.profile}
    print(f"executing {len(notebooks)} notebooks with profile={config.profile}")

    failures = []
    for path in notebooks:
        started = time.perf_counter()
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=arguments.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
            kernel_env=environment,
        )
        try:
            client.execute()
        except CellExecutionError as error:
            failures.append(path.name)
            print(f"  FAIL {path.name}: {str(error).splitlines()[-1][:160]}")
            continue

        # The profile is recorded so an archived notebook cannot be mistaken for
        # one run under different settings.
        notebook.metadata["boundary_aware_dynamics"] = {
            "profile": config.profile,
            "config_hash": config.config_hash,
        }
        nbformat.write(notebook, destination / path.name)
        print(f"  ok   {path.name} ({time.perf_counter() - started:.1f}s)")

    if failures:
        print(f"\n{len(failures)} notebook(s) failed: {', '.join(failures)}")
        return 1
    print(f"\nAll {len(notebooks)} notebooks executed -> {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
