"""Type-check notebook code with Pyright and report notebook/cell locations.

Run with the development environment::

    .venv/bin/python scripts/check_example_types.py
"""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = json.loads((ROOT / "pyrightconfig.json").read_text())
    config["venvPath"] = str(ROOT / config["venvPath"])
    config["extraPaths"] = [str(ROOT / path) for path in config["extraPaths"]]
    config["include"] = ["."]
    locations = {}
    with TemporaryDirectory(prefix="mouse-example-types-") as directory:
        target = Path(directory)
        for notebook in sorted((ROOT / "examples").glob("*.ipynb")):
            lines = []
            positions = []
            document = json.loads(notebook.read_text())
            for cell_index, cell in enumerate(document["cells"], start=1):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"]).splitlines()
                lines.extend([*source, ""])
                positions.extend((cell_index, line) for line in range(1, len(source) + 2))
            script = target / f"{notebook.stem}.py"
            script.write_text("\n".join(lines) + "\n")
            locations[str(script)] = notebook, positions
        project = target / "pyrightconfig.json"
        project.write_text(json.dumps(config))
        result = subprocess.run(
            [sys.executable, "-m", "pyright", "--project", str(project), "--outputjson"],
            capture_output=True, text=True, check=False,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        report = json.loads(result.stdout)
        for diagnostic in report["generalDiagnostics"]:
            path = diagnostic["file"]
            line = diagnostic["range"]["start"]["line"]
            if path in locations:
                notebook, positions = locations[path]
                cell_index, cell_line = positions[line]
                location = f"{notebook}: cell {cell_index}, line {cell_line}"
            else:
                location = f"{path}:{line + 1}"
            print(f"{location}: {diagnostic['severity']}: {diagnostic['message']}")
        summary = report["summary"]
        print(f"{len(locations)} notebooks: {summary['errorCount']} errors, "
              f"{summary['warningCount']} warnings")
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
