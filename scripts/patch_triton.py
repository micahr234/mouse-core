"""Apply Triton PR #10986 to the pinned 3.8 nightly for Python 3.15.

Upstream: https://github.com/triton-lang/triton/pull/10986
Python 3.15 requires AnnAssign.simple; the pinned wheel omits it.
"""
from importlib.metadata import distribution
from pathlib import Path


VERSION = "3.8.0+gitc01b6774"
BEFORE = "ast.AnnAssign(target=st_target, value=default_value, annotation=annotation)"
AFTER = "ast.AnnAssign(target=st_target, value=default_value, annotation=annotation, simple=1)"


def patch_codegen(path: Path, version: str) -> None:
    """Fail closed on an unexpected dependency; allow repeated installation."""
    if version != VERSION:
        raise RuntimeError(f"Expected Triton {VERSION}, found {version}")
    source = path.read_text()
    if BEFORE not in source and source.count(AFTER) == 1:
        return
    if source.count(BEFORE) != 1 or AFTER in source:
        raise RuntimeError(f"Unexpected Triton source in {path}; cannot apply PR #10986")
    path.write_text(source.replace(BEFORE, AFTER))


if __name__ == "__main__":
    installed = distribution("triton")
    patch_codegen(Path(installed.locate_file("triton/compiler/code_generator.py")), installed.version)
    print("Triton Python 3.15 AnnAssign fix verified (PR #10986).")
