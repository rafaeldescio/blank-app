
import runpy
import importlib.util
from pathlib import Path
import streamlit as st
import types
import nbformat
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC = ROOT_DIR / "src"
DATABASE_DIR = ROOT_DIR / "base-de-dados"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ROOT_DIR / "modelos"

def _import_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return module

def run_py(path: Path, **kwargs):
    """Run a .py script. If it defines main(st), call it; else exec the file."""
    if not path.exists():
        st.error(f"Arquivo não encontrado: {path}")
        return
    try:
        mod = _import_module_from_path(path)
        if hasattr(mod, "main"):
            st.success("Executando `main(st)` do módulo…")
            mod.main(st, **kwargs)  # type: ignore
        else:
            st.warning("`main(st)` não encontrado. Executando o arquivo como script (sem UI dedicada)…")
            runpy.run_path(str(path))
    except Exception as e:
        st.exception(e)

def render_notebook(nb_path: Path):
    """Render a .ipynb as Markdown/Code (no execution)."""
    if not nb_path.exists():
        st.error(f"Notebook não encontrado: {nb_path}")
        return
    try:
        nb = nbformat.read(nb_path, as_version=4)
        for i, cell in enumerate(nb.cells, start=1):
            if cell.cell_type == "markdown":
                st.markdown(cell.source)
            elif cell.cell_type == "code":
                with st.expander(f"Cell {i} (code)"):
                    st.code(cell.source, language="python")
                    st.caption("Esta célula não é executada automaticamente. Exporte o notebook para .py para executar.")
            else:
                st.text(f"[{cell.cell_type}]")
    except Exception as e:
        st.exception(e)
