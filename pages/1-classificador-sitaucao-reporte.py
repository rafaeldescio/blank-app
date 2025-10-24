import streamlit as st
from pathlib import Path
from utils import run_py, SRC, render_notebook

st.set_page_config(page_title="Classificador de Situação do Reporte", page_icon="🔎", layout="wide")
st.title("🔎 Classificador de Situação do Reporte")

st.markdown("Esta página chama `src/1-classificador-situação-reporte.py`.")
target = SRC / "1-classificador-situacao-reporte.py"
run_py(target)

st.markdown("---")
st.subheader("Treinamento do modelo (notebook)")
nb = SRC / "1-script-treinamento-modelo.ipynb"
render_notebook(nb)
