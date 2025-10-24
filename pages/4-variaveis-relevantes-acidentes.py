import streamlit as st
from pathlib import Path
from utils import run_py, SRC

st.set_page_config(page_title="Variáveis Relevantes — Acidentes", page_icon="🧮", layout="wide")
st.title("🧮 Variáveis Relevantes em Acidentes")
st.markdown("Esta página chama `src/4-variaveis-relevantes-acidentes.py`.")
target = SRC / "4-variaveis-relevantes-acidentes.py"
run_py(target)
