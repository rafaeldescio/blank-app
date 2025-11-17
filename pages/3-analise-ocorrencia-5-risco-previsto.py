
import streamlit as st
from pathlib import Path
from utils import run_py, SRC

st.set_page_config(page_title="Análise de Ocorrência & Risco Previsto", page_icon="📈", layout="wide")
st.title("📈 Análise de Ocorrência — 5: Risco Previsto")
st.markdown("Esta página chama `src/3-analise-ocorrencia-5-risco-previsto.py`.")
target = SRC / "3-analise-ocorrencia-5-risco-previsto.py"
run_py(target)