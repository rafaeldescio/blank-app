import streamlit as st
from pathlib import Path
from utils import run_py, SRC

st.set_page_config(page_title="Grupos de Quase-Acidentes", page_icon="🧩", layout="wide")
st.title("🧩 Grupos de Quase-Acidentes")
st.markdown("Esta página chama `src/2-grupos-quase-acidentes.py`.")
target = SRC / "2-grupos-quase-acidentes.py"
run_py(target)