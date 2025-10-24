
import streamlit as st

st.set_page_config(page_title="Segurança & Incidentes - Dashboard", page_icon="📊", layout="wide")

st.sidebar.title("📚 Navegação")
st.sidebar.info("Selecione uma página no menu lateral.")

st.title("Segurança Operacional — Análises e Modelos")
st.markdown(
    "Este app organiza suas análises em múltiplas páginas: **classificação de relatos**, "
    "**grupos de quase-acidentes**, **análise de risco previsto** e **variáveis relevantes**."
)

st.markdown("---")
st.subheader("Estrutura de diretórios esperada")
st.code("""
.
├── app.py
├── pages
│   ├── 1-classificador-situacao-reporte.py
│   ├── 2-grupos-quase-acidentes.py
│   ├── 3-analise-ocorrencia-5-risco-previsto.py
│   └── 4-variaveis-relevantes-acidentes.py
└── src
    ├── 1-classificador-situação-reporte.py
    ├── 1-script-treinamento-modelo.ipynb
    ├── 2-grupos-quase-acidentes.py
    ├── 3-analise-ocorrencia-5-risco-previsto.py
    └── 4-variaveis-relevantes-acidentes.py
""")

st.markdown("Para executar cada página, certifique-se de que os arquivos correspondentes existam em `src/`.")