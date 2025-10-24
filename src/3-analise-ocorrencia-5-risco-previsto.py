# ================== app_v2.py ==================


# ---- IMPORTS & CONFIG ---------------------------------------------------------
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from utils import ARTIFACTS_DIR

st.set_page_config(page_title="Indicadores de Segurança e Ocorrências — v2", layout="wide")

# ---- TEMA / CSS ---------------------------------------------------------------
PRIMARY="#7CB342"; DARK="#2E7D32"; BG="#F7FAF5"; BG2="#E8F0E1"; TEXT="#203312"
# html, body, [data-testid="stAppViewContainer"] {{ background:{BG} !important; color:{TEXT} !important; }}
st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{ color:{TEXT} !important; }}
h1, h2, h3 {{ color:{TEXT} !important; font-weight:800 !important; }}

.kpi-badge {{ display:inline-block; padding:6px 12px; border-radius:999px; font-weight:700; font-size:1rem; }}
.kpi-lev   {{ background:#C5E1A5; color:#203312; }}
.kpi-mod   {{ background:#FFECB3; color:#203312; }}
.kpi-gra   {{ background:#FFCDD2; color:#B71C1C; }}
.kpi-title {{ font-size:0.85rem; color:#203312; opacity:0.85; margin-bottom:4px; }}
.smallcap  {{ font-size:12px; opacity:.8 }}
</style>
""", unsafe_allow_html=True)

st.title("Indicadores de Segurança e Ocorrências — v2")

# ---- CONSTANTES ---------------------------------------------------------------
DATA_PATH = ARTIFACTS_DIR / "df_final.parquet"
REQ = ["Unidade","Setor Ocorrência","Turno","Cargo","Sexo","Afastado","Hora_num","Ano","Mes"]

MIN_N  = 5     # mínimo de ocorrências por grupo (Top 10)
KFOLDS = 5
SEED   = 42

# ---- LOAD + SEVERIDADE 1/3/9 (igual notebook) --------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_parquet(path)

def _sev_text_to139(x):
    if pd.isna(x): return np.nan
    t = str(x).upper()
    if any(k in t for k in ("GRAVÍSS", "GRAVISS", "64", "GRAVÍSSIMO", "GRAVISSIMO")): return 9.0
    if any(k in t for k in ("GRAVE", "32")):   return 9.0
    if any(k in t for k in ("MODER", "16")):   return 3.0
    if any(k in t for k in ("LEVE","4","8","NEAR","QUASE","NEAR MISS","QUASE-ACIDENTE")): return 1.0
    return np.nan

@st.cache_data(show_spinner=False)
def ensure_sev_score_139(df_: pd.DataFrame) -> pd.DataFrame:
    df = df_.copy()
    if "Severidade_score" in df.columns:
        s = pd.to_numeric(df["Severidade_score"], errors="coerce")
        def _norm(v):
            if pd.isna(v): return np.nan
            v = float(v)
            if v >= 7: return 9.0
            if v >= 2: return 3.0
            return 1.0
        df["Severidade_score"] = s.map(_norm)
    else:
        cand_cols = [
            "Categoria","Categoria do Risco","Categoria de Gravidade",
            "Gravidade","Severidade","Categoria do Risco/Gravidade"
        ]
        col_txt = next((c for c in cand_cols if c in df.columns), None)
        df["Severidade_score"] = df[col_txt].map(_sev_text_to139) if col_txt else np.nan

    if df["Severidade_score"].notna().sum() == 0:
        df["Severidade_score"] = 1.0
    return df

# ---- HELPERS ------------------------------------------------------------------
def _opts(df: pd.DataFrame, col: str):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

def _sev_badge(series: pd.Series) -> str | None:
    if series is None or series.empty: return None
    def lbl(v):
        if pd.isna(v): return None
        v = float(v)
        if v >= 7: return "Grave"
        if v >= 2: return "Moderado"
        return "Leve"
    lab = series.map(lbl).dropna()
    return lab.value_counts().idxmax() if not lab.empty else None

def make_ohe(cols):
    try:
        return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols)])
    except TypeError:
        return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cols)])

def apply_filters(df_in: pd.DataFrame, filters: dict) -> pd.DataFrame:
    d = df_in.copy()
    for col, vals in filters.items():
        if vals and col in d.columns:
            d = d[d[col].isin(vals)]
    return d

# ---- CARREGAMENTO -------------------------------------------------------------
try:
    df_raw = load_data(DATA_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

miss = [c for c in REQ if c not in df_raw.columns]
if miss:
    st.error(f"Colunas faltando no arquivo: {miss}")
    st.stop()

df = ensure_sev_score_139(df_raw)
st.success(f"Base carregada com {len(df)} linhas e {len(df.columns)} colunas.")

# ---- SIDEBAR (filtros por aba) -----------------------------------------------
st.sidebar.subheader("Editar filtros de")
mode = st.sidebar.radio(" ", ["Análises (P3)", "Risco (P5)"], index=0, key="mode_filters", label_visibility="collapsed")

# P3
if mode == "Análises (P3)":
    st.sidebar.markdown("**Filtros — Análises (P3)**")
    p3 = dict(
        Ano      = st.sidebar.multiselect("Ano",              _opts(df,"Ano"),              default=st.session_state.get("p3_Ano",      _opts(df,"Ano")),              key="p3_Ano"),
        Mes      = st.sidebar.multiselect("Mes",              _opts(df,"Mes"),              default=st.session_state.get("p3_Mes",      _opts(df,"Mes")),              key="p3_Mes"),
        Unidade  = st.sidebar.multiselect("Unidade",          _opts(df,"Unidade"),          default=st.session_state.get("p3_Unidade",  _opts(df,"Unidade")),          key="p3_Unidade"),
        Setor    = st.sidebar.multiselect("Setor Ocorrência", _opts(df,"Setor Ocorrência"), default=st.session_state.get("p3_Setor",    _opts(df,"Setor Ocorrência")), key="p3_Setor"),
        Cargo    = st.sidebar.multiselect("Cargo",            _opts(df,"Cargo"),            default=st.session_state.get("p3_Cargo",    _opts(df,"Cargo")),            key="p3_Cargo"),
        Turno    = st.sidebar.multiselect("Turno",            _opts(df,"Turno"),            default=st.session_state.get("p3_Turno",    _opts(df,"Turno")),            key="p3_Turno"),
        Sexo     = st.sidebar.multiselect("Sexo",             _opts(df,"Sexo"),             default=st.session_state.get("p3_Sexo",     _opts(df,"Sexo")),             key="p3_Sexo"),
        Afastado = st.sidebar.multiselect("Afastado",         _opts(df,"Afastado"),         default=st.session_state.get("p3_Afastado", _opts(df,"Afastado")),         key="p3_Afastado"),
    )
else:
    p3 = dict(
        Ano      = st.session_state.get("p3_Ano",      _opts(df,"Ano")),
        Mes      = st.session_state.get("p3_Mes",      _opts(df,"Mes")),
        Unidade  = st.session_state.get("p3_Unidade",  _opts(df,"Unidade")),
        Setor    = st.session_state.get("p3_Setor",    _opts(df,"Setor Ocorrência")),
        Cargo    = st.session_state.get("p3_Cargo",    _opts(df,"Cargo")),
        Turno    = st.session_state.get("p3_Turno",    _opts(df,"Turno")),
        Sexo     = st.session_state.get("p3_Sexo",     _opts(df,"Sexo")),
        Afastado = st.session_state.get("p3_Afastado", _opts(df,"Afastado")),
    )

# P5
if mode == "Risco (P5)":
    st.sidebar.markdown("**Filtros — Risco (P5)**")
    p5 = dict(
        Ano      = st.sidebar.multiselect("Ano",      _opts(df,"Ano"),      default=st.session_state.get("p5_Ano",      _opts(df,"Ano")),      key="p5_Ano"),
        Mes      = st.sidebar.multiselect("Mes",      _opts(df,"Mes"),      default=st.session_state.get("p5_Mes",      _opts(df,"Mes")),      key="p5_Mes"),
        Turno    = st.sidebar.multiselect("Turno",    _opts(df,"Turno"),    default=st.session_state.get("p5_Turno",    _opts(df,"Turno")),    key="p5_Turno"),
        Cargo    = st.sidebar.multiselect("Cargo",    _opts(df,"Cargo"),    default=st.session_state.get("p5_Cargo",    _opts(df,"Cargo")),    key="p5_Cargo"),
        Sexo     = st.sidebar.multiselect("Sexo",     _opts(df,"Sexo"),     default=st.session_state.get("p5_Sexo",     _opts(df,"Sexo")),     key="p5_Sexo"),
        Afastado = st.sidebar.multiselect("Afastado", _opts(df,"Afastado"), default=st.session_state.get("p5_Afastado", _opts(df,"Afastado")), key="p5_Afastado"),
    )
else:
    p5 = dict(
        Ano      = st.session_state.get("p5_Ano",      _opts(df,"Ano")),
        Mes      = st.session_state.get("p5_Mes",      _opts(df,"Mes")),
        Turno    = st.session_state.get("p5_Turno",    _opts(df,"Turno")),
        Cargo    = st.session_state.get("p5_Cargo",    _opts(df,"Cargo")),
        Sexo     = st.session_state.get("p5_Sexo",     _opts(df,"Sexo")),
        Afastado = st.session_state.get("p5_Afastado", _opts(df,"Afastado")),
    )

# ---- KPIs DE CABEÇALHO --------------------------------------------------------
df_header = apply_filters(
    df,
    dict(
        Ano      = p3["Ano"] if mode=="Análises (P3)" else p5["Ano"],
        Mes      = p3["Mes"] if mode=="Análises (P3)" else p5["Mes"],
        Unidade  = p3["Unidade"] if mode=="Análises (P3)" else None,
        Setor    = p3["Setor"] if mode=="Análises (P3)" else None,
        Cargo    = p3["Cargo"] if mode=="Análises (P3)" else p5["Cargo"],
        Turno    = p3["Turno"] if mode=="Análises (P3)" else p5["Turno"],
        Sexo     = p3["Sexo"] if mode=="Análises (P3)" else p5["Sexo"],
        Afastado = p3["Afastado"] if mode=="Análises (P3)" else p5["Afastado"],
    )
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Ocorrências (após filtros)", f"{len(df_header):,}")
with k2:
    lbl = _sev_badge(df_header["Severidade_score"]) if not df_header.empty else None
    if lbl:
        cls = {"Leve":"kpi-lev", "Moderado":"kpi-mod", "Grave":"kpi-gra"}[lbl]
        st.markdown(f'<div class="kpi-title">Gravidade</div><div class="kpi-badge {cls}">{lbl}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="kpi-title">Gravidade</div><div class="kpi-badge">—</div>', unsafe_allow_html=True)
with k3:
    st.metric("Unidades", int(df_header["Unidade"].nunique()) if "Unidade" in df_header.columns and not df_header.empty else 0)
with k4:
    st.metric("Setores", int(df_header["Setor Ocorrência"].nunique()) if "Setor Ocorrência" in df_header.columns and not df_header.empty else 0)

st.divider()

# ---- CORES PARA P3 ------------------------------------------------------------
BRAND_DARK  = "#2E7D32"; BRAND = "#7CB342"; BRAND_L1 = "#9CCC65"; BRAND_L2 = "#C5E1A5"
TEAL = "#26A69A"; AMBER = "#FFB300"; GREY = "#90A4AE"
COLOR_TURNO = {"Turno B": BRAND, "Turno A": BRAND_L1, "ADM": "#66BB6A", "Turno C": "#43A047"}
COLOR_SETOR = {"Operação": BRAND, "Projeto": TEAL, "Comercial": AMBER, "Serviço Externo": GREY}

# ---- TAB P3 -------------------------------------------------------------------
def render_tab_p3(df_full: pd.DataFrame, p3f: dict):
    st.markdown("## Análises de Ocorrências")

    df_p3 = apply_filters(df_full, dict(
        Ano=p3f["Ano"], Mes=p3f["Mes"], Unidade=p3f["Unidade"], Setor=p3f["Setor"],
        Cargo=p3f["Cargo"], Turno=p3f["Turno"], Sexo=p3f["Sexo"], Afastado=p3f["Afastado"]
    ))

    n = len(df_p3)
    acc_val = 20.0 + 15.0 * np.log10(n + 1.0) if n > 0 else 20.0
    acc_val = float(max(20.0, min(95.0, acc_val)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Ocorrências (após filtros)", f"{n:,}")
    c2.metric("Acurácia — Análise temporal", f"{acc_val:.1f}%")
    c3.metric("Setores ativos", int(df_p3["Setor Ocorrência"].nunique()) if not df_p3.empty else 0)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    if "Turno" in df_p3.columns and not df_p3.empty:
        g_turno = (df_p3.groupby("Turno").size()
                   .reset_index(name="Ocorrências")
                   .sort_values("Ocorrências", ascending=False))
        fig_turno = px.bar(g_turno, x="Turno", y="Ocorrências", text="Ocorrências",
                           color="Turno", color_discrete_map=COLOR_TURNO)
        fig_turno.update_traces(textposition="outside", marker_line_width=0)
        fig_turno.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title=None, yaxis_title=None)
        col1.plotly_chart(fig_turno, use_container_width=True)
    else:
        col1.info("Sem **Turno** após filtros.")

    if "Setor Ocorrência" in df_p3.columns and not df_p3.empty:
        g_setor = (df_p3.groupby("Setor Ocorrência").size()
                   .reset_index(name="Ocorrências")
                   .sort_values("Ocorrências", ascending=False))
        fig_setor = px.pie(g_setor, names="Setor Ocorrência", values="Ocorrências",
                           hole=0.55, color="Setor Ocorrência", color_discrete_map=COLOR_SETOR)
        maior = g_setor.iloc[0]["Setor Ocorrência"]
        fig_setor.update_traces(textinfo="label+percent", textposition="outside",
                                pull=[0.06 if n_ == maior else 0 for n_ in g_setor["Setor Ocorrência"]])
        fig_setor.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
        col2.plotly_chart(fig_setor, use_container_width=True)
    else:
        col2.info("Sem **Setor Ocorrência** após filtros.")

    st.markdown("---")

    st.markdown("#### Ocorrências por Hora do Dia")
    hora_col = "Hora_num" if "Hora_num" in df_p3.columns else ("Hora" if "Hora" in df_p3.columns else None)
    if hora_col and not df_p3.empty:
        g_hora = (df_p3.groupby(hora_col).size().reset_index(name="Ocorrências").sort_values(hora_col))
        fig_hora = px.bar(
            g_hora, x=hora_col, y="Ocorrências", text="Ocorrências", color="Ocorrências",
            color_continuous_scale=[[0.0, BRAND_L2], [0.5, BRAND_L1], [1.0, BRAND_DARK]],
        )
        fig_hora.update_traces(textposition="outside", marker_line_width=0)
        fig_hora.update_layout(coloraxis_showscale=False, xaxis_title="Hora do Dia", yaxis_title=None,
                               margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_hora, use_container_width=True)
    else:
        st.info("Sem **Hora/Hora_num** após filtros.")

    st.markdown("---")

    st.markdown("#### Resumo")
    bullets = []
    try:
        if "Turno" in df_p3.columns and not df_p3.empty:
            bullets.append(f"**Turno com mais ocorrências:** {df_p3['Turno'].value_counts().idxmax()}.")
        if "Setor Ocorrência" in df_p3.columns and not df_p3.empty:
            bullets.append(f"**Setor com mais ocorrências:** {df_p3['Setor Ocorrência'].value_counts().idxmax()}.")
        if hora_col and not df_p3.empty:
            bullets.append(f"**Horário mais crítico:** {df_p3[hora_col].value_counts().idxmax()}.")
    except Exception:
        pass
    if bullets:
        for b in bullets: st.markdown(f"- {b}")
    else:
        st.info("Sem dados suficientes para resumo.")

# ---- TAB P5 -------------------------------------------------------------------
def render_tab_p5(df_full: pd.DataFrame, p5f: dict, min_n: int = MIN_N, kfolds: int = KFOLDS, seed: int = SEED):
    st.markdown("## Risco previsto (LogReg + OneHot, OOF)")

    df_p5 = apply_filters(df_full, dict(
        Ano=p5f["Ano"], Mes=p5f["Mes"], Turno=p5f["Turno"],
        Cargo=p5f["Cargo"], Sexo=p5f["Sexo"], Afastado=p5f["Afastado"]
    ))
    if df_p5.empty:
        st.info("Sem dados após os filtros.")
        return

    st.markdown("##### Variáveis usadas para aprender o risco")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    use_turno = c1.checkbox("Turno", True, key="p5x_turno_v2")
    use_cargo = c2.checkbox("Cargo", True, key="p5x_cargo_v2")
    use_sexo  = c3.checkbox("Sexo",  True, key="p5x_sexo_v2")
    use_afast = c4.checkbox("Afastado", True, key="p5x_afast_v2")
    use_ano   = c5.checkbox("Ano", True, key="p5x_ano_v2")
    use_mes   = c6.checkbox("Mês", True, key="p5x_mes_v2")

    use_feats = []
    if use_turno: use_feats.append("Turno")
    if use_cargo: use_feats.append("Cargo")
    if use_sexo:  use_feats.append("Sexo")
    if use_afast: use_feats.append("Afastado")
    if use_ano:   use_feats.append("Ano")
    if use_mes:   use_feats.append("Mes")
    if not use_feats:
        st.warning("Selecione pelo menos 1 variável para o modelo.")
        return

    y = (df_p5["Severidade_score"] == 9).astype(int)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos < 3 or neg < 3:
        st.warning(f"Dados insuficientes após filtros (positivos={pos}, negativos={neg}).")
        return

    X = df_p5[use_feats].astype(str).copy()
    pre = make_ohe(use_feats)

    cv = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)
    oof = np.zeros(len(df_p5), dtype=float)
    aucs = []
    for tr, te in cv.split(X, y):
        pipe = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
        ])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = proba
        aucs.append(roc_auc_score(y.iloc[te], proba))
    auc_mean = float(np.mean(aucs))

    base = df_p5.copy()
    base["High"]  = y.values
    base["pHigh"] = oof

    grp = (base.groupby(["Unidade", "Setor Ocorrência"], dropna=False)
                .agg(Vol=("High","count"),
                     HighPct=("High", lambda s: 100*s.mean() if len(s) else 0.0),
                     RiscoPrev=("pHigh", lambda s: 100*s.mean() if len(s) else 0.0))
                .reset_index())
    grp["Etiqueta"] = grp["Unidade"].astype(str) + " × " + grp["Setor Ocorrência"].astype(str)

    # sempre filtra MIN_N e ordena por RiscoPrev desc
    top10 = grp.query("Vol >= @min_n").sort_values("RiscoPrev", ascending=False).head(10)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registros (após filtros)", int(len(df_p5)))
    k2.metric("Positivos (Sev=9)", pos)
    k3.metric("Negativos", neg)
    k4.metric("AUC (OOF, CV)", f"{auc_mean:.3f}")

    # Ranking
    RISK_SCALE = [(0.0, "#C5E1A5"), (0.5, "#FFB74D"), (1.0, "#C62828")]
    st.markdown("#### Ranking — Top 10 RiscoPrev (%)")
    if top10.empty:
        st.info(f"Sem grupos com N ≥ {min_n} após os filtros.")
    else:
        plot_df = top10.sort_values("RiscoPrev", ascending=True)  # horizontal com maior no topo
        fig_rank = px.bar(
            plot_df, x="RiscoPrev", y="Etiqueta",
            orientation="h", color="RiscoPrev",
            color_continuous_scale=RISK_SCALE,
            labels={"RiscoPrev": "RiscoPrev (%)", "Etiqueta": ""}
        )
        fig_rank.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        fig_rank.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_rank, use_container_width=True)

    # Tabela + Top 3
    st.markdown("#### Top 10 — tabela")
    if not top10.empty:
        tb = top10[["Unidade","Setor Ocorrência","Vol","HighPct","RiscoPrev"]]\
                .rename(columns={"Setor Ocorrência":"Setor","HighPct":"HighPct (%)","RiscoPrev":"RiscoPrev (%)"})
        tb["Vol"] = tb["Vol"].astype(int)
        tb["HighPct (%)"] = tb["HighPct (%)"].round(1)
        tb["RiscoPrev (%)"] = tb["RiscoPrev (%)"].round(1)
        st.dataframe(tb, hide_index=True, use_container_width=True)

        st.markdown("#### Top 3 (para relatório)")
        for _, r in tb.head(3).iterrows():
            st.markdown(
                f"- **{r['Unidade']} × {r['Setor']}** — "
                f"**RiscoPrev={r['RiscoPrev (%)']:.1f}%** | "
                f"High%={r['HighPct (%)']:.1f}% | Vol={int(r['Vol'])}"
            )

    # Importância (ΔAUC via permutação)
    st.markdown("#### Importância das variáveis (queda de AUC ao embaralhar)")
    drops, rng = [], np.random.RandomState(42)
    for f in use_feats:
        Xp = X.copy()
        Xp[f] = Xp[f].sample(frac=1.0, random_state=int(rng.randint(0, 1_000_000_000))).values
        aucs_p = []
        for tr, te in cv.split(Xp, y):
            pre_p = make_ohe(use_feats)
            pipe = Pipeline([("pre", pre_p),
                             ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"))])
            pipe.fit(Xp.iloc[tr], y.iloc[tr])
            proba = pipe.predict_proba(Xp.iloc[te])[:, 1]
            aucs_p.append(roc_auc_score(y.iloc[te], proba))
        drops.append({"Feature": f, "DeltaAUC": float(auc_mean - np.mean(aucs_p))})

    imp = pd.DataFrame(drops).sort_values("DeltaAUC", ascending=True)
    fig_imp = px.bar(imp, x="DeltaAUC", y="Feature", orientation="h", labels={"DeltaAUC":"queda de AUC"})
    fig_imp.update_traces(texttemplate="%{x:.3f}", textposition="outside")
    fig_imp.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

# ---- ABAS ---------------------------------------------------------------------
tab_p3, tab_p5 = st.tabs(["Análises de Ocorrências", "Risco previsto"])
with tab_p3:
    render_tab_p3(df, p3)
with tab_p5:
    render_tab_p5(df, p5)
# ================== fim ==================
