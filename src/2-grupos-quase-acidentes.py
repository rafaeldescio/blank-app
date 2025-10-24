import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import warnings
from utils import DATABASE_DIR

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Quase Acidentes",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhorar o visual
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stMetric > label {
        font-size: 1.2rem !important;
        font-weight: bold !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .plot-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Carrega e processa os dados"""
    try:
        # Simular dados para demo (substitua pelo seu path real)
        df = pd.read_csv(DATABASE_DIR / 'Comun_Ocorrencias_final.csv', 
                 on_bad_lines='skip',
                 encoding='utf-8',      # Definir o encoding que tentará abrir o arquivo
                 sep=',',               # Delimitador (padrão é vírgula)
                 header=0,              # Primeira linha como cabeçalho
                 #index_col=0            # Usar a primeira coluna como índice
                 )    
        
        df = df.drop('Relato da Ocorrência', axis=1)
        df = df.drop('Ações Imediatas', axis=1)
        df = df.drop('Ação Posterior/Programada', axis=1)
        df = df.drop('Criado', axis=1)
        df = df.drop('Criado por', axis=1)
        df = df.drop('QTD Plano de Ação', axis=1)
        
        df_quase = df[df["Tipo de Acidente"].isin(["Quase Acidente", "Quase Acidente Crítico"])]
        df = df_quase
        df['Data'] = pd.to_datetime(df['Data'], format="%d/%m/%Y")
        
        # Para demo, vou criar dados simulados
        np.random.seed(42)
        n_samples = 500
    
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def perform_clustering(df):
    """Executa o clustering DBSCAN"""
    # Seleciona colunas relevantes e aplica one-hot encoding
    X = pd.get_dummies(df[['Unidade', 'Empresa', 'Tipo Funcionário', 'Setor Ocorrência', 'Turno',
                           'Cargo', 'Área', 'Local', 'Parte do Corpo Atingida',
                           'Categoria do Risco', 'Acidente', 'Agente Causador', 'Sexo', 
                           'Tempo de empresa', 'Tipo de Acidente', 'Motivo', 'Gerência', 
                           'Situação Reporte', 'Categoria', 'Dano', 'Afastado', 
                           'Potencial Acidente', 'Outra Empresa']])
    
    # Normaliza os dados
    X_scaled = StandardScaler().fit_transform(X)
    
    # Aplica DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)
    
    # Adiciona clusters ao dataframe
    df_clustered = df.copy()
    df_clustered["cluster"] = clusters
    
    # PCA para visualização
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return df_clustered, X_pca, clusters, pca

def main(st):
    # Header principal
    st.markdown('<h1 class="main-header">⚠️ Dashboard de Análise de Quase Acidentes</h1>', 
                unsafe_allow_html=True)
    
    # Carrega os dados
    df = load_and_process_data()
    if df is None:
        return
    
    # Sidebar para filtros
    st.sidebar.title("🔍 Filtros")
    st.sidebar.markdown("---")
    
    # Filtro de data
    date_range = st.sidebar.date_input(
        "Período de Análise",
        value=[df['Data'].min(), df['Data'].max()],
        min_value=df['Data'].min(),
        max_value=df['Data'].max()
    )
    
    # Filtros adicionais
    unidades = st.sidebar.multiselect("Unidades", df['Unidade'].unique(), default=df['Unidade'].unique())
    tipos_acidente = st.sidebar.multiselect("Tipo de Acidente", df['Tipo de Acidente'].unique(), 
                                           default=df['Tipo de Acidente'].unique())
    categoria_risco = st.sidebar.multiselect("Categoria do Risco", df['Categoria do Risco'].unique(),
                                            default=df['Categoria do Risco'].unique())
    
    # Aplicar filtros
    if len(date_range) == 2:
        df_filtered = df[(df['Data'] >= pd.to_datetime(date_range[0])) & 
                        (df['Data'] <= pd.to_datetime(date_range[1]))]
    else:
        df_filtered = df
    
    df_filtered = df_filtered[
        (df_filtered['Unidade'].isin(unidades)) &
        (df_filtered['Tipo de Acidente'].isin(tipos_acidente)) &
        (df_filtered['Categoria do Risco'].isin(categoria_risco))
    ]
    
    # Métricas principais
    st.markdown("## 📊 Métricas Principais")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Ocorrências", len(df_filtered), 
                 delta=f"{len(df_filtered) - len(df):.0f}")
    
    with col2:
        quase_acidentes = len(df_filtered[df_filtered['Tipo de Acidente'] == 'Quase Acidente'])
        st.metric("Quase Acidentes", quase_acidentes)
    
    with col3:
        quase_acidentes_criticos = len(df_filtered[df_filtered['Tipo de Acidente'] == 'Quase Acidente Crítico'])
        st.metric("Quase Acidentes Críticos", quase_acidentes_criticos)
    
    with col4:
        risco_alto = len(df_filtered[df_filtered['Categoria do Risco'] == 'Alto'])
        st.metric("Risco Alto", risco_alto)
    
    st.markdown("---")
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Ocorrências por Mês")
        df_monthly = df_filtered.groupby(df_filtered['Data'].dt.to_period('M')).size().reset_index()
        df_monthly['Data'] = df_monthly['Data'].astype(str)
        
        fig = px.line(df_monthly, x='Data', y=0, 
                     title="Tendência Mensal de Ocorrências",
                     labels={'0': 'Número de Ocorrências'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏢 Ocorrências por Unidade")
        unidade_counts = df_filtered['Unidade'].value_counts()
        
        fig = px.bar(x=unidade_counts.index, y=unidade_counts.values,
                    title="Distribuição por Unidade",
                    labels={'x': 'Unidade', 'y': 'Número de Ocorrências'},
                    color=unidade_counts.values,
                    color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    
    # Análises adicionais
    st.markdown("## 📋 Análises Detalhadas")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Por Setor", "👥 Por Funcionário", "⏰ Por Turno", "📊 Causas"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            setor_risk = pd.crosstab(df_filtered['Setor Ocorrência'], df_filtered['Categoria do Risco'])
            fig = px.bar(setor_risk, title="Categoria de Risco por Setor")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            setor_counts = df_filtered['Setor Ocorrência'].value_counts()
            fig = px.pie(values=setor_counts.values, names=setor_counts.index,
                        title="Distribuição por Setor")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            sexo_counts = df_filtered['Sexo'].value_counts()
            fig = px.pie(values=sexo_counts.values, names=sexo_counts.index,
                        title="Distribuição por Sexo")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            tempo_empresa = df_filtered['Tempo de empresa'].value_counts()
            fig = px.bar(x=tempo_empresa.index, y=tempo_empresa.values,
                        title="Tempo de Empresa")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        turno_hora = pd.crosstab(df_filtered['Turno'], df_filtered['Categoria do Risco'])
        fig = px.bar(turno_hora, title="Categoria de Risco por Turno")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            agente_counts = df_filtered['Agente Causador'].value_counts()
            fig = px.pie(values=agente_counts.values, names=agente_counts.index,
                        title="Agente Causador")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            motivo_counts = df_filtered['Motivo'].value_counts()
            fig = px.bar(x=motivo_counts.index, y=motivo_counts.values,
                        title="Principal Motivo")
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de dados filtrados
    st.markdown("## 📊 Dados Detalhados")
    st.dataframe(df_filtered.head(100), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard desenvolvido para análise de quase acidentes e identificação de padrões de risco**")
    st.markdown("**Desenvolvedores: Clayton Kossoski, Endi Danila de Souza da Silva, Kokouvi Hola Kanyi Kodjovi**")

if __name__ == "__main__":
    main()