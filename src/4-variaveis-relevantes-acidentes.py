import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from utils import DATABASE_DIR, get_dataframe
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Análise de Variáveis Relevantes (Potencial Acidente)",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 10px;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Carrega e processa os dados"""
    try:
        df = get_dataframe()
        # df = pd.read_csv(DATABASE_DIR / 'Comun_Ocorrencias_final.csv',
        #                 on_bad_lines='skip',
        #                 encoding='utf-8',
        #                 sep=',',
        #                 header=0)
        
        # Drop colunas desnecessárias
        columns_to_drop = ['Relato da Ocorrência', 'Ações Imediatas', 
                          'Ação Posterior/Programada', 'Criado', 'Criado por', 
                          'QTD Plano de Ação']
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
        
        # Tratamento para Data
        df['Data'] = pd.to_datetime(df['Data'], format="%d/%m/%Y", errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

@st.cache_data
def train_model(df_filtered):
    """Treina o modelo Random Forest"""
    try:
        # Preparar dados para o modelo
        categorical_columns = ['Unidade', 'Empresa', 'Tipo Funcionário', 'Setor Ocorrência', 'Turno',
                             'Cargo', 'Área', 'Local', 'Parte do Corpo Atingida',
                             'Categoria do Risco', 'Acidente', 'Agente Causador', 'Sexo', 
                             'Tempo de empresa', 'Tipo de Acidente', 'Motivo', 'Gerência', 
                             'Situação Reporte', 'Categoria', 'Dano', 'Afastado', 
                             'Potencial Acidente', 'Outra Empresa']
        
        # Filtrar apenas colunas que existem no DataFrame
        existing_columns = [col for col in categorical_columns if col in df_filtered.columns]
        
        if 'Potencial Acidente' in df_filtered.columns:
            X = df_filtered[existing_columns]
            y = df_filtered['Potencial Acidente']
            
            # Encoding das variáveis categóricas
            X_encoded = pd.get_dummies(X)
            
            # Treinar modelo
            modelo = RandomForestClassifier(random_state=1, n_estimators=100)
            modelo.fit(X_encoded, y)
            
            # Calcular importância das features
            importancia = pd.Series(modelo.feature_importances_, index=X_encoded.columns)
            
            return modelo, importancia.sort_values(ascending=False)
        else:
            st.warning("Coluna 'Potencial Acidente' não encontrada nos dados")
            return None, None
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {e}")
        return None, None

def main(st):
    # Header principal
    st.markdown('<h1 class="main-header">⚠️ Análise de Variáveis Relevantes (Potencial Acidente)</h1>', unsafe_allow_html=True)
    
    # Sidebar para filtros
    st.sidebar.title("🔧 Configurações")
    
    # Carregar dados
    df = load_and_process_data()
    
    if df is None:
        st.error("Não foi possível carregar os dados. Verifique se o arquivo existe no caminho especificado.")
        return
    
    # Filtros na sidebar
    st.sidebar.subheader("📅 Filtro de Data")
    
    if 'Data' in df.columns and not df['Data'].isna().all():
        min_date = df['Data'].min().date() if not df['Data'].isna().all() else datetime(2025, 1, 1).date()
        max_date = df['Data'].max().date() if not df['Data'].isna().all() else datetime(2026, 1, 1).date()
        
        date_range = st.sidebar.date_input(
            "Selecione o período:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['Data'] >= pd.Timestamp(start_date)) & 
                            (df['Data'] <= pd.Timestamp(end_date))]
        else:
            df_filtered = df
    else:
        st.sidebar.warning("Coluna de data não encontrada ou inválida")
        df_filtered = df
    
    # Filtros adicionais
    if 'Unidade' in df_filtered.columns:
        unidades = st.sidebar.multiselect(
            "Selecione as Unidades:",
            options=df_filtered['Unidade'].unique(),
            default=df_filtered['Unidade'].unique()
        )
        df_filtered = df_filtered[df_filtered['Unidade'].isin(unidades)] if unidades else df_filtered
    
    # Métricas principais
    st.subheader("📊 Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total de Ocorrências",
            value=len(df_filtered),
            delta=f"Período selecionado"
        )
    
    with col2:
        if 'Acidente' in df_filtered.columns:
            acidentes = df_filtered['Acidente'].value_counts().get('Sim', 0)
            st.metric(
                label="Acidentes",
                value=acidentes,
                delta=f"{(acidentes/len(df_filtered)*100):.1f}% do total"
            )
    
    with col3:
        if 'Potencial Acidente' in df_filtered.columns:
            potencial = df_filtered['Potencial Acidente'].value_counts().get('Sim', 0)
            st.metric(
                label="Potencial Acidente",
                value=potencial,
                delta=f"{(potencial/len(df_filtered)*100):.1f}% do total"
            )
    
    with col4:
        if 'Afastado' in df_filtered.columns:
            afastados = df_filtered['Afastado'].value_counts().get('Sim', 0)
            st.metric(
                label="Afastamentos",
                value=afastados,
                delta=f"{(afastados/len(df_filtered)*100):.1f}% do total"
            )
    
    # Layout em duas colunas para gráficos
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Distribuição por Categoria")
        if 'Categoria' in df_filtered.columns:
            categoria_counts = df_filtered['Categoria'].value_counts()
            
            fig_pie = px.pie(
                values=categoria_counts.values,
                names=categoria_counts.index,
                title="Ocorrências por Categoria",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        st.subheader("🏢 Top 10 Setores")
        if 'Setor Ocorrência' in df_filtered.columns:
            setor_counts = df_filtered['Setor Ocorrência'].value_counts().head(10)
            
            fig_bar = px.bar(
                x=setor_counts.values,
                y=setor_counts.index,
                orientation='h',
                title="Ocorrências por Setor",
                color=setor_counts.values,
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Gráfico temporal
    st.subheader("📅 Tendência Temporal")
    if 'Data' in df_filtered.columns and not df_filtered['Data'].isna().all():
        df_temporal = df_filtered.groupby(df_filtered['Data'].dt.to_period('M')).size()
        df_temporal.index = df_temporal.index.to_timestamp()
        
        fig_line = px.line(
            x=df_temporal.index,
            y=df_temporal.values,
            title="Número de Ocorrências por Mês",
            labels={'x': 'Data', 'y': 'Número de Ocorrências'}
        )
        fig_line.update_traces(line=dict(width=3))
        fig_line.update_layout(xaxis_title="Data", yaxis_title="Número de Ocorrências")
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Análise de Machine Learning
    st.subheader("🤖 Análise de Machine Learning - Importância das Features")
    
    modelo, importancia = train_model(df_filtered)
    
    if modelo is not None and importancia is not None:
        # Top 20 features mais importantes
        top_features = importancia.head(20)
        
        fig_importance = px.bar(
            x=top_features.values * 100,
            y=top_features.index,
            orientation='h',
            title="Top 20 Features Mais Importantes (%)",
            color=top_features.values,
            color_continuous_scale='RdYlBu_r'
        )
        fig_importance.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=600,
            xaxis_title="Importância (%)",
            yaxis_title="Features"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Métricas do modelo
        st.subheader("📋 Informações do Modelo")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Algoritmo:** Random Forest")
        with col2:
            st.info(f"**Features:** {len(importancia)} variáveis")
        with col3:
            st.info(f"**Amostras:** {len(df_filtered)} registros")
    
    # Tabela de dados detalhados
    st.subheader("📋 Dados Detalhados")
    
    # Filtro para colunas a exibir
    if not df_filtered.empty:
        columns_to_show = st.multiselect(
            "Selecione as colunas para visualizar:",
            options=df_filtered.columns.tolist(),
            default=df_filtered.columns.tolist()[:10]
        )
        
        if columns_to_show:
            st.dataframe(
                df_filtered[columns_to_show].head(100),
                use_container_width=True,
                height=400
            )
            
            # Botão para download
            csv = df_filtered[columns_to_show].to_csv(index=False)
            st.download_button(
                label="⬇️ Baixar dados filtrados (CSV)",
                data=csv,
                file_name=f"ocorrencias_filtradas_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            Dashboard de Análise de Ocorrências | Desenvolvido com ❤️ usando Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("**Desenvolvedores: Clayton Kossoski, Endi Danila de Souza da Silva, Kokouvi Hola Kanyi Kodjovi**")

if __name__ == "__main__":
    main(st)