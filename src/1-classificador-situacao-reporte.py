import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


# Lib para manipulação de dados
import pandas as pd

# Libs da Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Integrando Keras e Scikit-Learn 
from scikeras.wrappers import KerasClassifier
# Criando arquitetura da rede neural
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import pandas as pd
#import tensorflow as tf
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import load_model
from utils import DATABASE_DIR, MODEL_DIR

df = pd.read_csv(DATABASE_DIR / 'Comun_Ocorrencias_final.csv', 
                 on_bad_lines='skip',
                 encoding='utf-8',      # Definir o encoding que tentará abrir o arquivo
                 sep=',',               # Delimitador (padrão é vírgula)
                 header=0,              # Primeira linha como cabeçalho
                 #index_col=0            # Usar a primeira coluna como índice
                 )       

df = df.drop('Data', axis=1)
df = df.drop('Relato da Ocorrência', axis=1)
df = df.drop('Ações Imediatas', axis=1)
df = df.drop('Ação Posterior/Programada', axis=1)
df = df.drop('Criado', axis=1)
df = df.drop('Criado por', axis=1)
df = df.drop('QTD Plano de Ação', axis=1)


classificador = load_model(MODEL_DIR / "modelo_situacao_reporte.keras")

print('classificador carregado:' + str(classificador))


# Configuração da página
st.set_page_config(
    page_title="Sistema de Predição de Reportes",
    page_icon="🔍",
    layout="wide"
)

# Título da aplicação
st.title("🔍 Sistema de Predição de Reportes")
st.markdown("Sistema para predição de reportes utilizando modelos de Deep Learning")
st.markdown("---")

# Sidebar para carregar modelo
# st.sidebar.header("🤖 Carregamento do Modelo")

@st.cache_resource
def load_model(model_file):
    """Carrega o modelo de deep learning"""
    try:
        if model_file.name.endswith('.h5') or model_file.name.endswith('.keras'):
            # Para modelos TensorFlow/Keras, salva temporariamente o arquivo
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(model_file.read())
                tmp_file_path = tmp_file.name
            
            model = keras.models.load_model(tmp_file_path)
            os.unlink(tmp_file_path)  # Remove arquivo temporário
            st.sidebar.success("✅ Modelo TensorFlow/Keras carregado!")
            
        elif model_file.name.endswith('.pkl'):
            # Para arquivos pickle, carrega diretamente do buffer
            model = pickle.load(model_file)
            st.sidebar.success("✅ Modelo Pickle carregado!")
            
        elif model_file.name.endswith('.joblib'):
            # Para arquivos joblib, carrega diretamente do buffer  
            model = joblib.load(model_file)
            st.sidebar.success("✅ Modelo Joblib carregado!")
            
        else:
            st.sidebar.error("❌ Formato não suportado. Use .h5, .keras, .pkl ou .joblib")
            return None
            
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None

# Upload do modelo
#uploaded_model = st.sidebar.file_uploader(
#    "Selecione o arquivo do modelo:",
#    type=['h5', 'keras', 'pkl', 'joblib'],
#    help="Formatos suportados: .h5, .keras, .pkl, .joblib"
#)



#model = None
#if uploaded_model is not None:
#    model = load_model(uploaded_model)

model = classificador

# Opção para carregar pré-processadores
#st.sidebar.markdown("---")
#st.sidebar.header("🔧 Pré-processadores (Opcional)")
#uploaded_preprocessors = st.sidebar.file_uploader(
#    "Selecione o arquivo de pré-processadores:",
#    type=['pkl', 'joblib'],
#    help="Arquivo contendo encoders, scalers e outros pré-processadores"
#)

#preprocessors = None
#if uploaded_preprocessors is not None:
#    try:
#        if uploaded_preprocessors.name.endswith('.pkl'):
#            preprocessors = pickle.load(uploaded_preprocessors)
#        elif uploaded_preprocessors.name.endswith('.joblib'):
#            preprocessors = joblib.load(uploaded_preprocessors)
#        st.sidebar.success("✅ Pré-processadores carregados!")
#    except Exception as e:
#        st.sidebar.error(f"❌ Erro ao carregar pré-processadores: {str(e)}")

def main(st):
    # Interface principal para preenchimento dos campos
    st.header("📝 Preenchimento dos Dados para Predição")

    if model is None:
        st.warning("⚠️ Por favor, carregue um modelo na barra lateral para continuar.")
        st.stop()

    # Inicializar session_state se necessário
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}

    # Botões para carregar dados automaticamente
    st.subheader("🚀 Preenchimento Automático")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

    with col_btn1:
        if st.button("📋 Não Reportável", key="btn_nao_reportavel_main", use_container_width=True):
            st.session_state.form_data = {
                'Unidade': 'Camaçari',
                'Empresa': 'Cibra',
                'Tipo Funcionário': 'Próprio',
                'Setor Ocorrência': 'Operação',
                'Turno': 'Turno C',
                'Cargo': 'Não aplicável',
                'Área': 'Granulação',
                'Local': 'Caldeira',
                'Parte do Corpo Atingida': 'Não Aplicável',
                'Categoria do Risco': 'Não informado',
                'Acidente': 'NIT - Não Impacta Taxa',
                'Agente Causador': 'Não informado',
                'Sexo': 'Feminino',
                'Tempo de empresa': 'Não informado',
                'Tipo de Acidente': 'Acidente com Dano Material',
                'Motivo': 'Não informado',
                'Gerência': 'Não informado',
                'Situação Reporte': 'Não Reportável',
                'Categoria': 'Não informado',
                'Dano': 'Não informado',
                'Afastado': 'Não informado',
                'Potencial Acidente': 'Não informado',
                'Outra Empresa': 'Não informado'
            }
            st.success("✅ Dados 'Não Reportável' carregados automaticamente!")
            st.rerun()

    with col_btn2:
        if st.button("📋 Reportável",  key="btn_reportavel_main", use_container_width=True):
            st.session_state.form_data = {
                'Unidade': 'Sinop',
                'Empresa': 'Armac',
                'Tipo Funcionário': 'Terceiro',
                'Setor Ocorrência': 'Operação',
                'Turno': 'Turno C',
                'Cargo': 'Operador de empilhadeira',
                'Área': 'Linha de enlonamento',
                'Local': 'Expedição',
                'Parte do Corpo Atingida': 'Não Aplicável',
                'Categoria do Risco': 'Equipamentos Móveis e Veículos',
                'Acidente': 'NIT - Não Impacta Taxa',
                'Agente Causador': 'Batida contra',
                'Sexo': 'Masculino',
                'Tempo de empresa': '0 a 3 Meses',
                'Tipo de Acidente': 'Acidente com Dano Material',
                'Motivo': 'Condição',
                'Gerência': 'Operação',
                'Situação Reporte': 'Reportável',
                'Categoria': 'Leve',
                'Dano': 'Leve',
                'Afastado': 'Não',
                'Potencial Acidente': 'Médio',
                'Outra Empresa': 'Não informado'
            }
            st.success("✅ Dados 'Reportável' carregados automaticamente!")
            st.rerun()

    with col_btn3:
        if st.button("🗑️ Limpar Campos", use_container_width=True):
            st.session_state.form_data = {}
            st.success("✅ Campos limpos!")
            st.rerun()

    st.markdown("---")

    # Criar formulário para os campos
    with st.form("prediction_form"):
        # Organizar campos em colunas
        col1, col2, col3 = st.columns(3)
        
        # Dicionário para armazenar os valores
        data = {}
        
        with col1:
            st.subheader("👥 Dados Pessoais e Funcionais")
            data['Unidade'] = st.text_input("Unidade", value=st.session_state.form_data.get('Unidade', ''))
            data['Empresa'] = st.text_input("Empresa", value=st.session_state.form_data.get('Empresa', ''))
            data['Tipo Funcionário'] = st.text_input("Tipo Funcionário", value=st.session_state.form_data.get('Tipo Funcionário', ''))
            data['Cargo'] = st.text_input("Cargo", value=st.session_state.form_data.get('Cargo', ''))
            
            # Campo Sexo com opções predefinidas
            sexo_options = ["", "Masculino", "Feminino"]
            sexo_default = st.session_state.form_data.get('Sexo', '')
            if sexo_default in sexo_options:
                sexo_index = sexo_options.index(sexo_default)
            else:
                sexo_index = 0
            data['Sexo'] = st.selectbox("Sexo", options=sexo_options, index=sexo_index)
            
            data['Tempo de empresa'] = st.text_input("Tempo de empresa", value=st.session_state.form_data.get('Tempo de empresa', ''))
            data['Gerência'] = st.text_input("Gerência", value=st.session_state.form_data.get('Gerência', ''))
            
            # Campo Outra Empresa
            outra_empresa_options = ["", "Sim", "Não", "Não informado"]
            outra_empresa_default = st.session_state.form_data.get('Outra Empresa', '')
            if outra_empresa_default in outra_empresa_options:
                outra_empresa_index = outra_empresa_options.index(outra_empresa_default)
            else:
                outra_empresa_index = 0
            data['Outra Empresa'] = st.selectbox("Outra Empresa", options=outra_empresa_options, index=outra_empresa_index)
        
        with col2:
            st.subheader("🏭 Dados do Local e Ocorrência")
            data['Setor Ocorrência'] = st.text_input("Setor Ocorrência", value=st.session_state.form_data.get('Setor Ocorrência', ''))
            data['Turno'] = st.text_input("Turno", value=st.session_state.form_data.get('Turno', ''))
            data['Área'] = st.text_input("Área", value=st.session_state.form_data.get('Área', ''))
            data['Local'] = st.text_input("Local", value=st.session_state.form_data.get('Local', ''))
            data['Parte do Corpo Atingida'] = st.text_input("Parte do Corpo Atingida", value=st.session_state.form_data.get('Parte do Corpo Atingida', ''))
            data['Categoria do Risco'] = st.text_input("Categoria do Risco", value=st.session_state.form_data.get('Categoria do Risco', ''))
            data['Agente Causador'] = st.text_input("Agente Causador", value=st.session_state.form_data.get('Agente Causador', ''))
            data['Tipo de Acidente'] = st.text_input("Tipo de Acidente", value=st.session_state.form_data.get('Tipo de Acidente', ''))
        
        with col3:
            st.subheader("📊 Classificação e Status")
            data['Acidente'] = st.text_input("Acidente", value=st.session_state.form_data.get('Acidente', ''))
            data['Motivo'] = st.text_input("Motivo", value=st.session_state.form_data.get('Motivo', ''))
            data['Situação Reporte'] = st.text_input("Situação Reporte", value=st.session_state.form_data.get('Situação Reporte', ''))
            data['Categoria'] = st.text_input("Categoria", value=st.session_state.form_data.get('Categoria', ''))
            data['Dano'] = st.text_input("Dano", value=st.session_state.form_data.get('Dano', ''))
            
            # Campo Afastado
            afastado_options = ["", "Sim", "Não", "Não informado"]
            afastado_default = st.session_state.form_data.get('Afastado', '')
            if afastado_default in afastado_options:
                afastado_index = afastado_options.index(afastado_default)
            else:
                afastado_index = 0
            data['Afastado'] = st.selectbox("Afastado", options=afastado_options, index=afastado_index)
            
            data['Potencial Acidente'] = st.text_input("Potencial Acidente", value=st.session_state.form_data.get('Potencial Acidente', ''))
        
        # Botão para fazer predição
        submitted = st.form_submit_button("🚀 Fazer Predição", use_container_width=True)

        def checarSituacaoReporte(pred):
            if(pred==1):
                return 'Reportável'
            elif(pred==0):
                return 'Não Reportável'
            else:
                return 'Indefinido'
            
        def reordenar_dict(dados, ordem):
            """
            Reordena um dicionário seguindo a ordem especificada.
            Os atributos que não estiverem na lista de ordem serão colocados no final.
            """
            return {chave: dados.get(chave) for chave in ordem if chave in dados} | {
                k: v for k, v in dados.items() if k not in ordem
            }
        
        # Ordem desejada (conforme seu exemplo de Sinop/Armac)
        ordem = [
            'Unidade', 'Empresa', 'Tipo Funcionário', 'Setor Ocorrência', 'Turno',
            'Cargo', 'Área', 'Local', 'Parte do Corpo Atingida', 'Categoria do Risco',
            'Acidente', 'Agente Causador', 'Sexo', 'Tempo de empresa', 'Tipo de Acidente',
            'Motivo', 'Gerência', 'Categoria', 'Dano', 'Afastado', 'Potencial Acidente',
            'Outra Empresa'
        ]
        
        if submitted:
            # Verificar se todos os campos obrigatórios foram preenchidos
            campos_vazios = [campo for campo, valor in data.items() if valor == "" or valor is None]
            
            if campos_vazios:
                st.error(f"⚠️ Por favor, preencha os seguintes campos: {', '.join(campos_vazios)}")
            else:
                # Preparar dados para predição
                try:
                    # Converter para DataFrame
                    df_input = pd.DataFrame([data])

                    targetIndex = df.columns.get_loc("Situação Reporte")
                    print(targetIndex)
                    X = df.drop(df.columns[targetIndex], axis=1) # todas as colunas, menos o alvo
                    
                    onehotencoder_X = ColumnTransformer(
                        transformers=[
                            ("OneHot", OneHotEncoder(handle_unknown='ignore'), 
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])], remainder='passthrough')
                    
                    X = onehotencoder_X.fit_transform(X)

                    print('----------------------------------------')
                    data.pop("Situação Reporte", None)  # None evita erro caso a chave não exista
                    data = reordenar_dict(data, ordem)
                    print(data)
                    print('----------------------------------------')

                    data = list(data.values())
                    data = np.array([data]) 

                    novo_transformado = onehotencoder_X.transform(data)
                    print(novo_transformado)
                    pred_proba = classificador.predict(novo_transformado)
                    pred_classe = np.argmax(pred_proba, axis=1)
                    print("Classe prevista:", checarSituacaoReporte(pred_classe[0]))
                    
                    st.success("✅ Predição: "+str(checarSituacaoReporte(pred_classe[0])))
                    #st.success("✅ Dados coletados com sucesso!")

                
                except Exception as e:
                    st.error(f"❌ Erro ao processar dados: {str(e)}")

    # Informações adicionais na sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Informações")
    st.sidebar.info("""
    **Campos Obrigatórios:**
    - Todos os 23 campos devem ser preenchidos
    - Tempo de empresa deve ser numérico
    - Campos Sim/Não têm seleção predefinida

    **Formatos de Modelo Suportados:**
    - TensorFlow/Keras (.h5, .keras)
    - Scikit-learn (.pkl, .joblib)
    - Modelos personalizados (.pkl)
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Sistema de Predição de Reportes | Desenvolvido com Streamlit
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Desenvolvedores: Clayton Kossoski, Endi Danila de Souza da Silva, Kokouvi Hola Kanyi Kodjovi**")