import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Visualizador de Dados", layout="wide")

st.title("📊 Visualizador de Dados — Kinem x Celulares")
st.write(
    "Arraste os **5 arquivos de uma vez** (1 do Kinem + 4 dos celulares: "
    "Braço-Acel, Braço-Gyro, Punho-Acel, Punho-Gyro). O app identifica "
    "cada um automaticamente pelo nome do arquivo."
)

CATEGORIAS = [
    "Kinem",
    "Braço - Acelerômetro",
    "Braço - Giroscópio",
    "Punho - Acelerômetro",
    "Punho - Giroscópio",
    "Outro",
]


def classificar_arquivo(nome_arquivo):
    """Tenta identificar a categoria do arquivo pelo nome."""
    n = nome_arquivo.lower()
    if "kinem" in n:
        return "Kinem"
    eh_braco = ("braç" in n) or ("brac" in n)
    eh_punho = "punho" in n
    eh_acel = "acel" in n
    eh_gyro = ("gyro" in n) or ("giro" in n)
    if eh_braco and eh_acel:
        return "Braço - Acelerômetro"
    if eh_braco and eh_gyro:
        return "Braço - Giroscópio"
    if eh_punho and eh_acel:
        return "Punho - Acelerômetro"
    if eh_punho and eh_gyro:
        return "Punho - Giroscópio"
    return "Outro"


def ler_arquivo(uploaded_file):
    """Lê um arquivo CSV/TXT (detecta encoding e delimitador automaticamente)."""
    conteudo = uploaded_file.getvalue()

    texto = None
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            texto = conteudo.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if texto is None:
        raise ValueError("Não foi possível decodificar o arquivo (encoding).")

    df = pd.read_csv(io.StringIO(texto), sep=None, engine="python")
    return df


def eh_coluna_posicao_y(col):
    """Coluna de posição vertical (ex: 'Medial do punho dir. Y'), não velocidade/aceleração."""
    c = col.strip()
    return c.endswith("Y") and not c.endswith(")")


def eh_coluna_acel_y(col):
    """Coluna de aceleração vertical no Kinem (ex: '... a(Y)')."""
    return col.strip().endswith("a(Y)")


# --- Upload único, com múltiplos arquivos de uma vez ---
st.sidebar.header("⚙️ Arquivos")
arquivos_enviados = st.sidebar.file_uploader(
    "Envie os 5 arquivos juntos (Kinem + 4 celulares)",
    type=["csv", "txt"],
    accept_multiple_files=True,
)

if not arquivos_enviados:
    st.info("Envie os arquivos na barra lateral para começar.")
    st.stop()

# --- Classificação automática (com opção de corrigir manualmente) ---
st.sidebar.subheader("Classificação detectada")
classificacoes = {}
for arq in arquivos_enviados:
    sugestao = classificar_arquivo(arq.name)
    escolha = st.sidebar.selectbox(
        arq.name,
        CATEGORIAS,
        index=CATEGORIAS.index(sugestao),
        key=f"cat_{arq.name}",
    )
    classificacoes[arq.name] = escolha

# --- Leitura dos arquivos ---
dataframes = {}
erros = {}
for arq in arquivos_enviados:
    categoria = classificacoes[arq.name]
    try:
        df = ler_arquivo(arq)
        # Se houver mais de um arquivo com a mesma categoria, diferencia pelo nome
        chave = categoria if categoria not in dataframes else f"{categoria} ({arq.name})"
        dataframes[chave] = df
    except Exception as e:
        erros[arq.name] = str(e)

if erros:
    for nome, msg in erros.items():
        st.error(f"Erro ao ler **{nome}**: {msg}")

# --- Dados brutos em abas ---
st.header("Dados brutos")
abas = st.tabs(list(dataframes.keys()))
for aba, (nome, df) in zip(abas, dataframes.items()):
    with aba:
        col1, col2, col3 = st.columns(3)
        col1.metric("Linhas", df.shape[0])
        col2.metric("Colunas", df.shape[1])
        col3.metric("Valores nulos", int(df.isna().sum().sum()))

        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar como CSV",
            data=csv_bytes,
            file_name=f"{nome.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
            mime="text/csv",
            key=f"download_{nome}",
        )

# --- Configuração e montagem dos 3 gráficos ---
st.header("📈 Gráficos")

# Tempo = primeira coluna de cada arquivo (padrão fixo para este formato)
tempo_por_fonte = {fonte: df.columns[0] for fonte, df in dataframes.items()}

series_disponiveis = []  # (rotulo, fonte, coluna)
for fonte, df in dataframes.items():
    col_tempo = tempo_por_fonte[fonte]
    for col in df.columns:
        if col == col_tempo:
            continue
        series_disponiveis.append((f"{fonte} — {col}", fonte, col))

rotulos_disponiveis = [s[0] for s in series_disponiveis]
mapa_series = {s[0]: (s[1], s[2]) for s in series_disponiveis}

with st.expander("⚙️ Configurações avançadas (coluna de tempo por arquivo)"):
    for fonte, df in dataframes.items():
        nova_col = st.selectbox(
            f"Coluna de tempo — {fonte}",
            list(df.columns),
            index=list(df.columns).index(tempo_por_fonte[fonte]),
            key=f"tempo_{fonte}",
        )
        tempo_por_fonte[fonte] = nova_col

# --- Sugestões automáticas por gráfico ---
sugestoes = {1: [], 2: [], 3: []}
for rotulo, fonte, col in series_disponiveis:
    if fonte == "Kinem":
        if eh_coluna_posicao_y(col) and "punho" in col.lower():
            sugestoes[1].append(rotulo)
        if eh_coluna_acel_y(col) and "punho" in col.lower():
            sugestoes[2].append(rotulo)
    elif "Acelerômetro" in fonte and col.strip() == "Y":
        sugestoes[2].append(rotulo)
    elif "Giroscópio" in fonte and col.strip() == "Y":
        sugestoes[3].append(rotulo)

# Se nenhuma coluna do Kinem com "punho" foi achada, pega a primeira disponível
if not sugestoes[1]:
    for rotulo, fonte, col in series_disponiveis:
        if fonte == "Kinem" and eh_coluna_posicao_y(col):
            sugestoes[1].append(rotulo)
            break
if not any("Kinem" in r for r in sugestoes[2]):
    for rotulo, fonte, col in series_disponiveis:
        if fonte == "Kinem" and eh_coluna_acel_y(col):
            sugestoes[2].insert(0, rotulo)
            break

titulos_grafico = {
    1: "Gráfico 1 — Deslocamento vertical (Kinem)",
    2: "Gráfico 2 — Aceleração vertical: Kinem x Celulares",
    3: "Gráfico 3 — Giroscópio Y: Braço x Punho",
}

col1, col2, col3 = st.columns(3)
colunas_layout = [col1, col2, col3]

selecoes = {}
for i, col_layout in zip([1, 2, 3], colunas_layout):
    with col_layout:
        st.subheader(titulos_grafico[i])
        selecoes[i] = st.multiselect(
            "Séries",
            options=rotulos_disponiveis,
            default=sugestoes[i],
            key=f"select_grafico_{i}",
        )

st.divider()

for i in [1, 2, 3]:
    escolhidas = selecoes[i]
    if not escolhidas:
        st.info(f"{titulos_grafico[i]}: nenhuma série selecionada.")
        continue

    fig = go.Figure()
    for rotulo in escolhidas:
        fonte, col = mapa_series[rotulo]
        df = dataframes[fonte]
        col_tempo = tempo_por_fonte[fonte]
        fig.add_trace(
            go.Scatter(
                x=df[col_tempo],
                y=df[col],
                mode="lines",
                name=rotulo,
            )
        )
    fig.update_layout(
        title=titulos_grafico[i],
        xaxis_title="Tempo",
        yaxis_title="Valor",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
