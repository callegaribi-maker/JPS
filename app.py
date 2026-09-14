import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Visualizador de Dados", layout="wide")

st.title("📊 Visualizador de Dados — Kinem x Celulares")
st.write(
    "Envie 1 arquivo do **Kinem** e até 4 arquivos de **Celular** "
    "(CSV ou TXT). Todos devem ter o tempo na primeira coluna. "
    "Depois configure os 3 gráficos escolhendo quais colunas entram em cada um."
)

# Delimitadores comuns para escolha manual (útil principalmente para .txt)
DELIMITADORES = {
    "Detectar automaticamente": None,
    "Vírgula ( , )": ",",
    "Ponto e vírgula ( ; )": ";",
    "Tabulação ( \\t )": "\t",
    "Pipe ( | )": "|",
    "Espaço": r"\s+",
}


def ler_arquivo(uploaded_file, delimitador):
    """Lê um arquivo CSV/TXT enviado e retorna um DataFrame."""
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

    if delimitador is None:
        df = pd.read_csv(io.StringIO(texto), sep=None, engine="python")
    else:
        df = pd.read_csv(io.StringIO(texto), sep=delimitador, engine="python")

    return df


def palpite_padrao(nome_arquivo, nome_coluna):
    """Sugere em qual gráfico (1, 2 ou 3) uma coluna provavelmente se encaixa,
    com base em palavras-chave comuns. Retorna None se não tiver palpite."""
    col = nome_coluna.lower()
    arq = nome_arquivo.lower()

    if "desloc" in col:
        return 1
    if ("acele" in col or "acc" in col or "aceler" in col) and "vertical" in col:
        return 2
    if "acele" in col or "acc" in col:
        if "y" in col:
            return 2
    if "girosc" in col or "gyro" in col:
        if "y" in col:
            return 3
    return None


# --- Definição das fontes de arquivo: 1 Kinem + 4 Celular ---
FONTES = ["Kinem"] + [f"Celular {i}" for i in range(1, 5)]

st.sidebar.header("⚙️ Arquivos")

uploads = {}
for fonte in FONTES:
    st.sidebar.markdown(f"**{fonte}**")
    up = st.sidebar.file_uploader(
        f"Arquivo do {fonte}",
        type=["csv", "txt"],
        key=f"upload_{fonte}",
    )
    delim_label = st.sidebar.selectbox(
        "Delimitador",
        list(DELIMITADORES.keys()),
        key=f"delim_{fonte}",
        label_visibility="collapsed",
    )
    uploads[fonte] = (up, DELIMITADORES[delim_label])
    st.sidebar.divider()

# --- Processa os arquivos enviados ---
dataframes = {}
erros = {}

for fonte, (up, delim) in uploads.items():
    if up is not None:
        try:
            df = ler_arquivo(up, delim)
            dataframes[fonte] = df
        except Exception as e:
            erros[fonte] = str(e)

if not dataframes and not erros:
    st.info("Envie pelo menos um arquivo na barra lateral para começar.")
    st.stop()

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
            file_name=f"{nome.replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"download_{nome}",
        )

# --- Configuração e montagem dos 3 gráficos ---
st.header("📈 Gráficos")

if len(dataframes) == 0:
    st.stop()

# Monta a lista de todas as combinações (arquivo, coluna) disponíveis,
# assumindo que a primeira coluna de cada arquivo é o tempo.
series_disponiveis = []  # lista de tuplas (rotulo, fonte, coluna_y)
tempo_por_fonte = {}

for fonte, df in dataframes.items():
    colunas = list(df.columns)
    if len(colunas) < 2:
        continue
    col_tempo = st.selectbox(
        f"Coluna de tempo — {fonte}",
        colunas,
        index=0,
        key=f"tempo_{fonte}",
    )
    tempo_por_fonte[fonte] = col_tempo
    for col in colunas:
        if col == col_tempo:
            continue
        rotulo = f"{fonte} — {col}"
        series_disponiveis.append((rotulo, fonte, col))

rotulos_disponiveis = [s[0] for s in series_disponiveis]
mapa_series = {s[0]: (s[1], s[2]) for s in series_disponiveis}

# Sugestões automáticas por gráfico
sugestoes = {1: [], 2: [], 3: []}
for rotulo, fonte, col in series_disponiveis:
    palpite = palpite_padrao(fonte, col)
    if palpite:
        sugestoes[palpite].append(rotulo)

titulos_grafico = {
    1: "Gráfico 1 — Deslocamento vertical (Kinem)",
    2: "Gráfico 2 — Aceleração vertical (Kinem) x Aceleração Y (celulares)",
    3: "Gráfico 3 — Giroscópio Y (celulares)",
}

colgraf1, colgraf2, colgraf3 = st.columns(3)
colunas_layout = [colgraf1, colgraf2, colgraf3]

selecoes = {}
for i, col_layout in zip([1, 2, 3], colunas_layout):
    with col_layout:
        st.subheader(titulos_grafico[i])
        selecoes[i] = st.multiselect(
            "Selecione as séries",
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
