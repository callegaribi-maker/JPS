import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Visualizador de Dados", layout="wide")

st.title("📊 Visualizador de Dados — Kinem x Celulares")
st.write(
    "Arraste os **5 arquivos de uma vez** (1 do Kinem + 4 dos celulares: "
    "Braço-Acel, Braço-Gyro, Punho-Acel, Punho-Gyro). O app identifica "
    "cada um automaticamente pelo nome do arquivo e ajuda a sincronizar "
    "os relógios de cada dispositivo."
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


def grupo_dispositivo(fonte):
    """Agrupa arquivos do mesmo celular (Braço ou Punho) para compartilhar
    o mesmo deslocamento de sincronização."""
    if fonte.startswith("Braço"):
        return "Braço"
    if fonte.startswith("Punho"):
        return "Punho"
    return None


def ler_arquivo(uploaded_file):
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
    c = col.strip()
    return c.endswith("Y") and not c.endswith(")")


def eh_coluna_acel_y(col):
    return col.strip().endswith("a(Y)")


def tempo_em_segundos(col_tempo, serie_tempo):
    """Converte a coluna de tempo para segundos, assumindo ms quando o
    nome da coluna sugerir isso (ex: 'TempoMs')."""
    if "ms" in col_tempo.lower():
        return serie_tempo.astype(float) / 1000.0
    return serie_tempo.astype(float)


def eh_coluna_acel_abs(col):
    return col.strip().endswith("a(abs)")


def detectar_offset(tempo_ref, valor_ref, tempo_alvo, valor_alvo, busca_max=90.0, dt=0.05, min_pontos=200):
    """Encontra o deslocamento (em segundos) que melhor alinha valor_alvo
    com valor_ref, testando deslocamentos entre -busca_max e +busca_max,
    via correlação de Pearson em uma grade de tempo comum. Retorna
    (offset, correlação) — a correlação fica sempre entre -1 e 1 e serve
    como indicador de confiança (valores baixos indicam sincronização
    pouco confiável, especialmente em movimentos repetitivos/cíclicos)."""
    tempo_ref = np.asarray(tempo_ref, dtype=float)
    valor_ref = np.asarray(valor_ref, dtype=float)
    tempo_alvo = np.asarray(tempo_alvo, dtype=float)
    valor_alvo = np.asarray(valor_alvo, dtype=float)

    t0, t1 = tempo_ref.min(), tempo_ref.max()
    grade = np.arange(t0, t1, dt)
    if len(grade) < 10:
        return 0.0, 0.0
    ref_interp = np.interp(grade, tempo_ref, valor_ref)

    melhor_offset, melhor_corr = 0.0, -np.inf
    for offset in np.arange(-busca_max, busca_max, dt):
        alvo_interp = np.interp(
            grade, tempo_alvo + offset, valor_alvo,
            left=np.nan, right=np.nan,
        )
        mask = ~np.isnan(alvo_interp)
        if mask.sum() < min_pontos:
            continue
        r_sub = ref_interp[mask]
        a_sub = alvo_interp[mask]
        if np.std(r_sub) == 0 or np.std(a_sub) == 0:
            continue
        corr = np.corrcoef(r_sub, a_sub)[0, 1]
        if corr > melhor_corr:
            melhor_corr, melhor_offset = corr, offset
    return float(melhor_offset), float(melhor_corr)


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

dataframes = {}
erros = {}
for arq in arquivos_enviados:
    categoria = classificacoes[arq.name]
    try:
        df = ler_arquivo(arq)
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

# --- Tempo (coluna + versão em segundos) por fonte ---
tempo_col_por_fonte = {fonte: df.columns[0] for fonte, df in dataframes.items()}
with st.expander("⚙️ Configurações avançadas (coluna de tempo por arquivo)"):
    for fonte, df in dataframes.items():
        nova_col = st.selectbox(
            f"Coluna de tempo — {fonte}",
            list(df.columns),
            index=list(df.columns).index(tempo_col_por_fonte[fonte]),
            key=f"tempo_{fonte}",
        )
        tempo_col_por_fonte[fonte] = nova_col

tempo_seg_por_fonte = {
    fonte: tempo_em_segundos(tempo_col_por_fonte[fonte], dataframes[fonte][tempo_col_por_fonte[fonte]])
    for fonte in dataframes
}

# --- Sincronização temporal entre Kinem e celulares ---
st.header("🔄 Sincronização temporal")
st.write(
    "O Kinem e cada celular começam a gravar em momentos diferentes. "
    "A detecção automática compara a **magnitude** da aceleração "
    "(√(X²+Y²+Z²) do celular vs. a(abs) do Kinem) — essa métrica não "
    "depende da orientação do aparelho, o que a torna mais confiável "
    "que comparar eixos individuais."
)
st.caption(
    "⚠️ Como o movimento é repetitivo (várias flexões parecidas), a "
    "correlação automática pode travar na repetição errada. Sempre "
    "confira visualmente no Gráfico 2 depois e ajuste manualmente os "
    "campos abaixo se os picos não coincidirem — quanto mais próxima "
    "de 1,0 a correlação mostrada, maior a confiança no resultado "
    "automático."
)

if "offsets" not in st.session_state:
    st.session_state.offsets = {"Braço": 0.0, "Punho": 0.0}

# Referência: magnitude da aceleração do Kinem no punho (a(abs)), que é
# invariante à orientação — mais comparável à magnitude do acelerômetro do celular.
ref_tempo, ref_valor, ref_nome = None, None, None
if "Kinem" in dataframes:
    df_kinem = dataframes["Kinem"]
    candidatos = [c for c in df_kinem.columns if eh_coluna_acel_abs(c)]
    preferida = next((c for c in candidatos if "punho" in c.lower()), None)
    col_ref = preferida or (candidatos[0] if candidatos else None)
    if col_ref:
        ref_tempo = tempo_seg_por_fonte["Kinem"].values
        ref_valor = df_kinem[col_ref].values
        ref_nome = col_ref

if ref_tempo is None:
    st.warning("Não encontrei uma coluna de aceleração (a(abs)) no Kinem para servir de referência.")
else:
    st.caption(f"Referência: Kinem — {ref_nome}")

    if st.button("🔍 Detectar sincronização automaticamente"):
        for grupo in ["Braço", "Punho"]:
            fonte_acel = f"{grupo} - Acelerômetro"
            if fonte_acel in dataframes:
                df_acel = dataframes[fonte_acel]
                cols_xyz = [c for c in ["X", "Y", "Z"] if c in df_acel.columns]
                if len(cols_xyz) == 3:
                    magnitude = np.sqrt((df_acel[cols_xyz] ** 2).sum(axis=1)).values
                else:
                    magnitude = df_acel[df_acel.columns[-1]].values
                offset, corr = detectar_offset(
                    ref_tempo, ref_valor,
                    tempo_seg_por_fonte[fonte_acel].values,
                    magnitude,
                )
                st.session_state.offsets[grupo] = offset
                nivel = "boa" if corr > 0.5 else ("fraca" if corr > 0.25 else "muito fraca — ajuste manualmente")
                st.success(f"{grupo}: deslocamento sugerido = {offset:.2f} s (correlação {corr:.2f} — confiança {nivel})")

    col_a, col_b = st.columns(2)
    st.session_state.offsets["Braço"] = col_a.number_input(
        "Deslocamento — Braço (s)",
        value=float(st.session_state.offsets["Braço"]),
        step=0.05, format="%.2f",
    )
    st.session_state.offsets["Punho"] = col_b.number_input(
        "Deslocamento — Punho (s)",
        value=float(st.session_state.offsets["Punho"]),
        step=0.05, format="%.2f",
    )

offsets = st.session_state.get("offsets", {"Braço": 0.0, "Punho": 0.0})


def tempo_ajustado(fonte):
    grupo = grupo_dispositivo(fonte)
    offset = offsets.get(grupo, 0.0) if grupo else 0.0
    return tempo_seg_por_fonte[fonte] + offset


# --- Configuração e montagem dos 3 gráficos ---
st.header("📈 Gráficos")

series_disponiveis = []
for fonte, df in dataframes.items():
    col_tempo = tempo_col_por_fonte[fonte]
    for col in df.columns:
        if col == col_tempo:
            continue
        series_disponiveis.append((f"{fonte} — {col}", fonte, col))

rotulos_disponiveis = [s[0] for s in series_disponiveis]
mapa_series = {s[0]: (s[1], s[2]) for s in series_disponiveis}

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
        x = tempo_ajustado(fonte)
        fig.add_trace(
            go.Scatter(x=x, y=df[col], mode="lines", name=rotulo)
        )
    fig.update_layout(
        title=titulos_grafico[i],
        xaxis_title="Tempo (s, sincronizado)",
        yaxis_title="Valor",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
