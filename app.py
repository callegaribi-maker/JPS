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
    "cada um automaticamente pelo nome e monta um único gráfico "
    "sincronizado, com deslocamento, aceleração e giroscópio."
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


def eh_coluna_acel_abs(col):
    return col.strip().endswith("a(abs)")


def tempo_em_segundos(col_tempo, serie_tempo):
    if "ms" in col_tempo.lower():
        return serie_tempo.astype(float) / 1000.0
    return serie_tempo.astype(float)


def detectar_pico_primeiro_movimento(tempo, valor, t_inicio=0.0, t_fim=None):
    """Acha o instante de maior desvio (em módulo, positivo ou negativo)
    em relação à mediana do sinal, dentro da janela [t_inicio, t_fim].
    Usado para localizar o pico de aceleração do primeiro movimento."""
    tempo = np.asarray(tempo, dtype=float)
    valor = np.asarray(valor, dtype=float)
    if t_fim is None:
        t_fim = tempo.max()
    mask = (tempo >= t_inicio) & (tempo <= t_fim)
    if mask.sum() == 0:
        return None, None
    tempo_janela = tempo[mask]
    valor_janela = valor[mask]
    baseline = np.median(valor_janela)
    idx_local = np.argmax(np.abs(valor_janela - baseline))
    return float(tempo_janela[idx_local]), float(valor_janela[idx_local])


def sugerir_corte(df, col_tempo, n_mad=8.0, fracao_cauda=0.05):
    """Sugere até que tempo usar os dados de um arquivo, detectando um
    salto anômalo apenas na 'cauda' final (últimos `fracao_cauda` da
    duração) — por exemplo, quando a câmera do Kinem para de rastrear no
    fim da gravação. Não mexe em anomalias no meio do sinal (que podem
    ser movimento real)."""
    tempo = df[col_tempo].values.astype(float)
    n = len(df)
    inicio_cauda = int(n * (1 - fracao_cauda))
    colunas_num = [c for c in df.columns if c != col_tempo and pd.api.types.is_numeric_dtype(df[c])]
    if not colunas_num or inicio_cauda >= n:
        return float(tempo.max())

    scores = np.zeros(n)
    for c in colunas_num:
        v = df[c].values.astype(float)
        mediana = np.median(v)
        mad = np.median(np.abs(v - mediana)) * 1.4826
        if mad == 0:
            continue
        z = np.abs(v - mediana) / mad
        scores = np.maximum(scores, z)

    scores_cauda = scores[inicio_cauda:]
    anomalos = np.where(scores_cauda > n_mad)[0]
    if len(anomalos) == 0:
        return float(tempo.max())
    idx_primeiro_anomalo = inicio_cauda + anomalos[0]
    margem = 2
    idx_corte = max(0, idx_primeiro_anomalo - margem)
    return float(tempo[idx_corte])


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

# --- Corte de artefatos no final dos arquivos (ex: quando a câmera do Kinem para) ---
st.header("✂️ Cortar dados no final (remover artefatos)")
st.write(
    "Se algum arquivo tiver um pico estranho no final (ex: quando a "
    "câmera do Kinem para de rastrear), defina até que segundo usar os "
    "dados daquele arquivo — o restante é descartado."
)

tempo_col_bruto = {fonte: df.columns[0] for fonte, df in dataframes.items()}
tempo_seg_bruto = {
    fonte: tempo_em_segundos(tempo_col_bruto[fonte], dataframes[fonte][tempo_col_bruto[fonte]])
    for fonte in dataframes
}

cortes_colunas = st.columns(len(dataframes)) if dataframes else []
cortes = {}
for col_layout, fonte in zip(cortes_colunas, dataframes.keys()):
    with col_layout:
        duracao_total = float(tempo_seg_bruto[fonte].max())
        sugestao_corte = min(
            sugerir_corte(dataframes[fonte], tempo_col_bruto[fonte]),
            duracao_total,
        )
        if sugestao_corte < duracao_total:
            st.caption(f"⚠️ Artefato detectado — corte sugerido: {sugestao_corte:.2f}s")
        cortes[fonte] = st.number_input(
            f"{fonte}: usar até (s)",
            min_value=0.0,
            max_value=duracao_total,
            value=sugestao_corte,
            step=0.5,
            key=f"corte_{fonte}",
        )

for fonte in list(dataframes.keys()):
    mask_corte = tempo_seg_bruto[fonte] <= cortes[fonte]
    dataframes[fonte] = dataframes[fonte][mask_corte].reset_index(drop=True)

with st.expander("📄 Ver dados brutos (opcional)"):
    abas = st.tabs(list(dataframes.keys()))
    for aba, (nome, df) in zip(abas, dataframes.items()):
        with aba:
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Baixar como CSV",
                data=csv_bytes,
                file_name=f"{nome.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                mime="text/csv",
                key=f"download_{nome}",
            )

tempo_col_por_fonte = {fonte: df.columns[0] for fonte, df in dataframes.items()}
tempo_seg_por_fonte = {
    fonte: tempo_em_segundos(tempo_col_por_fonte[fonte], dataframes[fonte][tempo_col_por_fonte[fonte]])
    for fonte in dataframes
}

# --- Sincronização temporal entre Kinem e celulares ---
st.header("🔄 Sincronização temporal")
st.write(
    "O Kinem e cada celular começam a gravar em momentos diferentes. "
    "A sincronização é feita pelo **pico de aceleração do primeiro "
    "movimento** — informe em que janela de tempo esse primeiro "
    "movimento aparece em cada arquivo (olhando o gráfico combinado "
    "logo abaixo) e o app alinha os relógios por esse pico."
)

if "offsets" not in st.session_state:
    st.session_state.offsets = {}
if "braco_offset" not in st.session_state:
    st.session_state.braco_offset = 0.0
if "punho_offset" not in st.session_state:
    st.session_state.punho_offset = 0.0

ref_tempo, ref_valor, ref_nome = None, None, None
if "Kinem" in dataframes:
    df_kinem = dataframes["Kinem"]
    opcoes_ref = {}
    candidatos_acel = [c for c in df_kinem.columns if eh_coluna_acel_abs(c)]
    preferido_acel = next((c for c in candidatos_acel if "punho" in c.lower()), None) or (candidatos_acel[0] if candidatos_acel else None)
    if preferido_acel:
        opcoes_ref[f"Aceleração — {preferido_acel}"] = preferido_acel

    candidatos_desloc = [c for c in df_kinem.columns if eh_coluna_posicao_y(c)]
    preferido_desloc = next((c for c in candidatos_desloc if "punho" in c.lower()), None) or (candidatos_desloc[0] if candidatos_desloc else None)
    if preferido_desloc:
        opcoes_ref[f"Deslocamento — {preferido_desloc}"] = preferido_desloc

    if opcoes_ref:
        indice_padrao = 0
        for i, k in enumerate(opcoes_ref.keys()):
            if k.startswith("Aceleração"):
                indice_padrao = i
                break
        escolha_ref = st.selectbox(
            "Sinal do Kinem usado como referência",
            list(opcoes_ref.keys()),
            index=indice_padrao,
        )
        col_ref = opcoes_ref[escolha_ref]
        ref_tempo = tempo_seg_por_fonte["Kinem"].values
        ref_valor = df_kinem[col_ref].values
        ref_nome = col_ref

if ref_tempo is not None:
    st.caption(f"Referência: Kinem — {ref_nome}")

    st.markdown("**Janela do primeiro movimento (em segundos, tempo original de cada arquivo)**")
    jc1, jc2, jc3 = st.columns(3)
    with jc1:
        st.caption("Kinem")
        kinem_ini = st.number_input("Início (Kinem)", value=0.0, step=0.5, key="kinem_ini")
        kinem_fim = st.number_input("Fim (Kinem)", value=15.0, step=0.5, key="kinem_fim")
    with jc2:
        st.caption("Braço (celular)")
        braco_ini = st.number_input("Início (Braço)", value=0.0, step=0.5, key="braco_ini")
        braco_fim = st.number_input("Fim (Braço)", value=15.0, step=0.5, key="braco_fim")
    with jc3:
        st.caption("Punho (celular)")
        punho_ini = st.number_input("Início (Punho)", value=0.0, step=0.5, key="punho_ini")
        punho_fim = st.number_input("Fim (Punho)", value=15.0, step=0.5, key="punho_fim")

    janelas = {"Braço": (braco_ini, braco_fim), "Punho": (punho_ini, punho_fim)}

    if st.button("🎯 Sincronizar pelo pico do primeiro movimento"):
        t_pico_kinem, v_pico_kinem = detectar_pico_primeiro_movimento(
            ref_tempo, ref_valor, kinem_ini, kinem_fim
        )
        if t_pico_kinem is None:
            st.warning("Não encontrei dados do Kinem na janela informada.")
        else:
            st.session_state.pico_referencia = t_pico_kinem
            st.info(f"Pico do Kinem em t = {t_pico_kinem:.2f} s (valor {v_pico_kinem:.2f})")
            for grupo in ["Braço", "Punho"]:
                fonte_acel = f"{grupo} - Acelerômetro"
                if fonte_acel in dataframes:
                    df_acel = dataframes[fonte_acel]
                    if "Y" in df_acel.columns:
                        sinal_celular = df_acel["Y"].values
                    else:
                        sinal_celular = df_acel[df_acel.columns[-1]].values
                    t_ini, t_fim = janelas[grupo]
                    t_pico_celular, v_pico_celular = detectar_pico_primeiro_movimento(
                        tempo_seg_por_fonte[fonte_acel].values, sinal_celular, t_ini, t_fim
                    )
                    if t_pico_celular is None:
                        st.warning(f"{grupo}: não encontrei dados na janela informada.")
                        continue
                    offset = t_pico_kinem - t_pico_celular
                    if grupo == "Braço":
                        st.session_state.braco_offset = offset
                    else:
                        st.session_state.punho_offset = offset
                    st.success(
                        f"{grupo}: pico em t = {t_pico_celular:.2f} s (valor {v_pico_celular:.2f}) "
                        f"→ deslocamento aplicado = {offset:.2f} s"
                    )

st.markdown(
    f"**Deslocamento aplicado** — Braço: `{st.session_state.braco_offset:.2f}s` · "
    f"Punho: `{st.session_state.punho_offset:.2f}s`"
)
offsets = {"Braço": st.session_state.braco_offset, "Punho": st.session_state.punho_offset}


def tempo_ajustado(fonte):
    grupo = grupo_dispositivo(fonte)
    offset = offsets.get(grupo, 0.0) if grupo else 0.0
    return tempo_seg_por_fonte[fonte] + offset


# --- Gráfico único combinado (3 eixos Y) ---
st.header("📈 Gráfico combinado")

series_disponiveis = []
for fonte, df in dataframes.items():
    col_tempo = tempo_col_por_fonte[fonte]
    for col in df.columns:
        if col == col_tempo:
            continue
        series_disponiveis.append((f"{fonte} — {col}", fonte, col))

rotulos_disponiveis = [s[0] for s in series_disponiveis]
mapa_series = {s[0]: (s[1], s[2]) for s in series_disponiveis}

# Define, para cada série, a que eixo ela pertence por padrão
EIXO_DESLOC, EIXO_ACEL, EIXO_GYRO = "y1", "y2", "y3"

def eixo_padrao(fonte, col):
    if fonte == "Kinem":
        if eh_coluna_posicao_y(col):
            return EIXO_DESLOC
        if eh_coluna_acel_y(col):
            return EIXO_ACEL
    if "Acelerômetro" in fonte and col.strip() == "Y":
        return EIXO_ACEL
    if "Giroscópio" in fonte and col.strip() == "Y":
        return EIXO_GYRO
    return None

sugestao_default = []
for rotulo, fonte, col in series_disponiveis:
    if fonte == "Kinem" and eh_coluna_acel_y(col) and "punho" in col.lower():
        sugestao_default.append(rotulo)
    elif "Acelerômetro" in fonte and col.strip() == "Y":
        sugestao_default.append(rotulo)

selecionadas = st.multiselect(
    "Séries no gráfico",
    options=rotulos_disponiveis,
    default=sugestao_default,
)

if not selecionadas:
    st.info("Selecione ao menos uma série para plotar.")
    st.stop()

zoom_no_pico = False
if "pico_referencia" in st.session_state:
    zoom_no_pico = st.checkbox(
        f"🔍 Dar zoom perto do pico de referência (t = {st.session_state.pico_referencia:.2f} s)",
        value=False,
    )

st.markdown("**Janela de tempo mostrada no gráfico**")
jw1, jw2 = st.columns(2)
janela_ini = jw1.number_input("Ver de (s)", value=0.0, step=1.0, key="janela_ini")
janela_fim = jw2.number_input("até (s)", value=20.0, step=1.0, key="janela_fim")

CORES = {
    EIXO_DESLOC: "#1f77b4",
    EIXO_ACEL: "#d62728",
    EIXO_GYRO: "#2ca02c",
}

fig = go.Figure()
eixos_usados = set()
for rotulo in selecionadas:
    fonte, col = mapa_series[rotulo]
    eixo = eixo_padrao(fonte, col) or EIXO_ACEL
    eixos_usados.add(eixo)
    df = dataframes[fonte]
    x = tempo_ajustado(fonte)
    fig.add_trace(
        go.Scatter(
            x=x, y=df[col], mode="lines", name=rotulo,
            yaxis=eixo,
        )
    )

layout_kwargs = dict(
    height=600,
    xaxis=dict(title="Tempo (s, sincronizado)", domain=[0.0, 1.0]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)

if zoom_no_pico and "pico_referencia" in st.session_state:
    t_ref = st.session_state.pico_referencia
    layout_kwargs["xaxis"]["range"] = [t_ref - 5, t_ref + 5]
else:
    layout_kwargs["xaxis"]["range"] = [janela_ini, janela_fim]

if EIXO_DESLOC in eixos_usados:
    layout_kwargs["yaxis"] = dict(
        title=dict(text="Deslocamento vertical", font=dict(color=CORES[EIXO_DESLOC])),
        tickfont=dict(color=CORES[EIXO_DESLOC]),
    )
if EIXO_ACEL in eixos_usados:
    layout_kwargs["yaxis2"] = dict(
        title=dict(text="Aceleração", font=dict(color=CORES[EIXO_ACEL])),
        tickfont=dict(color=CORES[EIXO_ACEL]),
        overlaying="y", side="right",
    )
if EIXO_GYRO in eixos_usados:
    layout_kwargs["yaxis3"] = dict(
        title=dict(text="Giroscópio", font=dict(color=CORES[EIXO_GYRO])),
        tickfont=dict(color=CORES[EIXO_GYRO]),
        overlaying="y", side="right", position=0.94,
        anchor="free",
    )

fig.update_layout(**layout_kwargs)

if "pico_referencia" in st.session_state:
    fig.add_vline(
        x=st.session_state.pico_referencia,
        line_dash="dash",
        line_color="gray",
        annotation_text="pico de referência",
        annotation_position="top",
    )

st.plotly_chart(fig, use_container_width=True)
