import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import io

st.set_page_config(page_title="Visualizador de Dados", layout="wide")

st.title("🦾 Flexão do Cotovelo — Kinem x Celulares")
st.write(
    "Arraste os **5 arquivos de uma vez** (1 do Kinem + 4 dos celulares: "
    "Braço-Acel, Braço-Gyro, Punho-Acel, Punho-Gyro). O app identifica, "
    "sincroniza e corta artefatos automaticamente, e mostra direto a "
    "análise de flexão do cotovelo por trial."
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


def calcular_angulo_segmento(df, marcador_a, marcador_b, eixo_vertical="Y"):
    """Calcula o ângulo (em graus) do segmento entre dois marcadores do
    Kinem em relação à vertical, a partir das colunas X/Y/Z de cada um.
    Esse ângulo reflete a postura mantida (não só o instante do
    movimento), de forma parecida com o que o acelerômetro do celular
    capta pela reorientação em relação à gravidade."""
    cols_a = [f"{marcador_a} X", f"{marcador_a} Y", f"{marcador_a} Z"]
    cols_b = [f"{marcador_b} X", f"{marcador_b} Y", f"{marcador_b} Z"]
    if not all(c in df.columns for c in cols_a + cols_b):
        return None
    p_a = df[cols_a].values
    p_b = df[cols_b].values
    vetor = p_b - p_a
    norma = np.linalg.norm(vetor, axis=1, keepdims=True)
    norma[norma == 0] = np.nan
    vetor_unit = vetor / norma
    idx_vertical = {"X": 0, "Y": 1, "Z": 2}[eixo_vertical]
    vertical = np.zeros(3)
    vertical[idx_vertical] = 1.0
    cos_angulo = vetor_unit @ vertical
    angulo = np.degrees(np.arccos(np.clip(cos_angulo, -1, 1)))
    return angulo


def calcular_flexao_cotovelo_kinem(df):
    """Ângulo de flexão do cotovelo (graus) a partir dos 3 marcadores do
    Kinem: 0° = extensão total, valores maiores = mais flexão."""
    cols_ombro = ["Acrômio dir. X", "Acrômio dir. Y", "Acrômio dir. Z"]
    cols_cotovelo = ["Epicôndilo lateral dir. X", "Epicôndilo lateral dir. Y", "Epicôndilo lateral dir. Z"]
    cols_punho = ["Medial do punho dir. X", "Medial do punho dir. Y", "Medial do punho dir. Z"]
    if not all(c in df.columns for c in cols_ombro + cols_cotovelo + cols_punho):
        return None
    ombro = df[cols_ombro].values
    cotovelo = df[cols_cotovelo].values
    punho = df[cols_punho].values
    v1 = ombro - cotovelo
    v2 = punho - cotovelo
    norma1 = np.linalg.norm(v1, axis=1)
    norma2 = np.linalg.norm(v2, axis=1)
    cos_ang = np.sum(v1 * v2, axis=1) / (norma1 * norma2)
    angulo_entre = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
    return 180.0 - angulo_entre


def calcular_tilt_celular(df):
    """Inclinação (graus) do eixo Y do celular em relação à vertical,
    a partir do acelerômetro (aproximação quase-estática)."""
    if not all(c in df.columns for c in ["X", "Y", "Z"]):
        return None
    v = df[["X", "Y", "Z"]].values
    mag = np.linalg.norm(v, axis=1)
    cos_a = v[:, 1] / mag
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def calcular_flexao_celular(tempo_braco, tilt_braco, tempo_punho, tilt_punho):
    """Estima a flexão do cotovelo pelos celulares como a diferença entre
    a inclinação do celular do braço e do punho, numa grade de tempo
    comum (após sincronização)."""
    t0 = max(tempo_braco.min(), tempo_punho.min())
    t1 = min(tempo_braco.max(), tempo_punho.max())
    if t1 <= t0:
        return None, None
    grade = np.arange(t0, t1, 0.02)
    tilt_braco_i = np.interp(grade, tempo_braco, tilt_braco)
    tilt_punho_i = np.interp(grade, tempo_punho, tilt_punho)
    flexao = np.abs(tilt_braco_i - tilt_punho_i)
    return grade, flexao


def detectar_trials(tempo, sinal, prominence=15.0, distance_s=3.0):
    """Detecta cada 'trial' (repetição) como um pico do sinal, separando
    os trials pelos pontos médios entre picos consecutivos. Para cada
    trial calcula o pico, a ADM (máx - mín dentro do trial) e guarda o
    próprio trecho do sinal (tempo relativo ao pico) para permitir
    plotar a curva completa de cada trial, não só o valor de pico."""
    tempo = np.asarray(tempo, dtype=float)
    sinal = np.asarray(sinal, dtype=float)
    dt = np.median(np.diff(tempo))
    distance = max(1, int(distance_s / dt))
    picos, _ = find_peaks(sinal, prominence=prominence, distance=distance)
    if len(picos) == 0:
        return []
    limites = [0] + [int((picos[i] + picos[i + 1]) / 2) for i in range(len(picos) - 1)] + [len(sinal) - 1]
    trials = []
    for i, p in enumerate(picos):
        ini, fim = limites[i], limites[i + 1]
        segmento = sinal[ini:fim + 1]
        tempo_segmento = tempo[ini:fim + 1]
        trials.append({
            "trial": i + 1,
            "tempo_pico": float(tempo[p]),
            "pico": float(sinal[p]),
            "adm": float(segmento.max() - segmento.min()),
            "tempo_rel": tempo_segmento - tempo[p],
            "sinal": segmento,
        })
    return trials


def calcular_erros(trials, indice_referencia=0):
    """Adiciona erro absoluto e relativo de cada trial em relação ao
    trial de referência (por padrão, o primeiro)."""
    if not trials:
        return trials
    pico_ref = trials[indice_referencia]["pico"]
    for t in trials:
        t["erro_abs"] = abs(t["pico"] - pico_ref)
        t["erro_rel_pct"] = (t["erro_abs"] / abs(pico_ref) * 100) if pico_ref != 0 else float("nan")
    return trials


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

# Adiciona ao Kinem uma coluna calculada: ângulo do antebraço em relação
# à vertical (postura mantida), comparável ao "degrau" do acelerômetro.
if "Kinem" in dataframes:
    angulo = calcular_angulo_segmento(
        dataframes["Kinem"], "Epicôndilo lateral dir.", "Medial do punho dir."
    )
    if angulo is not None:
        dataframes["Kinem"]["Ângulo antebraço (vertical)"] = angulo

# --- Corte automático de artefatos no final (silencioso, sem interface) ---
tempo_col_bruto = {fonte: df.columns[0] for fonte, df in dataframes.items()}
tempo_seg_bruto = {
    fonte: tempo_em_segundos(tempo_col_bruto[fonte], dataframes[fonte][tempo_col_bruto[fonte]])
    for fonte in dataframes
}
for fonte in list(dataframes.keys()):
    duracao_total = float(tempo_seg_bruto[fonte].max())
    corte = min(sugerir_corte(dataframes[fonte], tempo_col_bruto[fonte]), duracao_total)
    mask_corte = tempo_seg_bruto[fonte] <= corte
    dataframes[fonte] = dataframes[fonte][mask_corte].reset_index(drop=True)

tempo_col_por_fonte = {fonte: df.columns[0] for fonte, df in dataframes.items()}
tempo_seg_por_fonte = {
    fonte: tempo_em_segundos(tempo_col_por_fonte[fonte], dataframes[fonte][tempo_col_por_fonte[fonte]])
    for fonte in dataframes
}

# --- Sincronização temporal automática (silenciosa, sem interface) ---
JANELA_PADRAO = 15.0  # segundos, usada internamente para achar o 1º pico

braco_offset, punho_offset = 0.0, 0.0
pico_referencia = None

if "Kinem" in dataframes:
    df_kinem = dataframes["Kinem"]
    candidatos_acel = [c for c in df_kinem.columns if eh_coluna_acel_abs(c)]
    col_ref = next((c for c in candidatos_acel if "punho" in c.lower()), None) or (candidatos_acel[0] if candidatos_acel else None)
    if col_ref:
        ref_tempo = tempo_seg_por_fonte["Kinem"].values
        ref_valor = df_kinem[col_ref].values
        t_pico_kinem, _ = detectar_pico_primeiro_movimento(ref_tempo, ref_valor, 0.0, JANELA_PADRAO)
        if t_pico_kinem is not None:
            pico_referencia = t_pico_kinem
            for grupo in ["Braço", "Punho"]:
                fonte_acel = f"{grupo} - Acelerômetro"
                if fonte_acel in dataframes:
                    df_acel = dataframes[fonte_acel]
                    sinal_celular = df_acel["Y"].values if "Y" in df_acel.columns else df_acel[df_acel.columns[-1]].values
                    t_pico_celular, _ = detectar_pico_primeiro_movimento(
                        tempo_seg_por_fonte[fonte_acel].values, sinal_celular, 0.0, JANELA_PADRAO
                    )
                    if t_pico_celular is not None:
                        offset = t_pico_kinem - t_pico_celular
                        if grupo == "Braço":
                            braco_offset = offset
                        else:
                            punho_offset = offset

offsets = {"Braço": braco_offset, "Punho": punho_offset}


def tempo_ajustado(fonte):
    grupo = grupo_dispositivo(fonte)
    offset = offsets.get(grupo, 0.0) if grupo else 0.0
    return tempo_seg_por_fonte[fonte] + offset


# --- Ângulo de flexão do cotovelo: trials, ADM e erro vs trial 1 ---
st.header("🦾 Flexão do cotovelo — trials, ADM e erro")

flexao_kinem = None
tempo_flexao_kinem = None
if "Kinem" in dataframes:
    flexao_kinem = calcular_flexao_cotovelo_kinem(dataframes["Kinem"])
    if flexao_kinem is not None:
        tempo_flexao_kinem = tempo_seg_por_fonte["Kinem"].values

flexao_celular = None
tempo_flexao_celular = None
if "Braço - Acelerômetro" in dataframes and "Punho - Acelerômetro" in dataframes:
    tilt_braco = calcular_tilt_celular(dataframes["Braço - Acelerômetro"])
    tilt_punho = calcular_tilt_celular(dataframes["Punho - Acelerômetro"])
    if tilt_braco is not None and tilt_punho is not None:
        t_braco_sync = tempo_ajustado("Braço - Acelerômetro").values
        t_punho_sync = tempo_ajustado("Punho - Acelerômetro").values
        tempo_flexao_celular, flexao_celular = calcular_flexao_celular(
            t_braco_sync, tilt_braco, t_punho_sync, tilt_punho
        )

if flexao_kinem is None and flexao_celular is None:
    st.info("Não foi possível calcular o ângulo de flexão (verifique se os arquivos certos foram enviados).")
else:
    st.caption(
        "Kinem: ângulo articular real (0° = extensão total). Celular: "
        "estimativa pela diferença de inclinação entre o celular do "
        "Braço e o do Punho — é uma aproximação, não o mesmo cálculo do "
        "Kinem, então compare a *consistência entre trials* de cada "
        "dispositivo, não o valor absoluto entre eles."
    )

    pc1, pc2 = st.columns(2)
    prominence = pc1.number_input("Sensibilidade do pico (prominence, °)", value=15.0, step=1.0, min_value=1.0)
    distance_s = pc2.number_input("Distância mínima entre trials (s)", value=3.0, step=0.5, min_value=0.5)

    trials_kinem = detectar_trials(tempo_flexao_kinem, flexao_kinem, prominence, distance_s) if flexao_kinem is not None else []
    trials_celular = detectar_trials(tempo_flexao_celular, flexao_celular, prominence, distance_s) if flexao_celular is not None else []

    # --- Registro contínuo, com os trials marcados ---
    st.subheader("Registro contínuo (com os trials marcados)")
    fig_continuo = go.Figure()
    if flexao_kinem is not None:
        fig_continuo.add_trace(go.Scatter(
            x=tempo_flexao_kinem, y=flexao_kinem, mode="lines",
            name="Kinem", line=dict(color="#1f77b4"),
        ))
        if trials_kinem:
            fig_continuo.add_trace(go.Scatter(
                x=[t["tempo_pico"] for t in trials_kinem],
                y=[t["pico"] for t in trials_kinem],
                mode="markers+text",
                text=[f"T{t['trial']}" for t in trials_kinem],
                textposition="top center",
                marker=dict(color="#1f77b4", size=9, symbol="diamond"),
                name="Trials (Kinem)",
            ))
    if flexao_celular is not None:
        fig_continuo.add_trace(go.Scatter(
            x=tempo_flexao_celular, y=flexao_celular, mode="lines",
            name="Celular (estimado)", line=dict(color="#d62728"),
        ))
        if trials_celular:
            fig_continuo.add_trace(go.Scatter(
                x=[t["tempo_pico"] for t in trials_celular],
                y=[t["pico"] for t in trials_celular],
                mode="markers+text",
                text=[f"T{t['trial']}" for t in trials_celular],
                textposition="bottom center",
                marker=dict(color="#d62728", size=9, symbol="diamond"),
                name="Trials (Celular)",
            ))
    fig_continuo.update_layout(
        xaxis_title="Tempo (s)", yaxis_title="Ângulo (°)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_continuo, use_container_width=True)

    col_k, col_c = st.columns(2)

    with col_k:
        st.subheader("Kinem")
        if trials_kinem:
            todos_k = [t["trial"] for t in trials_kinem]
            incluidos_k = st.multiselect(
                "Trials incluídos na análise",
                options=todos_k,
                default=todos_k,
                key="incluidos_kinem",
            )
            trials_kinem_f = [t for t in trials_kinem if t["trial"] in incluidos_k]
            if trials_kinem_f:
                opcoes_trial = [f"Trial {t['trial']}" for t in trials_kinem_f]
                indice_default = 1 if len(opcoes_trial) > 1 else 0
                ref_idx_k = st.selectbox("Trial de referência", opcoes_trial, index=indice_default, key="ref_kinem")
                idx_k = opcoes_trial.index(ref_idx_k)
                trials_kinem_f = calcular_erros(trials_kinem_f, idx_k)
                df_trials_k = pd.DataFrame(trials_kinem_f)[["trial", "tempo_pico", "pico", "adm", "erro_abs", "erro_rel_pct"]]
                df_trials_k.columns = ["Trial", "t pico (s)", "Pico (°)", "ADM (°)", "Erro abs (°)", "Erro rel (%)"]
                st.dataframe(df_trials_k.round(2), use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum trial incluído.")
        else:
            trials_kinem_f = []
            st.info("Nenhum trial detectado — ajuste a sensibilidade acima.")

    with col_c:
        st.subheader("Celular (estimado)")
        if trials_celular:
            todos_c = [t["trial"] for t in trials_celular]
            incluidos_c = st.multiselect(
                "Trials incluídos na análise",
                options=todos_c,
                default=todos_c,
                key="incluidos_celular",
            )
            trials_celular_f = [t for t in trials_celular if t["trial"] in incluidos_c]
            if trials_celular_f:
                opcoes_trial_c = [f"Trial {t['trial']}" for t in trials_celular_f]
                indice_default_c = 1 if len(opcoes_trial_c) > 1 else 0
                ref_idx_c = st.selectbox("Trial de referência", opcoes_trial_c, index=indice_default_c, key="ref_celular")
                idx_c = opcoes_trial_c.index(ref_idx_c)
                trials_celular_f = calcular_erros(trials_celular_f, idx_c)
                df_trials_c = pd.DataFrame(trials_celular_f)[["trial", "tempo_pico", "pico", "adm", "erro_abs", "erro_rel_pct"]]
                df_trials_c.columns = ["Trial", "t pico (s)", "Pico (°)", "ADM (°)", "Erro abs (°)", "Erro rel (%)"]
                st.dataframe(df_trials_c.round(2), use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum trial incluído.")
        else:
            trials_celular_f = []
            st.info("Nenhum trial detectado — ajuste a sensibilidade acima.")

    if trials_kinem_f or trials_celular_f:
        fig_trials = go.Figure()
        if trials_kinem_f:
            fig_trials.add_trace(go.Scatter(
                x=[t["trial"] for t in trials_kinem_f],
                y=[t["pico"] for t in trials_kinem_f],
                mode="lines+markers", name="Kinem",
            ))
        if trials_celular_f:
            fig_trials.add_trace(go.Scatter(
                x=[t["trial"] for t in trials_celular_f],
                y=[t["pico"] for t in trials_celular_f],
                mode="lines+markers", name="Celular (estimado)",
            ))
        fig_trials.update_layout(
            title="Ângulo de pico por trial",
            xaxis_title="Trial", yaxis_title="Ângulo de pico (°)",
            height=350,
        )
        st.plotly_chart(fig_trials, use_container_width=True)

    # --- Curvas completas de cada trial (não só o valor de pico) ---
    if trials_kinem_f or trials_celular_f:
        st.subheader("Curvas de ângulo por trial (alinhadas pelo pico)")
        st.caption(
            "Cada linha é o registro completo do ângulo durante aquele "
            "trial, com o tempo centralizado no instante do pico (t=0 "
            "= pico de flexão)."
        )
        col_ck, col_cc = st.columns(2)
        with col_ck:
            if trials_kinem_f:
                fig_k = go.Figure()
                for t in trials_kinem_f:
                    fig_k.add_trace(go.Scatter(
                        x=t["tempo_rel"], y=t["sinal"],
                        mode="lines", name=f"Trial {t['trial']}",
                    ))
                fig_k.update_layout(
                    title="Kinem", xaxis_title="Tempo relativo ao pico (s)",
                    yaxis_title="Ângulo (°)", height=400,
                )
                st.plotly_chart(fig_k, use_container_width=True)
        with col_cc:
            if trials_celular_f:
                fig_c = go.Figure()
                for t in trials_celular_f:
                    fig_c.add_trace(go.Scatter(
                        x=t["tempo_rel"], y=t["sinal"],
                        mode="lines", name=f"Trial {t['trial']}",
                    ))
                fig_c.update_layout(
                    title="Celular (estimado)", xaxis_title="Tempo relativo ao pico (s)",
                    yaxis_title="Ângulo (°)", height=400,
                )
                st.plotly_chart(fig_c, use_container_width=True)
