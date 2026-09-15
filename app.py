import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc

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


def detectar_trials(tempo, sinal, prominence=15.0, distance_s=3.0, margem_extra_s=20.0):
    """Detecta cada 'trial' (repetição) como um pico do sinal, separando
    os trials pelos pontos médios entre picos consecutivos. Para cada
    trial calcula o pico e a ADM (máx - mín) usando o recorte natural
    (limitado pelos trials vizinhos). Além disso guarda uma janela mais
    ampla ao redor do pico (até `margem_extra_s`, limitada apenas pelo
    início/fim do sinal completo, não pelos vizinhos) para permitir
    visualizar mais contexto de cada trial, mesmo quando ele está perto
    de outro."""
    tempo = np.asarray(tempo, dtype=float)
    sinal = np.asarray(sinal, dtype=float)
    dt = np.median(np.diff(tempo))
    distance = max(1, int(distance_s / dt))
    picos, _ = find_peaks(sinal, prominence=prominence, distance=distance)
    if len(picos) == 0:
        return []
    limites = [0] + [int((picos[i] + picos[i + 1]) / 2) for i in range(len(picos) - 1)] + [len(sinal) - 1]
    margem_amostras = int(margem_extra_s / dt)
    trials = []
    for i, p in enumerate(picos):
        ini, fim = limites[i], limites[i + 1]
        segmento = sinal[ini:fim + 1]
        tempo_segmento = tempo[ini:fim + 1]

        ini_ext = max(0, p - margem_amostras)
        fim_ext = min(len(sinal) - 1, p + margem_amostras)
        segmento_ext = sinal[ini_ext:fim_ext + 1]
        tempo_ext = tempo[ini_ext:fim_ext + 1]

        trials.append({
            "trial": i + 1,
            "idx_pico": int(p),
            "tempo_pico": float(tempo[p]),
            "pico": float(sinal[p]),
            "adm": float(segmento.max() - segmento.min()),
            "tempo_rel": tempo_segmento - tempo[p],
            "sinal": segmento,
            "tempo_rel_ext": tempo_ext - tempo[p],
            "sinal_ext": segmento_ext,
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
def icc_2_1(data):
    """ICC(2,1): concordância absoluta, medida única, efeitos aleatórios
    de dois fatores. data: array (n_sujeitos, k_avaliadores/dispositivos)."""
    n, k = data.shape
    mean_subjects = data.mean(axis=1)
    mean_raters = data.mean(axis=0)
    grand_mean = data.mean()

    sst = ((data - grand_mean) ** 2).sum()
    ssr = k * ((mean_subjects - grand_mean) ** 2).sum()
    ssc = n * ((mean_raters - grand_mean) ** 2).sum()
    sse = sst - ssr - ssc

    df_r, df_c, df_e = n - 1, k - 1, (n - 1) * (k - 1)
    if df_r <= 0 or df_e <= 0:
        return None
    msr = ssr / df_r
    msc = ssc / df_c if df_c > 0 else 0
    mse = sse / df_e

    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom == 0:
        return None
    return (msr - mse) / denom


def amostra_icc_bonett(rho, k, w, alpha=0.05):
    """Bonett (2002): tamanho amostral p/ estimar ICC com precisão-alvo
    (largura do IC = 2w), com k medidas por sujeito."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    n = 1 + (2 * k * (1 - rho) * (1 + (k - 1) * rho) ** 2 * z ** 2) / ((k - 1) * w ** 2)
    return n


def amostra_diferenca_minima(sigma, delta, alpha=0.05, power=0.8, pareado=False):
    """Tamanho amostral p/ detectar uma diferença mínima `delta`, dado um
    desvio-padrão `sigma` (teste t, aproximação normal)."""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    if pareado:
        n = ((z_alpha + z_beta) * sigma / delta) ** 2
    else:
        n = 2 * ((z_alpha + z_beta) * sigma / delta) ** 2
    return n


def amostra_precisao_media(sigma, margem, alpha=0.05):
    """Tamanho amostral p/ estimar a média com uma margem de erro alvo
    (intervalo de confiança), dado o desvio-padrão observado."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    n = (z * sigma / margem) ** 2
    return n


def detectar_trials_validos(trials, limiar_fracao=0.5):
    """Marca como 'suspeitos' (aquecimento/erro) trials cujo pico seja
    muito menor que a mediana dos demais — usado só para sugerir a
    seleção inicial, sem excluir nada de forma permanente."""
    if len(trials) < 3:
        return [t["trial"] for t in trials]
    picos = [t["pico"] for t in trials]
    mediana = float(np.median(picos))
    if mediana == 0:
        return [t["trial"] for t in trials]
    return [t["trial"] for t in trials if t["pico"] >= limiar_fracao * mediana]


def recortar_trial_sem_sobreposicao(trial, trials_incluidos_ordenados, tempo_completo, sinal_completo):
    """Recorta o segmento de um trial usando como limites apenas os
    trials vizinhos que estão INCLUÍDOS na análise (ignora trials
    excluídos, como um aquecimento) — assim um trial ganha mais espaço
    quando o vizinho mais próximo foi desmarcado, sem nunca mostrar o
    ciclo de outro trial incluído."""
    idx_atual = trial["idx_pico"]
    posicao = next(i for i, t in enumerate(trials_incluidos_ordenados) if t["trial"] == trial["trial"])

    if posicao > 0:
        idx_anterior = trials_incluidos_ordenados[posicao - 1]["idx_pico"]
        ini = int((idx_anterior + idx_atual) / 2)
    else:
        ini = 0

    if posicao < len(trials_incluidos_ordenados) - 1:
        idx_seguinte = trials_incluidos_ordenados[posicao + 1]["idx_pico"]
        fim = int((idx_atual + idx_seguinte) / 2)
    else:
        fim = len(sinal_completo) - 1

    segmento = sinal_completo[ini:fim + 1]
    tempo_segmento = tempo_completo[ini:fim + 1]
    return tempo_segmento - tempo_completo[idx_atual], segmento


def gerar_figura_resumo(titulo, subtitulo, nomes_trials, kinem_valores, celular_valores,
                         angulo_icone, n_amostra, diferenca_media, angulo_calibracao=None):
    """Monta a figura de resumo (ícone do cotovelo + gráfico de barras +
    cards de estatísticas) pronta para apresentação, e devolve um buffer
    PNG em memória."""
    AZUL, VERMELHO = "#2563EB", "#DC2626"
    CINZA_ESCURO, CINZA_MEDIO, CINZA_CLARO = "#1E293B", "#64748B", "#F1F5F9"
    VERDE, FUNDO = "#16A34A", "#FFFFFF"

    fig = plt.figure(figsize=(13, 7.5), dpi=200, facecolor=FUNDO)
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.05, 1.6, 1.6],
        height_ratios=[0.55, 2.0, 0.62],
        hspace=0.55, wspace=0.28,
        left=0.045, right=0.975, top=0.93, bottom=0.07,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.8, titulo, fontsize=19, fontweight="bold", color=CINZA_ESCURO,
                   ha="left", va="center", transform=ax_title.transAxes)
    ax_title.text(0.0, 0.25, subtitulo, fontsize=12, color=CINZA_MEDIO,
                   ha="left", va="center", transform=ax_title.transAxes)

    ax_icone = fig.add_subplot(gs[1, 0])
    ax_icone.set_xlim(0, 10)
    ax_icone.set_ylim(-1.5, 10)
    ax_icone.axis("off")
    ax_icone.set_aspect("equal")

    ombro_orig = np.array([2.3, 8.3])
    cotovelo_orig = np.array([2.9, 3.6])
    comprimento_antebraco = 5.6
    vetor_braco = ombro_orig - cotovelo_orig
    vetor_braco_unit = vetor_braco / np.linalg.norm(vetor_braco)
    # Ângulo medido diretamente entre a linha do braço (preta) e a do
    # antebraço (azul): giramos o próprio vetor do braço por esse ângulo.
    theta = np.radians(angulo_icone)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vetor_antebraco = rot @ vetor_braco_unit
    punho_orig = cotovelo_orig + comprimento_antebraco * vetor_antebraco

    # Espelha o desenho inteiro (braço para o outro lado), mantendo a geometria correta
    def espelhar(p):
        return np.array([10 - p[0], p[1]])

    ombro = espelhar(ombro_orig)
    cotovelo = espelhar(cotovelo_orig)
    punho = espelhar(punho_orig)
    vetor_braco_unit = (ombro - cotovelo) / np.linalg.norm(ombro - cotovelo)
    vetor_antebraco = (punho - cotovelo) / np.linalg.norm(punho - cotovelo)

    for p1, p2, cor in [(ombro, cotovelo, CINZA_ESCURO), (cotovelo, punho, AZUL)]:
        ax_icone.plot([p1[0], p2[0]], [p1[1], p2[1]], color=cor, lw=9, solid_capstyle="round", zorder=3)
        ax_icone.plot([p1[0], p2[0]], [p1[1], p2[1]], color="white", lw=2.4, solid_capstyle="round", zorder=4, alpha=0.25)
    for p, r, cor in [(ombro, 0.42, CINZA_ESCURO), (cotovelo, 0.5, VERDE), (punho, 0.36, AZUL)]:
        ax_icone.add_patch(plt.Circle(p, r, color=cor, zorder=5, ec="white", lw=2.2))

    raio_arco = 1.7
    ang_braco_deg = np.degrees(np.arctan2(vetor_braco_unit[1], vetor_braco_unit[0]))
    ang_antebraco_deg = np.degrees(np.arctan2(vetor_antebraco[1], vetor_antebraco[0]))
    arco = Arc(cotovelo, raio_arco * 2, raio_arco * 2,
               theta1=min(ang_antebraco_deg, ang_braco_deg), theta2=max(ang_antebraco_deg, ang_braco_deg),
               color=VERDE, lw=3, zorder=2)
    ax_icone.add_patch(arco)

    ponto_meio_ang = np.radians((ang_braco_deg + ang_antebraco_deg) / 2)
    pos_label = cotovelo + (raio_arco + 0.75) * np.array([np.cos(ponto_meio_ang), np.sin(ponto_meio_ang)])
    ax_icone.text(pos_label[0], pos_label[1], f"{angulo_icone}°", fontsize=15, fontweight="bold",
                  color=VERDE, ha="center", va="center", zorder=6)

    ax_icone.text(ombro[0] + 0.35, ombro[1] + 0.55, "Ombro", fontsize=10.5, color=CINZA_MEDIO, ha="center")
    ax_icone.text(cotovelo[0] - 1.15, cotovelo[1] - 0.15, "Cotovelo", fontsize=10.5, color=CINZA_MEDIO, ha="right")
    ax_icone.text(punho[0] - 0.25, punho[1] - 0.45, "Punho", fontsize=10.5, color=CINZA_MEDIO, ha="right")
    ax_icone.text(8.5, -1.35, "Ângulo de flexão do cotovelo", fontsize=11, fontweight="bold",
                  color=CINZA_ESCURO, ha="right", va="bottom")

    ax_bar = fig.add_subplot(gs[1, 1:])
    x = np.arange(len(nomes_trials))
    largura = 0.34
    barras_k = ax_bar.bar(x - largura/2, kinem_valores, largura, label="Kinem (referência)",
                           color=AZUL, edgecolor="white", linewidth=0.6, zorder=3)
    barras_c = ax_bar.bar(x + largura/2, celular_valores, largura, label="Smartphone (calibrado)",
                           color=VERMELHO, edgecolor="white", linewidth=0.6, zorder=3)
    for barras in (barras_k, barras_c):
        for b in barras:
            altura = b.get_height()
            ax_bar.text(b.get_x() + b.get_width()/2, altura + 0.9, f"{altura:.1f}°",
                        ha="center", va="bottom", fontsize=9.5, color=CINZA_ESCURO)

    valor_max = max(list(kinem_valores) + list(celular_valores)) if kinem_valores or celular_valores else 100
    ax_bar.set_ylim(0, valor_max * 1.18)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(nomes_trials, fontsize=10.5, color=CINZA_ESCURO)
    ax_bar.tick_params(axis="x", pad=8)
    ax_bar.set_ylabel("Ângulo de pico (°)", fontsize=11.5, color=CINZA_ESCURO)
    ax_bar.set_title("Concordância entre instrumentos, por repetição", fontsize=13.5,
                      fontweight="bold", color=CINZA_ESCURO, pad=12, loc="left")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.spines[["left", "bottom"]].set_color(CINZA_MEDIO)
    ax_bar.tick_params(colors=CINZA_MEDIO)
    ax_bar.yaxis.grid(True, color=CINZA_CLARO, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.legend(loc="upper left", frameon=False, fontsize=10.5, ncol=2, bbox_to_anchor=(0.0, 1.02))

    stats = [
        (f"{n_amostra}", "sujeitos pareados\n(α=0,05 · poder=80%)", CINZA_ESCURO),
        (f"{diferenca_media:.1f}°", "diferença média\nKinem × smartphone", VERDE),
    ]
    if angulo_calibracao is not None:
        stats.append((f"{angulo_calibracao:.0f}°", "pose de referência\nda calibração funcional", AZUL))
    n_cols = len(stats)
    for i, (valor, legenda, cor) in enumerate(stats):
        ax = fig.add_subplot(gs[2, i] if n_cols == 3 else gs[2, i*3//n_cols:(i+1)*3//n_cols])
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.03, 0.08), 0.94, 0.86, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=CINZA_CLARO, edgecolor="none", transform=ax.transAxes, zorder=1,
        ))
        ax.text(0.5, 0.62, valor, fontsize=26, fontweight="bold", color=cor,
                ha="center", va="center", transform=ax.transAxes, zorder=2)
        ax.text(0.5, 0.24, legenda, fontsize=10, color=CINZA_MEDIO,
                ha="center", va="center", transform=ax.transAxes, zorder=2, linespacing=1.4)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, facecolor=FUNDO, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


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
        "Braço e o do Punho, corrigida pela calibração funcional "
        "abaixo (se configurada)."
    )

    pc1, pc2 = st.columns(2)
    prominence = pc1.number_input("Sensibilidade do pico (prominence, °)", value=15.0, step=1.0, min_value=1.0)
    distance_s = pc2.number_input("Distância mínima entre trials (s)", value=3.0, step=0.5, min_value=0.5)

    trials_kinem = detectar_trials(tempo_flexao_kinem, flexao_kinem, prominence, distance_s) if flexao_kinem is not None else []
    trials_celular = detectar_trials(tempo_flexao_celular, flexao_celular, prominence, distance_s) if flexao_celular is not None else []

    # --- Calibração funcional (pose de referência com ângulo conhecido, ambos os dispositivos) ---
    offset_calibracao = 0.0
    offset_calibracao_kinem = 0.0
    if trials_celular or trials_kinem:
        st.subheader("🎯 Calibração funcional")
        st.caption(
            "Um trial em que o ângulo real era conhecido (pose de "
            "referência) permite corrigir a escala absoluta dos dois "
            "dispositivos — sem isso, os valores só são confiáveis em "
            "termos relativos (consistência entre trials), não em "
            "graus absolutos."
        )
        cal1, cal2 = st.columns(2)
        opcoes_cal = [f"Trial {t['trial']}" for t in (trials_celular or trials_kinem)]
        indice_padrao_cal = 1 if len(opcoes_cal) > 1 else 0  # Trial 2 por padrão
        trial_calibracao = cal1.selectbox("Trial de calibração", opcoes_cal, index=indice_padrao_cal)
        angulo_conhecido = cal2.number_input("Ângulo real conhecido nesse trial (°)", value=90.0, step=1.0)

        num_trial_cal = int(trial_calibracao.replace("Trial ", ""))

        if trials_celular:
            pico_bruto_cal = next((t["pico"] for t in trials_celular if t["trial"] == num_trial_cal), None)
            if pico_bruto_cal is not None:
                offset_calibracao = angulo_conhecido - pico_bruto_cal
                st.success(
                    f"Celular: offset = {angulo_conhecido:.1f}° − {pico_bruto_cal:.1f}° (bruto) "
                    f"= **{offset_calibracao:+.1f}°**"
                )

        if trials_kinem:
            pico_bruto_cal_k = next((t["pico"] for t in trials_kinem if t["trial"] == num_trial_cal), None)
            if pico_bruto_cal_k is not None:
                offset_calibracao_kinem = angulo_conhecido - pico_bruto_cal_k
                st.success(
                    f"Kinem: offset = {angulo_conhecido:.1f}° − {pico_bruto_cal_k:.1f}° (bruto) "
                    f"= **{offset_calibracao_kinem:+.1f}°**"
                )

    flexao_celular_bruta = flexao_celular
    flexao_celular = flexao_celular + offset_calibracao if flexao_celular is not None else None
    for t in trials_celular:
        t["pico_bruto"] = t["pico"]
        t["pico"] = t["pico"] + offset_calibracao
        t["sinal"] = t["sinal"] + offset_calibracao
        t["sinal_ext"] = t["sinal_ext"] + offset_calibracao

    flexao_kinem_bruta = flexao_kinem
    flexao_kinem = flexao_kinem + offset_calibracao_kinem if flexao_kinem is not None else None
    for t in trials_kinem:
        t["pico_bruto"] = t["pico"]
        t["pico"] = t["pico"] + offset_calibracao_kinem
        t["sinal"] = t["sinal"] + offset_calibracao_kinem
        t["sinal_ext"] = t["sinal_ext"] + offset_calibracao_kinem

    # --- 1) Visualização: cortar/alinhar o início do registro ---
    st.subheader("Visualização")
    vc1, vc2, vc3 = st.columns(3)
    cortar_seg = vc1.number_input("Cortar primeiros X segundos (só na visualização)", value=0.0, step=1.0, min_value=0.0)
    alinhar_zero = vc2.checkbox("Alinhar o tempo em zero após o corte", value=True)
    alinhar_base = vc3.checkbox("Alinhar linha de base (Y) — Kinem e Celular saindo do 0", value=True)

    def tempo_visual(tempo):
        tempo = np.asarray(tempo)
        return tempo - cortar_seg if alinhar_zero else tempo

    fig_continuo = go.Figure()
    if flexao_kinem is not None:
        mask_v = tempo_flexao_kinem >= cortar_seg
        valores_k = flexao_kinem[mask_v]
        base_k = valores_k.min() if (alinhar_base and len(valores_k) > 0) else 0.0
        picos_ajuste_k = [t["pico"] - base_k for t in trials_kinem if t["tempo_pico"] >= cortar_seg]
        fig_continuo.add_trace(go.Scatter(
            x=tempo_visual(tempo_flexao_kinem[mask_v]), y=valores_k - base_k, mode="lines",
            name="Kinem", line=dict(color="#1f77b4"),
        ))
        picos_visiveis_k = [t for t in trials_kinem if t["tempo_pico"] >= cortar_seg]
        if picos_visiveis_k:
            fig_continuo.add_trace(go.Scatter(
                x=tempo_visual(np.array([t["tempo_pico"] for t in picos_visiveis_k])),
                y=picos_ajuste_k,
                mode="markers+text",
                text=[f"T{t['trial']}" for t in picos_visiveis_k],
                textposition="top center",
                marker=dict(color="#1f77b4", size=9, symbol="diamond"),
                name="Trials (Kinem)",
            ))
    if flexao_celular is not None:
        mask_vc = tempo_flexao_celular >= cortar_seg
        valores_c = flexao_celular[mask_vc]
        base_c = valores_c.min() if (alinhar_base and len(valores_c) > 0) else 0.0
        picos_ajuste_c = [t["pico"] - base_c for t in trials_celular if t["tempo_pico"] >= cortar_seg]
        fig_continuo.add_trace(go.Scatter(
            x=tempo_visual(tempo_flexao_celular[mask_vc]), y=valores_c - base_c, mode="lines",
            name="Celular (estimado)", line=dict(color="#d62728"),
        ))
        picos_visiveis_c = [t for t in trials_celular if t["tempo_pico"] >= cortar_seg]
        if picos_visiveis_c:
            fig_continuo.add_trace(go.Scatter(
                x=tempo_visual(np.array([t["tempo_pico"] for t in picos_visiveis_c])),
                y=picos_ajuste_c,
                mode="markers+text",
                text=[f"T{t['trial']}" for t in picos_visiveis_c],
                textposition="bottom center",
                marker=dict(color="#d62728", size=9, symbol="diamond"),
                name="Trials (Celular)",
            ))
    fig_continuo.update_layout(
        xaxis_title="Tempo (s)",
        yaxis_title="Ângulo (°)" + (" — relativo à linha de base" if alinhar_base else ""),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_continuo, use_container_width=True)

    # --- Seleção de trials incluídos + referência (usada nas próximas seções) ---
    col_k, col_c = st.columns(2)
    with col_k:
        st.markdown("**Kinem — trials incluídos**")
        if trials_kinem:
            todos_k = [t["trial"] for t in trials_kinem]
            default_k = detectar_trials_validos(trials_kinem)
            if len(default_k) < len(todos_k):
                excluidos = sorted(set(todos_k) - set(default_k))
                st.caption(f"⚠️ Trial(s) {excluidos} parecem aquecimento/outlier (pico bem menor que os demais) — já vêm desmarcados.")
            incluidos_k = st.multiselect("Trials", options=todos_k, default=default_k, key="incluidos_kinem", label_visibility="collapsed")
            trials_kinem_f = [t for t in trials_kinem if t["trial"] in incluidos_k]
            if trials_kinem_f:
                opcoes_trial = [f"Trial {t['trial']}" for t in trials_kinem_f]
                indice_default = 1 if len(opcoes_trial) > 1 else 0
                ref_idx_k = st.selectbox("Trial de referência", opcoes_trial, index=indice_default, key="ref_kinem")
                idx_k = opcoes_trial.index(ref_idx_k)
                trials_kinem_f = calcular_erros(trials_kinem_f, idx_k)
            else:
                st.info("Nenhum trial incluído.")
        else:
            trials_kinem_f = []
            st.info("Nenhum trial detectado — ajuste a sensibilidade acima.")
    with col_c:
        st.markdown("**Celular — trials incluídos**")
        if trials_celular:
            todos_c = [t["trial"] for t in trials_celular]
            default_c = detectar_trials_validos(trials_celular)
            if len(default_c) < len(todos_c):
                excluidos_c = sorted(set(todos_c) - set(default_c))
                st.caption(f"⚠️ Trial(s) {excluidos_c} parecem aquecimento/outlier (pico bem menor que os demais) — já vêm desmarcados.")
            incluidos_c = st.multiselect("Trials", options=todos_c, default=default_c, key="incluidos_celular", label_visibility="collapsed")
            trials_celular_f = [t for t in trials_celular if t["trial"] in incluidos_c]
            if trials_celular_f:
                opcoes_trial_c = [f"Trial {t['trial']}" for t in trials_celular_f]
                indice_default_c = 1 if len(opcoes_trial_c) > 1 else 0
                ref_idx_c = st.selectbox("Trial de referência", opcoes_trial_c, index=indice_default_c, key="ref_celular")
                idx_c = opcoes_trial_c.index(ref_idx_c)
                trials_celular_f = calcular_erros(trials_celular_f, idx_c)
            else:
                st.info("Nenhum trial incluído.")
        else:
            trials_celular_f = []
            st.info("Nenhum trial detectado — ajuste a sensibilidade acima.")

    # --- 2) Trials sobrepostos, em relação ao trial de referência ---
    if trials_kinem_f or trials_celular_f:
        st.subheader("Trials sobrepostos (relativo ao trial de referência)")
        st.caption(
            "Cada linha é o registro completo do ângulo durante aquele "
            "trial (sem repetir o ciclo do vizinho), com o tempo "
            "centralizado no instante do pico (t=0 = pico de flexão). "
            "A linha do trial de referência aparece mais grossa. Um "
            "trial ganha mais espaço automaticamente quando o vizinho "
            "mais próximo está desmarcado (como o Trial 1)."
        )

        jc1, jc2, jc3 = st.columns(3)
        janela_antes = jc1.number_input("Mostrar no máximo até (s antes do pico)", value=15.0, min_value=0.5, step=0.5)
        janela_depois = jc2.number_input("no máximo até (s depois do pico)", value=15.0, min_value=0.5, step=0.5)
        alinhar_zero_y = jc3.checkbox("Alinhar todos no zero (Y)", value=True)

        def preparar_curva_ext(t):
            """Para trials excluídos (ex: Trial 1): usa a janela ampla fixa."""
            tempo_rel = t["tempo_rel_ext"]
            sinal = t["sinal_ext"]
            mask_janela = (tempo_rel >= -janela_antes) & (tempo_rel <= janela_depois)
            tempo_w = tempo_rel[mask_janela]
            sinal_w = sinal[mask_janela]
            if alinhar_zero_y and len(sinal_w) > 0:
                sinal_w = sinal_w - sinal_w.min()
            return tempo_w, sinal_w

        def preparar_curva_incluido(t, incluidos_ordenados, tempo_completo, sinal_completo):
            """Para trials incluídos: recorta sem sobrepor o vizinho
            incluído mais próximo, depois aplica a janela máxima."""
            tempo_rel, sinal = recortar_trial_sem_sobreposicao(t, incluidos_ordenados, tempo_completo, sinal_completo)
            mask_janela = (tempo_rel >= -janela_antes) & (tempo_rel <= janela_depois)
            tempo_w = tempo_rel[mask_janela]
            sinal_w = sinal[mask_janela]
            if alinhar_zero_y and len(sinal_w) > 0:
                sinal_w = sinal_w - sinal_w.min()
            return tempo_w, sinal_w

        col_ck, col_cc = st.columns(2)
        with col_ck:
            if trials_kinem_f:
                fig_k = go.Figure()
                incluidos_ord_k = sorted(trials_kinem_f, key=lambda t: t["tempo_pico"])
                trials_numeros_k = {t["trial"] for t in trials_kinem_f}
                trial1_k = next((t for t in trials_kinem if t["trial"] == 1), None)
                if trial1_k is not None and 1 not in trials_numeros_k:
                    tempo_w, sinal_w = preparar_curva_ext(trial1_k)
                    fig_k.add_trace(go.Scatter(
                        x=tempo_w, y=sinal_w, mode="lines",
                        name="Trial 1 (excluído da análise)",
                        line=dict(width=2, dash="dash", color="gray"),
                    ))
                for t in trials_kinem_f:
                    eh_ref = f"Trial {t['trial']}" == ref_idx_k
                    tempo_w, sinal_w = preparar_curva_incluido(t, incluidos_ord_k, tempo_flexao_kinem, flexao_kinem)
                    fig_k.add_trace(go.Scatter(
                        x=tempo_w, y=sinal_w,
                        mode="lines", name=f"Trial {t['trial']}" + (" (ref.)" if eh_ref else ""),
                        line=dict(width=4 if eh_ref else 2),
                    ))
                fig_k.update_layout(
                    title="Kinem", xaxis_title="Tempo relativo ao pico (s)",
                    yaxis_title="Ângulo (°)" + (" — relativo à linha de base" if alinhar_zero_y else ""),
                    height=400,
                )
                st.plotly_chart(fig_k, use_container_width=True)
        with col_cc:
            if trials_celular_f:
                fig_c = go.Figure()
                incluidos_ord_c = sorted(trials_celular_f, key=lambda t: t["tempo_pico"])
                trials_numeros_c = {t["trial"] for t in trials_celular_f}
                trial1_c = next((t for t in trials_celular if t["trial"] == 1), None)
                if trial1_c is not None and 1 not in trials_numeros_c:
                    tempo_w, sinal_w = preparar_curva_ext(trial1_c)
                    fig_c.add_trace(go.Scatter(
                        x=tempo_w, y=sinal_w, mode="lines",
                        name="Trial 1 (excluído da análise)",
                        line=dict(width=2, dash="dash", color="gray"),
                    ))
                for t in trials_celular_f:
                    eh_ref = f"Trial {t['trial']}" == ref_idx_c
                    tempo_w, sinal_w = preparar_curva_incluido(t, incluidos_ord_c, tempo_flexao_celular, flexao_celular)
                    fig_c.add_trace(go.Scatter(
                        x=tempo_w, y=sinal_w,
                        mode="lines", name=f"Trial {t['trial']}" + (" (ref.)" if eh_ref else ""),
                        line=dict(width=4 if eh_ref else 2),
                    ))
                fig_c.update_layout(
                    title="Celular (estimado)", xaxis_title="Tempo relativo ao pico (s)",
                    yaxis_title="Ângulo (°)" + (" — relativo à linha de base" if alinhar_zero_y else ""),
                    height=400,
                )
                st.plotly_chart(fig_c, use_container_width=True)

    # --- 3) Tabela com média e desvio-padrão ---
    st.subheader("Tabela — trials, ADM e erro (com média e desvio-padrão)")

    def tabela_com_resumo(trials_f):
        df = pd.DataFrame(trials_f)[["trial", "tempo_pico", "pico", "adm", "erro_abs", "erro_rel_pct"]]
        df.columns = ["Trial", "t pico (s)", "Pico (°)", "ADM (°)", "Erro abs (°)", "Erro rel (%)"]
        resumo = pd.DataFrame({
            "Trial": ["Média", "Desvio padrão (SD)"],
            "t pico (s)": [df["t pico (s)"].mean(), df["t pico (s)"].std()],
            "Pico (°)": [df["Pico (°)"].mean(), df["Pico (°)"].std()],
            "ADM (°)": [df["ADM (°)"].mean(), df["ADM (°)"].std()],
            "Erro abs (°)": [df["Erro abs (°)"].mean(), df["Erro abs (°)"].std()],
            "Erro rel (%)": [df["Erro rel (%)"].mean(), df["Erro rel (%)"].std()],
        })
        return pd.concat([df, resumo], ignore_index=True)

    col_tk, col_tc = st.columns(2)
    with col_tk:
        st.markdown("**Kinem**")
        if trials_kinem_f:
            st.dataframe(tabela_com_resumo(trials_kinem_f).round(2), use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum trial incluído.")
    with col_tc:
        st.markdown("**Celular (estimado)**")
        if trials_celular_f:
            st.dataframe(tabela_com_resumo(trials_celular_f).round(2), use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum trial incluído.")

    if trials_kinem_f or trials_celular_f:
        fig_trials = go.Figure()
        if trials_kinem_f:
            fig_trials.add_trace(go.Bar(
                x=[f"Trial {t['trial']}" for t in trials_kinem_f],
                y=[t["pico"] for t in trials_kinem_f],
                name="Kinem",
                marker_color="#1f77b4",
                text=[f"{t['pico']:.1f}°" for t in trials_kinem_f],
                textposition="outside",
            ))
        if trials_celular_f:
            fig_trials.add_trace(go.Bar(
                x=[f"Trial {t['trial']}" for t in trials_celular_f],
                y=[t["pico"] for t in trials_celular_f],
                name="Celular (estimado)",
                marker_color="#d62728",
                text=[f"{t['pico']:.1f}°" for t in trials_celular_f],
                textposition="outside",
            ))
        fig_trials.update_layout(
            title="Ângulo de pico por trial",
            xaxis_title="Trial", yaxis_title="Ângulo de pico (°)",
            barmode="group",
            height=380,
        )
        st.plotly_chart(fig_trials, use_container_width=True)

    # --- 4) Cálculo amostral ---
    st.subheader("📐 Cálculo amostral")
    st.caption(
        "Usando a variabilidade observada nestes dados (desvio-padrão "
        "do ângulo de pico entre trials) como ponto de partida — todos "
        "os valores abaixo são editáveis."
    )

    picos_kinem_ref = [t["pico"] for t in trials_kinem_f] if trials_kinem_f else []
    sd_kinem = float(np.std(picos_kinem_ref, ddof=1)) if len(picos_kinem_ref) > 1 else 5.0

    from scipy.stats import norm
    z_alpha_ref = norm.ppf(1 - 0.05 / 2)
    z_beta_ref = norm.ppf(0.80)
    n_alvo_exemplo = 23
    delta_exemplo = (z_alpha_ref + z_beta_ref) * sd_kinem / (n_alvo_exemplo ** 0.5)

    st.info(
        f"**Exemplo de justificativa para o projeto** (medidas pareadas, "
        f"mesma pessoa nas duas condições):\n\n"
        f"Usando o desvio-padrão do ângulo de pico observado nestes dados "
        f"(SD = {sd_kinem:.2f}°), com nível de significância α = 0,05 "
        f"(bicaudal, z = {z_alpha_ref:.2f}) e poder estatístico de 80% "
        f"(z = {z_beta_ref:.2f}), a fórmula para amostra pareada é:\n\n"
        f"n = [(z_α/2 + z_β) × SD / δ]²\n\n"
        f"Isolando δ (diferença mínima detectável) para uma amostra de "
        f"**N = {n_alvo_exemplo} sujeitos pareados**:\n\n"
        f"δ = (z_α/2 + z_β) × SD / √N = ({z_alpha_ref:.2f} + {z_beta_ref:.2f}) × "
        f"{sd_kinem:.2f}° / √{n_alvo_exemplo} ≈ **{delta_exemplo:.2f}°**\n\n"
        f"Ou seja: com {n_alvo_exemplo} sujeitos avaliados nas duas condições "
        f"(medidas pareadas), o estudo teria poder de 80% para detectar uma "
        f"diferença mínima de aproximadamente {delta_exemplo:.2f}° no ângulo "
        f"de flexão do cotovelo, ao nível de significância de 5%."
    )

    tab_diff, tab_precisao = st.tabs([
        "Diferença mínima (poder)", "Precisão da média (IC)"
    ])

    with tab_diff:
        st.write(
            "Quantas pessoas são necessárias para detectar uma "
            "**diferença mínima** entre duas condições, dado o "
            "desvio-padrão observado entre trials."
        )
        dc1, dc2, dc3, dc4 = st.columns(4)
        sigma_input = dc1.number_input("Desvio-padrão (°)", value=round(sd_kinem, 1), min_value=0.1, step=0.5, key="sigma_diff")
        delta_input = dc2.number_input("Diferença mínima a detectar (°)", value=round(delta_exemplo, 2), min_value=0.1, step=0.5)
        alpha_diff = dc3.number_input("Alfa", value=0.05, min_value=0.01, max_value=0.20, step=0.01, key="alpha_diff")
        power_diff = dc4.number_input("Poder (1-β)", value=0.80, min_value=0.5, max_value=0.99, step=0.05)
        tipo_teste = st.selectbox("Tipo de comparação", ["Medidas pareadas (mesma pessoa)", "Grupos independentes"])
        n_diff = amostra_diferenca_minima(sigma_input, delta_input, alpha_diff, power_diff, pareado=(tipo_teste.startswith("Medidas")))
        if tipo_teste.startswith("Medidas"):
            st.success(f"**N ≈ {int(np.ceil(n_diff))} pessoas** (medidas repetidas)")
        else:
            st.success(f"**N ≈ {int(np.ceil(n_diff))} pessoas por grupo** (≈ {int(np.ceil(n_diff))*2} no total)")

    with tab_precisao:
        st.write(
            "Quantas pessoas são necessárias para estimar a **média** "
            "com uma margem de erro alvo (intervalo de confiança), "
            "dado o desvio-padrão observado."
        )
        pc1b, pc2b, pc3b = st.columns(3)
        sigma_input2 = pc1b.number_input("Desvio-padrão (°)", value=round(sd_kinem, 1), min_value=0.1, step=0.5, key="sigma_precisao")
        margem_input = pc2b.number_input("Margem de erro alvo (°)", value=3.0, min_value=0.1, step=0.5)
        alpha_precisao = pc3b.number_input("Alfa", value=0.05, min_value=0.01, max_value=0.20, step=0.01, key="alpha_precisao")
        n_precisao = amostra_precisao_media(sigma_input2, margem_input, alpha_precisao)
        st.success(f"**N ≈ {int(np.ceil(n_precisao))} pessoas**")

    # --- 5) Figura resumo para apresentação ---
    st.header("🖼️ Figura resumo para apresentação")
    st.caption(
        "Gera uma figura pronta (PNG em alta resolução) com um ícone do "
        "ângulo de flexão do cotovelo, o gráfico de concordância entre "
        "os trials incluídos e os números-chave do estudo."
    )

    fc1, fc2 = st.columns(2)
    titulo_figura = fc1.text_input("Título", value="Validação de Smartphone para Mensuração da Flexão do Cotovelo")
    subtitulo_figura = fc2.text_input("Subtítulo", value="Comparação com sistema de captura de movimento (Kinem®) após calibração funcional")
    angulo_icone = st.slider("Ângulo mostrado no ícone (ilustrativo)", min_value=10, max_value=150, value=35, step=5)

    if st.button("🎨 Gerar figura"):
        nums_comuns_fig = sorted(set(t["trial"] for t in trials_kinem_f) & set(t["trial"] for t in trials_celular_f))
        if len(nums_comuns_fig) == 0:
            st.warning("Não há trials em comum entre Kinem e Celular (ambos incluídos) para montar o gráfico.")
        else:
            picos_k_map_fig = {t["trial"]: t["pico"] for t in trials_kinem_f}
            picos_c_map_fig = {t["trial"]: t["pico"] for t in trials_celular_f}
            nomes_trials_fig = [f"Trial {n}" + (" (ref.)" if f"Trial {n}" == ref_idx_k else "") for n in nums_comuns_fig]
            kinem_fig = [picos_k_map_fig[n] for n in nums_comuns_fig]
            celular_fig = [picos_c_map_fig[n] for n in nums_comuns_fig]
            diferenca_media_fig = float(np.mean([abs(k - c) for k, c in zip(kinem_fig, celular_fig)]))

            buf = gerar_figura_resumo(
                titulo=titulo_figura,
                subtitulo=subtitulo_figura,
                nomes_trials=nomes_trials_fig,
                kinem_valores=kinem_fig,
                celular_valores=celular_fig,
                angulo_icone=angulo_icone,
                n_amostra=n_alvo_exemplo,
                diferenca_media=diferenca_media_fig,
                angulo_calibracao=angulo_conhecido if "angulo_conhecido" in dir() else None,
            )
            st.image(buf, use_container_width=True)
            st.download_button(
                "⬇️ Baixar figura (PNG)",
                data=buf,
                file_name="figura_resumo_apresentacao.png",
                mime="image/png",
            )
