import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Visualizador de Dados", layout="wide")

st.title("📊 Visualizador de Dados Brutos")
st.write(
    "Envie até 5 arquivos (CSV ou TXT) — cada um pode ter uma estrutura "
    "diferente. Os dados brutos de cada arquivo serão exibidos em uma aba."
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

    # Tenta algumas codificações comuns
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            texto = conteudo.decode(encoding)
            break
        except UnicodeDecodeError:
            texto = None
    if texto is None:
        raise ValueError("Não foi possível decodificar o arquivo (encoding).")

    if delimitador is None:
        # engine='python' + sep=None faz o pandas tentar inferir o separador
        df = pd.read_csv(io.StringIO(texto), sep=None, engine="python")
    else:
        df = pd.read_csv(io.StringIO(texto), sep=delimitador, engine="python")

    return df


# --- Upload dos 5 arquivos, cada um com seu próprio seletor de delimitador ---
st.sidebar.header("⚙️ Configuração dos arquivos")

arquivos = []
for i in range(1, 6):
    st.sidebar.markdown(f"**Arquivo {i}**")
    up = st.sidebar.file_uploader(
        f"Selecionar arquivo {i}",
        type=["csv", "txt"],
        key=f"upload_{i}",
    )
    delim_label = st.sidebar.selectbox(
        f"Delimitador do arquivo {i}",
        list(DELIMITADORES.keys()),
        key=f"delim_{i}",
    )
    arquivos.append((up, DELIMITADORES[delim_label]))
    st.sidebar.divider()

# --- Processa os arquivos enviados ---
dataframes = {}
erros = {}

for i, (up, delim) in enumerate(arquivos, start=1):
    if up is not None:
        try:
            df = ler_arquivo(up, delim)
            dataframes[f"Arquivo {i} — {up.name}"] = df
        except Exception as e:
            erros[f"Arquivo {i} — {up.name}"] = str(e)

# --- Exibição ---
if not dataframes and not erros:
    st.info("Envie pelo menos um arquivo na barra lateral para começar.")
else:
    if erros:
        for nome, msg in erros.items():
            st.error(f"Erro ao ler **{nome}**: {msg}")

    if dataframes:
        abas = st.tabs(list(dataframes.keys()))
        for aba, (nome, df) in zip(abas, dataframes.items()):
            with aba:
                col1, col2, col3 = st.columns(3)
                col1.metric("Linhas", df.shape[0])
                col2.metric("Colunas", df.shape[1])
                col3.metric("Valores nulos", int(df.isna().sum().sum()))

                st.subheader("Dados brutos")
                st.dataframe(df, use_container_width=True)

                st.subheader("Tipos de coluna")
                st.dataframe(
                    df.dtypes.astype(str).rename("tipo"),
                    use_container_width=True,
                )

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Baixar como CSV",
                    data=csv_bytes,
                    file_name=f"{nome.split(' — ')[0].replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"download_{nome}",
                )
