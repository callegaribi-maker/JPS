# Visualizador de Dados Brutos

App em Streamlit para importar até 5 arquivos (CSV ou TXT), cada um com
estrutura diferente, e visualizar os dados brutos de cada um em abas
separadas.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app abrirá em `http://localhost:8501`.

## Publicar no GitHub

1. Crie um repositório novo no GitHub (pode ser público ou privado).
2. Na pasta deste projeto, rode:

```bash
git init
git add .
git commit -m "Primeiro commit - app de visualização de dados"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/NOME_DO_REPO.git
git push -u origin main
```

## Publicar no Streamlit Community Cloud

1. Acesse https://share.streamlit.io e faça login com sua conta GitHub.
2. Clique em **New app**.
3. Selecione o repositório e a branch (`main`).
4. Em **Main file path**, informe `app.py`.
5. Clique em **Deploy**.

Depois do deploy, o Streamlit Cloud atualiza o app automaticamente a cada
`git push` no repositório.

## Como funciona

- Na barra lateral, envie até 5 arquivos (CSV ou TXT).
- Para cada arquivo é possível escolher o delimitador (vírgula, ponto e
  vírgula, tabulação, pipe, espaço) ou deixar o app detectar
  automaticamente — útil quando os arquivos `.txt` não usam vírgula.
- Cada arquivo enviado aparece em sua própria aba, com:
  - número de linhas, colunas e valores nulos;
  - tabela com os dados brutos;
  - tipos de cada coluna;
  - botão para baixar os dados já convertidos em CSV.
