from pathlib import Path

import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def limpar_texto(valor):
    # Converte valores ausentes do CSV em uma string vazia.
    if pd.isna(valor):
        return ""

    # Remove espaços nas pontas e troca espaços repetidos,
    # tabulações e quebras de linha por um único espaço.
    texto = " ".join(str(valor).split())

    # Descarta textos vazios ou sem nenhuma letra ou número.
    # Exemplos: uma vírgula solta, "!!!" ou uma sequência só de emojis.
    if not any(caractere.isalnum() for caractere in texto):
        return ""

    # Preserva a grafia, os acentos e a pontuação da mensagem válida.
    return texto


if __name__ == "__main__":
    pasta_projeto = Path(__file__).resolve().parent
    caminho_csv = pasta_projeto / "database" / "olist_order_reviews_dataset.csv"

    data = load_data(caminho_csv)

    # Aplica a limpeza só à mensagem e guarda o resultado em outra coluna.
    data["mensagem_limpa"] = data["review_comment_message"].map(limpar_texto)

    # Mantém as linhas com mensagem válida e seus identificadores.
    tem_mensagem = data["mensagem_limpa"].ne("")
    avaliacoes = data.loc[tem_mensagem].copy()

    print(f"Total de avaliações: {len(data)}")
    print(f"Com mensagem válida: {len(avaliacoes)}")
    print(f"Sem mensagem válida: {(~tem_mensagem).sum()}")

    # Mostra apenas uma prévia das cinco primeiras avaliações filtradas.
    colunas_previa = ["review_id", "order_id", "mensagem_limpa"]
    print()
    print(avaliacoes[colunas_previa].head().to_string(index=False))

    caminho_saida = pasta_projeto / "database" / "avaliacoes_limpas.csv"
    avaliacoes[colunas_previa].to_csv(caminho_saida, index=False, encoding="utf-8")
    print(f"CSV salvo em {caminho_saida}")
