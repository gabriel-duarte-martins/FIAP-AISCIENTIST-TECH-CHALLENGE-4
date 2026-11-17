from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.fasttext import load_facebook_vectors

from tokenizacao import tokenizar

pasta = Path(__file__).resolve().parent
avaliacoes = pd.read_csv(
    pasta / "database" / "avaliacoes_limpas.csv",
    keep_default_na=False,
)

print("Carregando o modelo fastText...", flush=True)
modelo = load_facebook_vectors(str(pasta / "wiki.pt.bin"))


def vetor_do_texto(texto):
    palavras = tokenizar(texto)
    if not palavras:
        raise ValueError("O texto precisa ter pelo menos uma letra ou número.")

    vetor = np.mean([modelo[palavra] for palavra in palavras], axis=0)
    norma = np.linalg.norm(vetor)
    if norma == 0 or not np.isfinite(norma):
        raise ValueError("Não foi possível gerar um vetor válido para o texto.")

    return vetor / norma


total = len(avaliacoes)
vetores = np.empty((total, modelo.vector_size), dtype=np.float32)

print(f"Gerando vetores para {total} avaliações...", flush=True)
for indice, texto in enumerate(avaliacoes["mensagem_limpa"]):
    vetores[indice] = vetor_do_texto(texto)
    processadas = indice + 1
    if processadas % 5000 == 0 or processadas == total:
        print(f"{processadas}/{total} avaliações processadas", flush=True)

caminho_indice = pasta / "database" / "indice_avaliacoes_spacy.npz"

np.savez(
    caminho_indice,
    vetores=vetores,
    review_id=avaliacoes["review_id"].to_numpy(dtype=str),
    order_id=avaliacoes["order_id"].to_numpy(dtype=str),
    mensagem_limpa=avaliacoes["mensagem_limpa"].to_numpy(dtype=str),
    modelo="wiki.pt.bin",
)

print(f"Índice salvo em: {caminho_indice}")

print(f"\nPronto. A busca compara sua pergunta com {total} avaliações.")
while True:
    pergunta = input("\nSua pergunta (Enter vazio para sair): ").strip()
    if not pergunta:
        break

    try:
        vetor_pergunta = vetor_do_texto(pergunta)
    except ValueError as erro:
        print(erro)
        continue

    # O produto escalar de vetores normalizados dá a similaridade de cosseno.
    similaridades = vetores @ vetor_pergunta
    melhores = np.argsort(-similaridades)[:3]

    for posicao, indice in enumerate(melhores, start=1):
        avaliacao = avaliacoes.iloc[indice]
        print(f"\n{posicao}. Similaridade: {similaridades[indice]:.3f}")
        print(avaliacao["mensagem_limpa"])
        print(f"Avaliação: {avaliacao['review_id']} | Pedido: {avaliacao['order_id']}")