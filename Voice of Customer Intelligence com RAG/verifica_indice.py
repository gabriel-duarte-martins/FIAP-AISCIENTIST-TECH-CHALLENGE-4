from pathlib import Path

import numpy as np
import pandas as pd

database = Path(__file__).resolve().parent / "database"
avaliacoes = pd.read_csv(
    database / "avaliacoes_limpas.csv", dtype=str, keep_default_na=False
)

with np.load(database / "indice_avaliacoes_spacy.npz", allow_pickle=False) as indice:
    vetores = indice["vetores"]

    if vetores.shape != (40827, 300) or not np.isfinite(vetores).all():
        raise ValueError(f"Matriz de vetores inválida: {vetores.shape}")

    for coluna in ("review_id", "order_id", "mensagem_limpa"):
        if not np.array_equal(indice[coluna], avaliacoes[coluna].to_numpy()):
            raise ValueError(f"A coluna {coluna} diverge do CSV.")

    print(f"Modelo registrado: {indice['modelo'].item()}")
    print(f"Índice validado: {vetores.shape[0]} vetores de {vetores.shape[1]} dimensões.")
    print("Os IDs e os comentários estão na mesma ordem do CSV.")