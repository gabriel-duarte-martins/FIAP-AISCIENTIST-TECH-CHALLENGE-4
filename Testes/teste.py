from pathlib import Path

from gensim.models.fasttext import load_facebook_vectors

# Modelo FastText oficial em português
modelo = load_facebook_vectors(str(Path(__file__).resolve().parents[1] / "Projeto" / "wiki.pt.bin"))

# Vetor de palavra comum
v = modelo['amor']
print(v.shape)  # (300,)

# Vetor de NEOLOGISMO que NUNCA esteve no corpus
v_novo = modelo['instagramável']
print(v_novo.shape)  # (300,) - foi inferido pelos subwords!

# Vetor de gíria
v_giria = modelo['baita']
print(v_giria.shape)  # (300,) - funciona

# Em Word2Vec, isso jogaria KeyError. Em FastText, sempre tem vetor.