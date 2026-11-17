from spacy.lang.pt import Portuguese

tokenizador = Portuguese().tokenizer


def tokenizar(texto):
    return [
        token.text.lower()
        for token in tokenizador(texto)
        if any(caractere.isalnum() for caractere in token.text)
    ]
