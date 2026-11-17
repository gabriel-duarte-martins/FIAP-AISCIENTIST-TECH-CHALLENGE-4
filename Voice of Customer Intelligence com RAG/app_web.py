import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import ollama

from buscar_avaliações import BuscaAvaliacoes, formatar_contexto, gerar_resposta


class Aplicacao(BaseHTTPRequestHandler):
    def enviar(self, status, tipo, corpo):
        self.send_response(status)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def json(self, status, dados):
        corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
        self.enviar(status, "application/json; charset=utf-8", corpo)

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.enviar(200, "text/html; charset=utf-8", Path(__file__).with_name("index.html").read_bytes())

    def do_POST(self):
        if self.path != "/api/pergunta":
            self.send_error(404)
            return
        try:
            tamanho = int(self.headers.get("Content-Length", "0"))
            if not 0 < tamanho <= 8192:
                raise ValueError("Envie uma pergunta de até 1.000 caracteres.")
            dados = json.loads(self.rfile.read(tamanho))
            pergunta = dados.get("pergunta") if isinstance(dados, dict) else None
            if not isinstance(pergunta, str) or not 0 < len(pergunta.strip()) <= 1000:
                raise ValueError("Digite uma pergunta de até 1.000 caracteres.")
            pergunta = pergunta.strip()
            fontes = self.server.busca.recuperar(pergunta)
        except (ValueError, KeyError) as erro:
            self.json(400, {"erro": str(erro)})
            return

        try:
            resposta = gerar_resposta(pergunta, formatar_contexto(fontes))
        except (ollama.ResponseError, ollama.RequestError, ConnectionError):
            self.json(503, {"erro": "Não foi possível gerar a resposta. Confira se o Ollama está aberto e se o Llama 3.2 está disponível."})
            return

        self.json(200, {"resposta": resposta, "fontes": fontes})


if __name__ == "__main__":
    with HTTPServer(("127.0.0.1", 8000), Aplicacao) as servidor:
        servidor.busca = BuscaAvaliacoes()
        print("Abra http://127.0.0.1:8000 no navegador.", flush=True)
        servidor.serve_forever()
