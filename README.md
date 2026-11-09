# Assistente de Documentos com RAG

Assistente conversacional para consultar documentos PDF e TXT com Django, LangChain, LangGraph, FAISS e Groq.

## Funcionalidades

- Upload de PDF e TXT de até 10 MB.
- Busca semântica com embeddings multilíngues e FAISS.
- Respostas com contexto recuperado dos documentos.
- Histórico por sessão e opção de reiniciar a conversa.
- Interface HTML com Tailwind CSS, servida pelo próprio Django.

## Organização

```text
assistente-documentos-rag/
├── backend/
│   ├── rag_api/              # API, modelos, migrações e motor RAG
│   └── rag_project/          # Settings, URLs, WSGI e ASGI
├── frontend/
│   └── templates/
│       └── index.html        # Interface web
├── manage.py                 # Entrada dos comandos Django
├── .env.example              # Exemplo de configuração
├── pyproject.toml            # Metadados e dependências
├── requirements.txt          # Instalação com pip e Render
├── uv.lock                   # Dependências resolvidas pelo uv
└── render.yaml               # Configuração de deploy
```

Frontend e backend usam o mesmo servidor. Não é necessário iniciar um processo separado para a interface.

## Instalação

Requer Python 3.10+ e uma chave da Groq.

```bash
git clone https://github.com/lucasmanoeldev/assistente-documentos-rag.git
cd assistente-documentos-rag
uv sync
```

Copie `.env.example` para `.env` na raiz e preencha:

```env
GROQ_API_KEY=sua_chave_aqui
GROQ_MODEL=llama-3.3-70b-versatile
```

Execute os comandos a partir da raiz:

```bash
uv run python manage.py migrate
uv run python manage.py runserver
```

Acesse [http://localhost:8000](http://localhost:8000).

Alternativa com pip:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

No primeiro processamento, o modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` será baixado; é necessário acesso à internet.

## Fluxo RAG

1. O documento é carregado e dividido em trechos de 1500 caracteres, com sobreposição de 300.
2. Os embeddings são indexados no FAISS.
3. O agente busca trechos relevantes e usa a Groq para gerar a resposta.
4. O histórico e os índices são persistidos localmente por sessão.

O motor atual usa `GROQ_API_KEY` e `GROQ_MODEL` do ambiente. A interface ainda contém opções de outros provedores, mas elas não estão implementadas no motor.

## API

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/api/health` | Estado da sessão |
| POST | `/api/upload` | Upload de documento |
| POST | `/api/chat` | Pergunta sobre o documento |
| DELETE | `/api/reset` | Reinicia a sessão |

Consulte `backend/rag_api/urls.py` para as rotas auxiliares de configuração.

## Dados locais

O banco `db.sqlite3`, os índices em `faiss_indexes/` e os arquivos temporários em `temp_uploads/` continuam na raiz e são ignorados pelo Git. A reorganização preserva os caminhos dos dados existentes.

## Deploy no Render

O `render.yaml` instala as dependências, coleta os arquivos estáticos, executa as migrações e inicia:

```bash
gunicorn --chdir backend rag_project.wsgi:application --timeout 120 --workers 2
```

Configure `GROQ_API_KEY` e, opcionalmente, `GROQ_MODEL` no ambiente do serviço. Para persistir índices e SQLite entre deploys, configure armazenamento persistente.

## Verificação local

```bash
uv run python manage.py check
uv run python manage.py test
```

## Imagem

<img width="1905" height="948" alt="image" src="https://github.com/user-attachments/assets/9f337923-c899-4c29-83e0-4f39f822bc32" />

## Licença

Este projeto é para fins educacionais. O repositório ainda não contém uma licença formal.
