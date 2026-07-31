# Assistente de Documentos com RAG

[![GitHub](https://img.shields.io/badge/GitHub-lucasmanoeldev%2Fassistente--documentos--rag-181717?logo=github)](https://github.com/lucasmanoeldev/assistente-documentos-rag)

Assistente conversacional inteligente que utiliza **RAG (Retrieval-Augmented Generation)** para responder perguntas com base no conteúdo de documentos PDF e TXT.

## Funcionalidades

- Upload de documentos PDF e TXT (até 10 MB)
- Indexação local com **ChromaDB** e embeddings `paraphrase-multilingual-MiniLM-L12-v2`
- Chat interativo com histórico de conversa persistente por sessão
- Recuperação semântica de trechos relevantes via FAISS
- Respostas geradas por LLMs da **Groq** ou **OpenRouter**
- Interface web responsiva construída com **Tailwind CSS**

## Stack

| Camada | Tecnologias |
|--------|-------------|
| **Backend** | Django 5.0 + Django REST Framework |
| **LLM / Agentes** | LangChain, LangGraph, Groq API / OpenRouter |
| **Embeddings** | HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`) |
| **Vector Store** | FAISS |

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/api/health` | Status da sessão e documentos |
| `POST` | `/api/upload` | Upload de PDF ou TXT |
| `POST` | `/api/chat` | Envia pergunta e recebe resposta |
| `DELETE` | `/api/reset` | Reseta agente e histórico |

## Pré-requisitos

- Python **3.10+**
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes)

## Configuração

1. Clone o repositório:

   ```bash
   git clone https://github.com/lucasmanoeldev/assistente-documentos-rag.git
   cd assistente-documentos-rag
   ```

2. Crie o ambiente virtual e instale as dependências:

   ```bash
   uv sync
   ```

3. Configure as variáveis de ambiente no arquivo `.env`:

   ```env
   GROQ_API_KEY=sua_chave_aqui
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

4. Execute as migrações:

   ```bash
   uv run manage.py migrate
   ```

## Executando o projeto

```bash
uv run manage.py runserver
```

Acesse a interface web em [http://localhost:8000](http://localhost:8000).

## Como funciona

1. O usuário faz upload de um arquivo PDF ou TXT
2. O documento é dividido em chunks de 1500 caracteres com sobreposição de 300
3. Os chunks são transformados em embeddings e armazenados no FAISS local
4. Quando o usuário faz uma pergunta, o agente RAG usa LangGraph para buscar os trechos mais relevantes
5. A LLM gera uma resposta baseada exclusivamente no conteúdo encontrado

## Provedores suportados

| Provedor | Modelos |
|----------|---------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| **Google Gemini** | gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro |
| **Groq** | llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768 |
| **OpenRouter** | openai/gpt-4o, anthropic/claude-3.5-sonnet, google/gemini-1.5-flash |
| **NVIDIA** | mistralai/mistral-7b-instruct-v0.3, meta/llama-3.1-8b-instruct |

## Estrutura do projeto

```
.
├── app.py                      # Código legado (não utilizado na versão Django)
├── faiss_indexes/              # Índices FAISS gerados por sessão
├── manage.py                   # CLI do Django
├── pyproject.toml              # Dependências e metadados
├── rag_api/                    # Aplicação Django
│   ├── models.py              # Modelos Document e ChatMessage
│   ├── rag_engine.py          # Motor RAG (regras e interface de chat)
│   ├── serializers.py         # Serializadores DRF
│   ├── templates/
│   │   └── index.html         # Interface web (Tailwind)
│   ├── urls.py                 # Rotas da API
│   └── views.py               # Views DRF
└── rag_project/               # Configuração do projeto Django
    ├── settings.py
    └── urls.py
```

## Licença

Este projeto é para fins educacionais. Sinta-se livre para usá-lo e adaptá-lo conforme necessário.