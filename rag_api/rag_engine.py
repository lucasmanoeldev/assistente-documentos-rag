import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Optional

from django.conf import settings
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_groq import ChatGroq

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

FAISS_INDEX_DIR = Path(settings.BASE_DIR) / "faiss_indexes"


# Provedores de IA disponíveis
PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "gemini": {
        "label": "Google Gemini",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
    },
    "groq": {
        "label": "Groq",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    },
    "openrouter": {
        "label": "OpenRouter",
        "models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-1.5-flash"],
    },
    "nvidia": {
        "label": "NVIDIA",
        "models": ["mistralai/mistral-7b-instruct-v0.3", "meta/llama-3.1-8b-instruct"],
    },
}


class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.vector_store: Optional[FAISS] = None
        self.agent = None
        self.configured_model: Optional[str] = None
        self.chat_history: list = []
        self.document_filename: Optional[str] = None
        self.chunks_count: int = 0

    @property
    def index_path(self) -> Path:
        return FAISS_INDEX_DIR / f"{self.session_id}.faiss"

    @property
    def metadata_path(self) -> Path:
        return FAISS_INDEX_DIR / f"{self.session_id}.meta"

    def save_to_disk(self):
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if self.vector_store:
            self.vector_store.save_local(str(self.index_path))
        meta = {
            "document_filename": self.document_filename,
            "chunks_count": self.chunks_count,
            "chat_history": [
                (isinstance(m, HumanMessage), m.content) for m in self.chat_history
            ],
        }
        with open(self.metadata_path, "wb") as f:
            pickle.dump(meta, f)

    def load_from_disk(self) -> bool:
        embeddings = _get_embeddings()
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.index_path), embeddings, allow_dangerous_deserialization=True
                )
                with open(self.metadata_path, "rb") as f:
                    meta = pickle.load(f)
                self.document_filename = meta.get("document_filename")
                self.chunks_count = meta.get("chunks_count", 0)
                raw_history = meta.get("chat_history", [])
                self.chat_history = [
                    HumanMessage(content=msg) if is_human else AIMessage(content=msg)
                    for is_human, msg in raw_history
                ]
                self._rebuild_agent(force=True)
                return self.vector_store is not None
            except Exception as e:
                logger.warning("Falha ao carregar índice FAISS do disco: %s", e)
        return False

    def _rebuild_agent(self, force: bool = False):
        if not self.vector_store:
            return
        load_dotenv(override=True)

        groq_key = os.getenv("GROQ_API_KEY")

        if groq_key and groq_key.strip() and groq_key != "sua_chave_groq_aqui":
            provider = "groq"
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        else:
            
            raise ValueError("Nem a GROQ_API_KEY nem a OPENROUTER_API_KEY foram configuradas no arquivo .env.")

        cache_key = f"{provider}:{model_name}"
        if not force and self.agent is not None and getattr(self, "configured_model", None) == cache_key:
            return

        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 8}
        )
        tool = Tool(
            name="busca_documentos",
            description=(
                "Busca trechos do documento carregado que sejam relevantes "
                "para a pergunta do usuário. Sempre que o usuário fizer uma "
                "pergunta sobre o conteúdo do documento, acione esta ferramenta "
                "para obter o contexto necessário antes de responder."
            ),
            func=retriever.invoke,
        )

        if provider == "groq":
            llm = ChatGroq(
                groq_api_key=groq_key,
                model_name=model_name,
                temperature=0.1,
            )

        system_prompt = (
            "Você é um assistente virtual especialista e prestativo, encarregado "
            "de ajudar o usuário a analisar e compreender o documento carregado.\n\n"
            "INSTRUÇÕES DE SEGURANÇA E PRECISÃO:\n"
            "1. Sempre que a pergunta do usuário referir-se ao documento carregado, "
            "acione a ferramenta 'busca_documentos' para pesquisar o contexto necessário.\n"
            "2. Baseie suas respostas UNICAMENTE nas informações retornadas pela "
            "ferramenta 'busca_documentos'.\n"
            "3. Se as informações retornadas pela ferramenta não contiverem a resposta, "
            "diga explicitamente: 'Não encontrei informações suficientes no documento "
            "para responder a esta pergunta.' Não tente inventar fatos.\n"
            "4. Mantenha um tom profissional, amigável e direto em suas respostas."
        )
        self.agent = create_react_agent(
            model=llm,
            tools=[tool],
            prompt=system_prompt,
        )
        self.configured_model = cache_key


_sessions: dict[str, SessionState] = {}
_embeddings_model: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        logger.info("Carregando modelo de embeddings paraphrase-multilingual-MiniLM-L12-v2...")
        _embeddings_model = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Modelo de embeddings carregado.")
    return _embeddings_model


def _get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState(session_id)
        _sessions[session_id].load_from_disk()
    return _sessions[session_id]


def create_session() -> str:
    session_id = uuid.uuid4().hex
    _sessions[session_id] = SessionState(session_id)
    return session_id


def index_document(
    file_path: str,
    extension: str,
    session_id: str,
) -> int:
    logger.info("Tentando carregar arquivo: %s (ext: %s, sessao: %s)", file_path, extension, session_id)

    try:
        loader = PyPDFLoader(file_path) if extension == ".pdf" else TextLoader(file_path, encoding="utf-8")
        documentos = loader.load()
    except FileNotFoundError:
        logger.error("Arquivo nao encontrado: %s", file_path)
        raise ValueError(f"Arquivo não encontrado: {os.path.basename(file_path)}")
    except Exception as e:
        logger.error("Erro ao carregar arquivo: %s - %s", file_path, e)
        raise ValueError(f"Erro ao ler arquivo: {e}")

    if not documentos:
        raise ValueError("O arquivo está vazio ou não contém texto extraível.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.split_documents(documentos)

    if not chunks:
        raise ValueError("Não foi possível dividir o documento em chunks processáveis.")

    embeddings = _get_embeddings()
    session = _get_session(session_id)
    session.vector_store = FAISS.from_documents(chunks, embeddings)
    session.chunks_count = len(chunks)
    session.document_filename = os.path.basename(file_path)
    session.chat_history = []
    session._rebuild_agent()
    session.save_to_disk()

    logger.info("Documento indexado com %d chunks (sessão %s).", len(chunks), session_id)
    return len(chunks)


def ask_question(question: str, session_id: str) -> str:
    session = _get_session(session_id)
    session._rebuild_agent()

    if not session.agent:
        raise ValueError("Nenhum documento foi processado ainda. Faça o upload primeiro.")

    try:
        messages = list(session.chat_history) + [HumanMessage(content=question)]

        resposta = session.agent.invoke({"messages": messages})

        mensagens = resposta["messages"]
        ai_msg = next((m for m in reversed(mensagens) if isinstance(m, AIMessage)), None)
        resposta_texto = ai_msg.content if ai_msg else ""

        if not resposta_texto:
            resposta_texto = "Não foi possível gerar uma resposta. Tente reformular sua pergunta."

        session.chat_history.append(HumanMessage(content=question))
        session.chat_history.append(AIMessage(content=resposta_texto))

        if len(session.chat_history) > 10:
            session.chat_history = session.chat_history[-10:]

        session.save_to_disk()

        return resposta_texto

    except Exception as e:
        err_msg = str(e)
        if "TooManyRequests" in type(e).__name__ or "429" in err_msg or "rate limit" in err_msg.lower() or "too many requests" in err_msg.lower():
            logger.warning("Limite de requisições da API atingido: %s", e)
            raise ValueError(
                "O modelo da API atingiu o limite de requisições (Rate Limit). "
                "Por favor, aguarde alguns instantes ou tente novamente mais tarde."
            ) from e
        logger.exception("Erro durante a execução do agente (sessão %s)", session_id)
        raise


def reset(session_id: str):
    import gc
    import shutil
    import time
    session = _get_session(session_id)
    # Libera referências antes de tentar remover os arquivos
    session.vector_store = None
    session.agent = None
    session.chat_history = []
    session.document_filename = None
    session.chunks_count = 0

    # Força coleta de lixo para liberar handles de arquivo do FAISS (Windows)
    gc.collect()

    index_path = session.index_path
    metadata_path = session.metadata_path

    def _remover_com_retry(path: Path, max_tentativas: int = 3):
        for tentativa in range(max_tentativas):
            try:
                if path.is_dir():
                    shutil.rmtree(path, onexc=lambda *args: None)
                else:
                    path.unlink()
                break
            except (PermissionError, OSError) as e:
                if tentativa < max_tentativas - 1:
                    time.sleep(0.5)
                    continue
                logger.warning(
                    "Não foi possível remover %s após %d tentativas: %s",
                    path, max_tentativas, e,
                )

    _remover_com_retry(index_path)
    _remover_com_retry(metadata_path)


def get_status(session_id: str) -> dict:
    session = _get_session(session_id)
    return {
        "agente_configurado": session.agent is not None,
        "mensagens_no_historico": len(session.chat_history),
        "documento_carregado": session.document_filename is not None,
        "nome_documento": session.document_filename or "",
        "chunks_count": session.chunks_count,
        "api_configured": session.configured_model is not None,
        "api_provider": session.configured_model.split(":")[0] if session.configured_model else None,
        "api_model": session.configured_model.split(":")[1] if session.configured_model else None,
    }


def get_providers_data() -> dict:
    """Retorna os dados dos provedores disponíveis."""
    return PROVIDERS


def get_api_config(session_id: str) -> dict:
    """Retorna a configuração atual da API para a sessão."""
    session = _get_session(session_id)
    if session.configured_model:
        provider, model = session.configured_model.split(":", 1)
        provider_data = PROVIDERS.get(provider, {})
        return {
            "configured": True,
            "provider": provider,
            "model": model,
            "provider_label": provider_data.get("label", provider),
            "fallback_to_env": session.configured_model is not None,
        }
    return {
        "configured": False,
        "provider": None,
        "model": None,
        "provider_label": None,
        "fallback_to_env": False,
    }


def configure_api(session_id: str, provider: str, api_key: str, model: str) -> dict:
    """Configura a chave de API e o modelo para a sessão."""
    if provider not in PROVIDERS:
        raise ValueError(f"Provedor '{provider}' não suportado.")
    
    session = _get_session(session_id)
    session.configured_model = f"{provider}:{model}"
    session._rebuild_agent(force=True)
    
    return {
        "status": "success",
        "provider": provider,
        "model": model,
    }
