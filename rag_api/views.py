import logging
import os
import re
import uuid

from django.conf import settings
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import rag_engine as rag
from .models import Document
from .serializers import ChatRequestSerializer, ApiConfigSerializer

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(settings.BASE_DIR, "temp_uploads")
EXTENSOES_PERMITIDAS = {".pdf", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip('. ')
    if not filename:
        filename = "arquivo"
    return filename[:200]


def _get_session_id(request) -> str:
    session_id = request.session.get("rag_session_id")
    if not session_id:
        session_id = rag.create_session()
        request.session["rag_session_id"] = session_id
    return session_id


@api_view(["GET"])
def health_check(request):
    session_id = _get_session_id(request)
    info = rag.get_status(session_id)
    return Response({
        "status": "online",
        "message": "Servidor ativo.",
        "agente_configurado": info["agente_configurado"],
        "mensagens_no_historico": info["mensagens_no_historico"],
        "documento_carregado": info["documento_carregado"],
        "nome_documento": info["nome_documento"],
        "chunks_count": info["chunks_count"],
        "api_configured": info["api_configured"],
        "api_provider": info["api_provider"],
        "api_model": info["api_model"],
    })


@api_view(["POST"])
def upload_documento(request):
    session_id = _get_session_id(request)

    file = request.FILES.get("file")
    if not file:
        return Response(
            {"detail": "Nenhum arquivo enviado."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    extensao = os.path.splitext(file.name or "")[1].lower()
    if extensao not in EXTENSOES_PERMITIDAS:
        return Response(
            {"detail": f"Extensão '{extensao}' não suportada. Envie PDF ou TXT."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if file.size > MAX_FILE_SIZE:
        return Response(
            {"detail": f"Arquivo muito grande. O limite é de 10MB."},
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    safe_filename = sanitize_filename(file.name)
    unique_prefix = uuid.uuid4().hex[:8]
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{unique_prefix}_{safe_filename}")

    try:
        with open(file_path, "wb") as buffer:
            for chunk in file.chunks():
                buffer.write(chunk)
    except OSError as e:
        return Response(
            {"detail": f"Erro ao salvar arquivo: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        chunks_count = rag.index_document(file_path, extensao, session_id)

        Document.objects.create(
            session_id=session_id,
            filename=safe_filename,
            file_path=file_path,
            file_size=file.size,
            extension=extensao,
            chunks_count=chunks_count,
        )

        return Response({
            "status": "success",
            "message": f"Arquivo '{safe_filename}' indexado com sucesso!",
            "chunks_gerados": chunks_count,
        }, status=status.HTTP_201_CREATED)

    except ValueError as e:
        return Response({"detail": str(e)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
    except Exception as e:
        logger.exception("Erro ao processar documento")
        return Response(
            {"detail": f"Erro ao processar documento: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


@api_view(["POST"])
def chat_com_documento(request):
    session_id = _get_session_id(request)

    serializer = ChatRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    try:
        resposta = rag.ask_question(
            serializer.validated_data["question"],
            session_id,
        )
        return Response({"answer": resposta})
    except ValueError as e:
        return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.exception("Erro no agente")
        return Response(
            {"detail": f"Erro no agente: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["DELETE"])
def resetar_agente(request):
    session_id = _get_session_id(request)
    rag.reset(session_id)
    Document.objects.filter(session_id=session_id).delete()
    return Response({"status": "success", "message": "Agente e histórico resetados."})


@api_view(["GET"])
def listar_provedores(request):
    """Retorna a lista de provedores e seus modelos disponíveis."""
    return Response(rag.get_providers_data())


@api_view(["GET", "POST"])
def configurar_api(request):
    """
    GET: Retorna a configuração atual da API para a sessão.
    POST: Configura a chave de API e modelo para a sessão.
    """
    session_id = _get_session_id(request)

    if request.method == "GET":
        config = rag.get_api_config(session_id)
        return Response(config)

    # POST
    serializer = ApiConfigSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    try:
        result = rag.configure_api(
            session_id,
            serializer.validated_data["provider"],
            serializer.validated_data["api_key"],
            serializer.validated_data["model"],
        )
        return Response(result, status=status.HTTP_200_OK)
    except ValueError as e:
        return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.exception("Erro ao configurar API")
        return Response(
            {"detail": f"Erro ao configurar API: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def obter_interface(request):
    return render(request, "index.html")

