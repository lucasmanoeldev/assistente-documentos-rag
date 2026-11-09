from rest_framework import serializers

from .rag_engine import PROVIDERS


class ChatRequestSerializer(serializers.Serializer):
    question = serializers.CharField(min_length=1, error_messages={
        "min_length": "A pergunta deve ter pelo menos 1 caractere.",
        "blank": "A pergunta não pode estar vazia.",
        "required": "O campo 'question' é obrigatório.",
    })


class ChatResponseSerializer(serializers.Serializer):
    answer = serializers.CharField()


class UploadResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    chunks_gerados = serializers.IntegerField()


class HealthResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    agente_configurado = serializers.BooleanField()
    mensagens_no_historico = serializers.IntegerField()


class ResetResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()


class ApiConfigSerializer(serializers.Serializer):
    provider = serializers.ChoiceField(
        choices=list(PROVIDERS.keys()),
        error_messages={
            "invalid_choice": "Provedor inválido. Escolha entre: openai, gemini, groq, openrouter, nvidia.",
            "required": "O campo 'provider' é obrigatório.",
        },
    )
    api_key = serializers.CharField(
        min_length=8,
        error_messages={
            "min_length": "A chave de API deve ter pelo menos 8 caracteres.",
            "required": "O campo 'api_key' é obrigatório.",
            "blank": "A chave de API não pode estar vazia.",
        },
    )
    model = serializers.CharField(
        min_length=1,
        error_messages={
            "min_length": "O modelo não pode estar vazio.",
            "required": "O campo 'model' é obrigatório.",
        },
    )


class ApiConfigResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    provider = serializers.CharField()
    model = serializers.CharField()


class ApiConfigStatusSerializer(serializers.Serializer):
    configured = serializers.BooleanField()
    provider = serializers.CharField()
    model = serializers.CharField()
    provider_label = serializers.CharField()
    fallback_to_env = serializers.BooleanField()
