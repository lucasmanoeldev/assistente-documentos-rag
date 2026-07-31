from django.urls import path
from . import views

urlpatterns = [
    path("api/health", views.health_check, name="health"),
    path("api/upload", views.upload_documento, name="upload"),
    path("api/chat", views.chat_com_documento, name="chat"),
    path("api/reset", views.resetar_agente, name="reset"),
    path("api/providers", views.listar_provedores, name="providers"),
    path("api/configure", views.configurar_api, name="configure"),
]
