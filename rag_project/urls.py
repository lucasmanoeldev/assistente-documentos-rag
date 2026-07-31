from django.urls import path, include
from rag_api.views import obter_interface

urlpatterns = [
    path("", obter_interface, name="index"),
    path("", include("rag_api.urls")),
]
