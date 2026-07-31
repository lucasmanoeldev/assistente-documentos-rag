from django.contrib import admin

from .models import ChatMessage, Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ["filename", "session_id", "file_size", "chunks_count", "uploaded_at"]
    list_filter = ["extension", "uploaded_at"]
    search_fields = ["filename", "session_id"]
    readonly_fields = ["file_hash", "uploaded_at"]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ["role", "session_id", "timestamp", "content_preview"]
    list_filter = ["role", "timestamp"]
    search_fields = ["session_id", "content"]

    def content_preview(self, obj):
        return obj.content[:80] + "..." if len(obj.content) > 80 else obj.content
    content_preview.short_description = "Conteúdo"
