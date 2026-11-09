import hashlib
import os

from django.conf import settings
from django.db import models


def document_upload_path(instance, filename):
    return f"documents/{instance.session_id}/{filename}"


class Document(models.Model):
    session_id = models.CharField(max_length=64, db_index=True)
    filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=512)
    file_size = models.IntegerField()
    file_hash = models.CharField(max_length=64, editable=False)
    extension = models.CharField(max_length=10)
    chunks_count = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def save(self, *args, **kwargs):
        if not self.file_hash and self.file_path and os.path.exists(self.file_path):
            sha256 = hashlib.sha256()
            with open(self.file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)
            self.file_hash = sha256.hexdigest()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.filename} ({self.session_id})"


class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ("user", "Usuário"),
        ("assistant", "Assistente"),
        ("system", "Sistema"),
    ]
    session_id = models.CharField(max_length=64, db_index=True)
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}..."
