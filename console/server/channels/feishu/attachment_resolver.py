"""Feishu attachment download and ContentPart conversion."""

from pathlib import Path

from agiwo.agent import ContentPart, ContentType
from agiwo.utils.logging import get_logger

from server.channels.feishu.api_client import FeishuApiClient
from server.channels.models import Attachment, InboundMessage

logger = get_logger(__name__)

_MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

_ATTACHMENT_CONTENT_TYPE_MAP: dict[str, ContentType] = {
    "image": ContentType.IMAGE,
    "audio": ContentType.AUDIO,
    "media": ContentType.VIDEO,
    "file": ContentType.FILE,
}


class FeishuAttachmentResolver:
    def __init__(
        self,
        *,
        api: FeishuApiClient,
        tmp_dir: Path,
    ) -> None:
        self._api = api
        self._tmp_dir = tmp_dir

    async def resolve_attachments(
        self,
        message: InboundMessage,
    ) -> list[ContentPart]:
        parts: list[ContentPart] = []
        for attachment in message.attachments:
            try:
                part = await self._resolve_single_attachment(message, attachment)
                if part is not None:
                    parts.append(part)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "feishu_attachment_resolve_failed",
                    key=attachment.key,
                    attachment_type=attachment.type,
                    error=str(exc),
                )
        return parts

    async def _resolve_single_attachment(
        self,
        message: InboundMessage,
        attachment: Attachment,
    ) -> ContentPart | None:
        if attachment.type == "sticker":
            return None

        data = await self._download_attachment(message, attachment)
        if len(data) > _MAX_DOWNLOAD_SIZE:
            logger.warning(
                "feishu_attachment_too_large",
                key=attachment.key,
                size=len(data),
                limit=_MAX_DOWNLOAD_SIZE,
            )
            return None

        mime = _detect_mime(data)
        ext = _mime_to_ext(mime, attachment.name)
        filename = f"{attachment.key}{ext}"
        local_path = self._tmp_dir / filename
        local_path.write_bytes(data)

        content_type = _ATTACHMENT_CONTENT_TYPE_MAP.get(
            attachment.type,
            ContentType.FILE,
        )
        return ContentPart(
            type=content_type,
            url=str(local_path),
            mime_type=mime,
            metadata={
                "name": attachment.name or filename,
                "size": len(data),
                "source": "feishu",
            },
        )

    async def _download_attachment(
        self,
        message: InboundMessage,
        attachment: Attachment,
    ) -> bytes:
        if attachment.type == "image":
            return await self._api.download_image(attachment.key)
        return await self._api.download_message_resource(
            message.message_id,
            attachment.key,
            "file",
        )


def _detect_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif data[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    elif data[:4] == b"GIF8":
        mime = "image/gif"
    elif data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        mime = "image/webp"
    elif data[:4] == b"%PDF":
        mime = "application/pdf"
    elif len(data) > 8 and data[4:8] == b"ftyp":
        mime = "video/mp4"
    else:
        mime = "application/octet-stream"
    return mime


def _mime_to_ext(mime: str, original_name: str = "") -> str:
    if original_name:
        suffix = Path(original_name).suffix
        if suffix:
            return suffix

    ext_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
        "video/mp4": ".mp4",
        "audio/mp4": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
    }
    return ext_map.get(mime, ".bin")


__all__ = ["FeishuAttachmentResolver"]
