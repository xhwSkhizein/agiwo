import json
import time
from uuid import uuid4

import httpx


class FeishuApiClient:
    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        api_base_url: str,
        request_timeout_seconds: float = 10.0,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._api_base_url = api_base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=request_timeout_seconds)

        self._tenant_access_token: str | None = None
        self._token_expire_at: float = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def add_message_reaction(self, message_id: str, emoji_type: str) -> None:
        payload = {
            "reaction_type": {
                "emoji_type": emoji_type,
            }
        }
        await self._authorized_request(
            "POST",
            f"/open-apis/im/v1/messages/{message_id}/reactions",
            json_body=payload,
        )

    async def reply_text(self, message_id: str, text: str) -> None:
        payload = {
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
            "reply_in_thread": False,
            "uuid": str(uuid4()),
        }
        await self._authorized_request(
            "POST",
            f"/open-apis/im/v1/messages/{message_id}/reply",
            json_body=payload,
        )

    async def create_text_message(self, chat_id: str, text: str) -> None:
        payload = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
            "uuid": str(uuid4()),
        }
        await self._authorized_request(
            "POST",
            "/open-apis/im/v1/messages",
            params={"receive_id_type": "chat_id"},
            json_body=payload,
        )

    async def get_user_display_name(self, open_id: str) -> str | None:
        payload = await self._authorized_request(
            "GET",
            f"/open-apis/contact/v3/users/{open_id}",
            params={"user_id_type": "open_id"},
        )
        data = payload.get("data")
        if not isinstance(data, dict):
            return None

        user = data.get("user")
        if not isinstance(user, dict):
            return None

        name = user.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

        en_name = user.get("en_name")
        if isinstance(en_name, str) and en_name.strip():
            return en_name.strip()

        nickname = user.get("nickname")
        if isinstance(nickname, str) and nickname.strip():
            return nickname.strip()

        return None

    async def _authorized_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict | None = None,
    ) -> dict:
        token = await self._get_tenant_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        return await self._request(method, path, headers=headers, params=params, json_body=json_body)

    async def _get_tenant_access_token(self) -> str:
        now = time.time()
        if self._tenant_access_token is not None and now < self._token_expire_at:
            return self._tenant_access_token

        response = await self._request(
            "POST",
            "/open-apis/auth/v3/tenant_access_token/internal",
            headers={"Content-Type": "application/json; charset=utf-8"},
            json_body={
                "app_id": self._app_id,
                "app_secret": self._app_secret,
            },
        )

        token = response.get("tenant_access_token")
        if not token:
            raise RuntimeError("feishu_tenant_access_token_missing")

        expire_seconds = int(response.get("expire", 7200))
        # Refresh 2 minutes early to avoid race around expiration.
        self._tenant_access_token = token
        self._token_expire_at = now + max(60, expire_seconds - 120)
        return token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        params: dict[str, str] | None = None,
        json_body: dict | None = None,
    ) -> dict:
        url = f"{self._api_base_url}{path}"
        response = await self._client.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
        )

        payload = response.json()
        code = int(payload.get("code", -1))
        if code != 0:
            msg = payload.get("msg", "unknown_error")
            request_id = payload.get("request_id", "")
            raise RuntimeError(f"feishu_api_error code={code} msg={msg} request_id={request_id}")

        response.raise_for_status()

        return payload
