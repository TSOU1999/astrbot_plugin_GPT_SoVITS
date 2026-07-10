"""Fish Audio TTS 提供商。

支持 PVC（持久化模型）与 IVC（即时内联克隆）双模式。
HTTP 栈：httpx + socks 代理（懒加载）。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from ..config import PluginConfig
from ..preprocess import TextPreprocessor
from ..result import TTSResult
from .base import BaseTTSProvider, register_provider

if TYPE_CHECKING:
    from astrbot.core.platform import AstrMessageEvent
    from astrbot.core.star.context import Context


@register_provider("fish_audio")
class FishAudioProvider(BaseTTSProvider):
    """Fish Audio 提供商。"""

    def __init__(self, cfg: PluginConfig, context: Context):
        super().__init__(cfg, context)
        fish_cfg = cfg.fish_audio

        self.api_key = fish_cfg.api_key
        self.base_url = fish_cfg.base_url.rstrip("/")
        self.tts_url = f"{self.base_url}/v1/tts"
        self.model_url = f"{self.base_url}/model"

        self.tts_model = fish_cfg.tts_model
        self.timeout = fish_cfg.timeout
        self.proxy = fish_cfg.proxy or None
        self.max_concurrency = fish_cfg.max_concurrency
        self.retry_times = fish_cfg.retry_times

        self.preprocessor = TextPreprocessor(cfg)

        # 并发控制
        self._semaphore: asyncio.Semaphore | None = None
        self._client: Any = None  # httpx.AsyncClient（懒加载）

    # ========== 生命周期 ==========

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ========== Provider 接口 ==========

    def default_params(self) -> dict[str, Any]:
        """返回 Fish Audio 的 TTS 参数全集（兼白名单）。

        extra_body 在拼装期解析合并，参与缓存哈希。
        """
        fish_cfg = self.cfg.fish_audio
        params: dict[str, Any] = {
            "text": "",
            "format": fish_cfg.format,
            "normalize": fish_cfg.normalize,
            "chunk_length": fish_cfg.chunk_length,
            "latency": fish_cfg.latency,
            # 哈希专用键（发送前 pop）
            "_provider": "fish_audio",
            "_model": fish_cfg.tts_model,
        }

        # PVC / IVC 模式
        if fish_cfg.clone_mode == "ivc":
            params["_clone_mode"] = "ivc"
            ivc_fingerprint = self._ivc_fingerprint()
            if ivc_fingerprint:
                params["_ivc_fingerprint"] = ivc_fingerprint
        else:
            params["reference_id"] = fish_cfg.reference_id
            params["_clone_mode"] = "pvc"

        # extra_body 拼装期合并
        extra_raw = fish_cfg.extra_body.strip()
        if extra_raw and extra_raw != "{}":
            try:
                extra = json.loads(extra_raw)
                params.update(extra)
            except json.JSONDecodeError:
                logger.warning(f"extra_body JSON 解析失败，已忽略: {extra_raw}")

        return params

    async def prepare(
        self, event: AstrMessageEvent, text: str
    ) -> tuple[str, dict[str, Any] | None]:
        """网关翻译 + 打标签（extra_params 返回 None）。"""
        final_text = await self.preprocessor.process(event, text)
        return final_text, None

    async def tts(self, params: dict[str, Any]) -> TTSResult:
        """执行 Fish Audio TTS 请求（POST + JSON + 模型名在请求头）。"""
        # 分离哈希专用键
        send_params = params.copy()
        send_params.pop("_provider", None)
        send_params.pop("_model", None)
        send_params.pop("_clone_mode", None)
        send_params.pop("_ivc_fingerprint", None)

        request_text = str(send_params.get("text", ""))

        # 构建请求体
        clone_mode = params.get("_clone_mode", "pvc")
        if clone_mode == "ivc":
            body = self._build_ivc_body(send_params)
        else:
            body = self._build_pvc_body(send_params)

        # 带重试的请求
        last_error = ""
        for attempt in range(self.retry_times + 1):
            try:
                return await self._do_tts(body)
            except _NonRetryableError as e:
                return TTSResult(ok=False, error=str(e), text=request_text)
            except _PVCRejectedError:
                # PVC reference_id 被拒 → 立即降级 IVC（不消耗重试次数）
                if (
                    clone_mode == "pvc"
                    and self.cfg.fish_audio.pvc_auto_fallback
                    and self.cfg.fish_audio.ivc_audio_path
                ):
                    logger.warning("PVC 请求被拒，立即降级 IVC")
                    ivc_body = self._build_ivc_body(send_params)
                    try:
                        return await self._do_tts(ivc_body)
                    except Exception as e:
                        return TTSResult(
                            ok=False,
                            error=f"IVC 降级也失败: {e}",
                            text=request_text,
                        )
                else:
                    return TTSResult(
                        ok=False,
                        error="PVC 请求被拒，且未启用自动降级或 IVC 素材未配置",
                        text=request_text,
                    )
            except Exception as e:
                last_error = str(e)
                if attempt < self.retry_times:
                    logger.warning(f"Fish Audio 请求失败，重试 {attempt + 1}/{self.retry_times}: {e}")
                    await asyncio.sleep(1)

        return TTSResult(ok=False, error=last_error, text=request_text)

    # ========== 内部 ==========

    def _build_pvc_body(self, params: dict[str, Any]) -> dict[str, Any]:
        """PVC 模式请求体。"""
        return {
            "text": params.get("text", ""),
            "reference_id": params.get("reference_id", ""),
            "format": params.get("format", "mp3"),
            "normalize": params.get("normalize", True),
            "chunk_length": params.get("chunk_length", 200),
            "latency": params.get("latency", "normal"),
        }

    def _build_ivc_body(self, params: dict[str, Any]) -> dict[str, Any]:
        """IVC 模式请求体（内联参考音频）。"""
        fish_cfg = self.cfg.fish_audio
        audio_path = fish_cfg.ivc_audio_path
        ref_text = fish_cfg.ivc_ref_text

        if not audio_path:
            raise _NonRetryableError("IVC 模式需要配置 ivc_audio_path")

        audio_bytes = Path(audio_path).read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode()

        body: dict[str, Any] = {
            "text": params.get("text", ""),
            "references": [
                {
                    "audio": audio_b64,
                    "text": ref_text,
                }
            ],
            "format": params.get("format", "mp3"),
            "normalize": params.get("normalize", True),
            "chunk_length": params.get("chunk_length", 200),
            "latency": params.get("latency", "normal"),
        }
        return body

    def _ivc_fingerprint(self) -> str:
        """计算 IVC 参考音频的声线指纹（用于缓存哈希区分不同声线）。"""
        fish_cfg = self.cfg.fish_audio
        audio_path = fish_cfg.ivc_audio_path
        if not audio_path:
            return ""
        try:
            p = Path(audio_path)
            if not p.exists():
                return ""
            # 文件字节哈希 + mtime
            raw = p.read_bytes()
            h = hashlib.sha256(raw).hexdigest()[:12]
            mtime = str(int(p.stat().st_mtime))
            return f"{h}_{mtime}"
        except Exception:
            return ""

    async def _do_tts(self, body: dict[str, Any]) -> TTSResult:
        """单次 TTS 请求（含并发控制与 Ratelimit 头观测）。

        Raises:
            _NonRetryableError: 不应重试的业务错误（402 余额不足等）。
            _PVCRejectedError: PVC reference_id 被拒（由上层 tts() 处理降级）。
        """
        client = await self._get_client()

        async with self._semaphore:
            resp = await client.post(
                self.tts_url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "model": self.tts_model,
                },
            )

        # Ratelimit 头观测
        limit_concurrency = resp.headers.get("Ratelimit-Limit-Concurrency", "?")
        remaining = resp.headers.get("Ratelimit-Remaining-Concurrency", "?")
        logger.debug(
            f"Fish Audio Ratelimit: concurrency limit={limit_concurrency}, remaining={remaining}"
        )

        # 402 余额不足
        if resp.status_code == 402:
            raise _NonRetryableError(
                "Fish Audio API 余额不足（API 钱包需单独充值，与网页端 credits 无关）。"
                "请到 https://fish.audio/developers/ 充值。"
            )

        if resp.status_code != 200:
            detail = await resp.atext()

            # PVC reference_id 被拒 → 抛专用异常，由上层 tts() 降级
            if resp.status_code == 400 and "reference_id" in body:
                raise _PVCRejectedError(f"HTTP {resp.status_code}: {detail}")

            raise _NonRetryableError(f"HTTP {resp.status_code}: {detail}")

        data = await resp.aread()
        if not data:
            raise _NonRetryableError("Fish Audio 返回空响应")

        return TTSResult(
            ok=True,
            data=data,
            text=str(body.get("text", "")),
        )

    async def _get_client(self):
        """懒加载 httpx.AsyncClient（含 socks 代理）。"""
        if self._client is not None:
            return self._client

        import httpx  # 懒加载 — GSV 路径不触发

        limits = httpx.Limits(
            max_connections=self.max_concurrency + 2,
            max_keepalive_connections=self.max_concurrency,
        )

        client_kwargs: dict[str, Any] = {
            "limits": limits,
            "timeout": httpx.Timeout(self.timeout),
        }

        if self.proxy:
            client_kwargs["proxy"] = self.proxy
            logger.info(f"Fish Audio 使用代理: {self.proxy}")

        self._client = httpx.AsyncClient(**client_kwargs)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._client


class _NonRetryableError(Exception):
    """不应重试的业务错误（402、400 格式错误等）。"""
    pass


class _PVCRejectedError(Exception):
    """PVC reference_id 被拒，触发上层降级 IVC 逻辑。"""
    pass
