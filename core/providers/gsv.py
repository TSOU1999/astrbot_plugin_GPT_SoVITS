from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiohttp import ClientError, ClientSession, ClientTimeout

from astrbot.api import logger

from ..config import PluginConfig
from ..emotion import EmotionJudger
from ..entry import EntryManager
from ..result import TTSResult
from .base import BaseTTSProvider, register_provider

if TYPE_CHECKING:
    from astrbot.core.platform import AstrMessageEvent
    from astrbot.core.star.context import Context


@register_provider("gpt_sovits")
class GSVProvider(BaseTTSProvider):
    """GPT-SoVITS 提供商（原 client.py 逻辑平移 + 情绪判别）。"""

    def __init__(self, cfg: PluginConfig, context: Context):
        super().__init__(cfg, context)

        # ---- HTTP 客户端（原 GSVApiClient） ----
        client_cfg = cfg.client
        self.base_url = client_cfg.base_url.rstrip("/")
        self.gpt_url = f"{self.base_url}/set_gpt_weights"
        self.sovits_url = f"{self.base_url}/set_sovits_weights"
        self.control_url = f"{self.base_url}/control"
        self.tts_url = f"{self.base_url}/tts"
        self.session = ClientSession(timeout=ClientTimeout(total=client_cfg.timeout))

        # ---- 情绪体系（原 main.__init__ 中的实例化） ----
        self.entry_mgr = EntryManager(cfg)
        self.judger = EmotionJudger(cfg)

    # ========== 生命周期 ==========

    async def initialize(self) -> None:
        """加载 GPT / SoVITS 模型权重。"""
        if self.cfg.model.gpt_path:
            result = await self._set_gpt_weights(self.cfg.model.gpt_path)
            if result.ok:
                logger.info(f"GPT 模型已加载: {self.cfg.model.gpt_path}")
            else:
                logger.error(f"GPT 模型加载失败: {result.error}")

        if self.cfg.model.sovits_path:
            result = await self._set_sovits_weights(self.cfg.model.sovits_path)
            if result.ok:
                logger.info(f"SoVITS 模型已加载: {self.cfg.model.sovits_path}")
            else:
                logger.error(f"SoVITS 模型加载失败: {result.error}")

    async def close(self) -> None:
        if self.session:
            await self.session.close()

    async def restart(self) -> TTSResult:
        return await self._request(self.control_url, params={"command": "restart"})

    # ========== Provider 接口 ==========

    def default_params(self) -> dict[str, Any]:
        """返回 GSV 的 19 个 TTS 参数全集（兼白名单）。"""
        return self.cfg.default_params.copy()

    async def prepare(
        self, event: AstrMessageEvent, text: str
    ) -> tuple[str, dict[str, Any] | None]:
        """情绪判别 → entry.to_params()（文本不变）。原 main._get_emotion_params。"""
        entry = None

        if self.cfg.judge.enabled_llm:
            labels = self.entry_mgr.get_names()
            emotion = await self.judger.judge_emotion(event, text=text, labels=labels)
            if emotion:
                entry = self.entry_mgr.get_entry(emotion)

        if entry is None:
            entry = self.entry_mgr.match_entry(text)

        return text, entry.to_params() if entry else None

    async def tts(self, params: dict[str, Any]) -> TTSResult:
        """执行 GSV TTS 请求（GET + query）。"""
        return await self._request(self.tts_url, params=params)

    # ========== 内部 HTTP ==========

    async def _request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> TTSResult:
        request_text = ""
        if params:
            request_text = str(params.get("text", ""))
            params = {
                k: str(v).lower() if isinstance(v, bool) else v
                for k, v in params.items()
            }

        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    return TTSResult(
                        ok=False,
                        error=f"HTTP {resp.status}: {detail}",
                        text=request_text,
                    )

                return TTSResult(
                    ok=True,
                    data=await resp.read(),
                    text=request_text,
                )

        except ClientError as e:
            logger.error(f"[HTTP] 请求失败: {url} | {e}")
            return TTSResult(ok=False, error=str(e), text=request_text)

        except Exception as e:
            logger.exception(f"[HTTP] 未知异常: {url}")
            return TTSResult(ok=False, error=str(e), text=request_text)

    async def _set_gpt_weights(self, path: str) -> TTSResult:
        return await self._request(self.gpt_url, params={"weights_path": path})

    async def _set_sovits_weights(self, path: str) -> TTSResult:
        return await self._request(self.sovits_url, params={"weights_path": path})
