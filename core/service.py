from typing import Any

from astrbot.api import logger

from .config import PluginConfig
from .local_data import LocalDataManager
from .providers.base import BaseTTSProvider
from .result import TTSResult


class GPTSoVITSService:
    """TTS 推理服务层（提供商无关，缓存编排不变）。"""

    def __init__(
        self,
        config: PluginConfig,
        provider: BaseTTSProvider,
        local_data: LocalDataManager,
    ):
        self.cfg = config
        self.provider = provider
        self.local_data = local_data

    async def inference(
        self,
        text: str,
        extra_params: dict[str, Any] | None = None,
    ) -> TTSResult:
        """TTS 推理（缓存编排原样保留）。"""
        params = self.provider.default_params()
        if text:
            params["text"] = text

        if extra_params:
            filtered_params = {
                k: v for k, v in extra_params.items() if k in params
            }
            params.update(filtered_params)
            logger.debug(f"已更新已有参数: {filtered_params}")

        cached_audio = self.local_data.get_cached_audio(params)
        if cached_audio:
            cache_path, cached_data = cached_audio
            logger.debug("命中缓存，跳过 TTS 请求")
            return TTSResult(
                ok=True,
                data=cached_data,
                text=str(params.get("text", "")),
                file_path=str(cache_path),
            )

        logger.debug(f"向 TTS 提供商发起请求，参数: {params}")
        result = await self.provider.tts(params)

        if bool(result):
            cache_path = self.local_data.save_audio(result.data, params)
            if cache_path:
                result.file_path = str(cache_path)
        else:
            logger.error(f"TTS 推理失败: {result.error}")

        return result

    async def restart(self):
        result = await self.provider.restart()
        if not result.ok:
            logger.error(f"重启失败: {result.error}")
