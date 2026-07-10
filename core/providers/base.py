from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..result import TTSResult

if TYPE_CHECKING:
    from astrbot.core.platform import AstrMessageEvent
    from astrbot.core.star.context import Context

    from ..config import PluginConfig


PROVIDER_REGISTRY: dict[str, type["BaseTTSProvider"]] = {}


def register_provider(name: str):
    """装饰器：将 TTS 提供商类注册到 PROVIDER_REGISTRY。"""

    def deco(cls: type[BaseTTSProvider]) -> type[BaseTTSProvider]:
        PROVIDER_REGISTRY[name] = cls
        return cls

    return deco


class BaseTTSProvider(ABC):
    """TTS 提供商抽象基类。

    子类必须实现：
        - tts(params) → TTSResult
        - default_params() → dict   （参数全集，兼白名单）
    可选覆写：
        - prepare(event, text) → (final_text, extra_params)
        - initialize() / close() / restart()
    """

    def __init__(self, cfg: "PluginConfig", context: "Context"):
        self.cfg = cfg
        self.context = context

    # ---- 抽象方法 ----

    @abstractmethod
    async def tts(self, params: dict[str, Any]) -> TTSResult:
        """执行 TTS 请求，返回结果。"""
        ...

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """返回该提供商的 TTS 参数全集（所有可能出现在请求中的键）。
        同时充当 extra_params 的白名单。
        """
        ...

    # ---- 可选覆写 ----

    async def prepare(
        self, event: "AstrMessageEvent", text: str
    ) -> tuple[str, dict[str, Any] | None]:
        """预处理文本，返回 (最终合成文本, extra_params)。

        GSVProvider 覆写：情绪判别 → entry.to_params()（文本不变）
        FishAudioProvider 覆写：网关翻译 + 打标签（extra_params 返回 None）
        默认：原样通过。
        """
        return text, None

    async def initialize(self) -> None:
        """提供商初始化（如加载模型）。"""
        pass

    async def close(self) -> None:
        """清理资源（如关闭 HTTP 会话）。"""
        pass

    async def restart(self) -> TTSResult:
        """重启提供商服务。默认不支持。"""
        return TTSResult(ok=False, error="该提供商不支持重启")
