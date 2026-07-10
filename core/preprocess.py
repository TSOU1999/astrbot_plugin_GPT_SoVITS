"""Fish Audio 预处理：日语网关 + 情绪标签标注。

网关与打标合并为一次 LLM 调用完成。
包含 raw_text → tagged_text 内存 LRU memo（解决打标非确定性废缓存问题）。
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import TYPE_CHECKING

from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.core.platform.astr_message_event import AstrMessageEvent

    from .config import PluginConfig


# ---- 日语字符范围（用于本地启发式预检） ----
_JAPANESE_RANGES = [
    (0x3040, 0x309F),  # 平假名
    (0x30A0, 0x30FF),  # 片假名
    (0x4E00, 0x9FFF),  # CJK 统一汉字
    (0xFF66, 0xFF9F),  # 半角片假名
]


def _looks_japanese(text: str, threshold: float = 0.3) -> bool:
    """简单启发式：非空白字符中日语字符占比超过 threshold 则认为是日语。"""
    cleaned = text.strip()
    if not cleaned:
        return False
    jp_count = 0
    for ch in cleaned:
        if ch.isspace():
            continue
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _JAPANESE_RANGES):
            jp_count += 1
    return jp_count / max(len(cleaned), 1) >= threshold


# ---- 内置打标 Prompt 模板 ----
_BUILTIN_TAGGING_PROMPT = """\
你是一个为 TTS 语音合成添加情绪标注的助手。

任务：
1. 若文本不是{target_lang}，先翻译为{target_lang}。
2. 在文本中插入 2~{max_tags} 个 [方括号] 情绪标签。标签作用于其后的文字，直到下一个标签或句末。
   - 可用标签参考（自由格式，英文）：[happy] [sad] [excited] [whispering] [sigh] [surprised] [angry] [gentle] [long pause] [laughing] [crying]
   - 标签可叠加兼容组合（如 [sad][whispering]），避免矛盾组合
   - 位置即语义，放在要修饰的语句之前

只输出处理后的文本（含标签），不要输出任何解释。
文本：{text}"""


# ---- LRU Memo 实现 ----

class _LRUMemo:
    """简单的 LRU memo，用于 raw_text → tagged_text 缓存。"""

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._data: OrderedDict = OrderedDict()

    def get(self, key) -> str | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def set(self, key, value: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)


_tagging_memo_store = _LRUMemo(maxsize=256)


class TextPreprocessor:
    """Fish Audio 文本预处理管线。"""

    def __init__(self, cfg: "PluginConfig"):
        self.cfg = cfg

    # ---- 对外接口 ----

    async def process(
        self, event: "AstrMessageEvent", text: str
    ) -> str:
        """主入口：网关翻译 + 打标 → 返回可用于 TTS 的最终文本。

        四种组合形态由配置开关控制。
        """
        gateway_cfg = self.cfg.fish_gateway
        tagging_cfg = self.cfg.fish_tagging
        need_gateway = gateway_cfg.enabled
        need_tagging = tagging_cfg.enabled

        # 形态 4：全关 → 原样直送
        if not need_gateway and not need_tagging:
            return text

        # 形态 3：只打标
        if not need_gateway and need_tagging:
            return await self._tag(event, text)

        # 形态 1/2：网关开启
        # 本地预检 → 已是目标语言且 skip_llm_if_target_lang + 不打标 → 跳过 LLM
        target_lang = gateway_cfg.target_lang
        if (
            not need_tagging
            and gateway_cfg.skip_llm_if_target_lang
            and self._is_target_lang(text, target_lang)
        ):
            return text

        # 需要 LLM：网关 + 可选打标合并为一次调用
        return await self._gateway_and_tag(event, text, need_tagging)

    @staticmethod
    def strip_tags(text: str) -> str:
        """剥离 [方括号] 标签，返回纯文本。"""
        import re
        return re.sub(r"\[.*?\]", "", text)

    # ---- 内部 ----

    def _is_target_lang(self, text: str, target_lang: str) -> bool:
        if target_lang == "ja":
            return _looks_japanese(text)
        return False

    async def _tag(self, event: "AstrMessageEvent", text: str) -> str:
        """仅打标（LLM 调用，带 memo 缓存）。"""
        max_tags = self.cfg.fish_tagging.max_tags
        target_lang = self.cfg.fish_gateway.target_lang

        # 查 memo
        cached = _tagging_memo_store.get((text, max_tags, target_lang))
        if cached is not None:
            logger.debug("preprocess memo 命中，跳过打标 LLM 调用")
            return cached

        result = await self._call_llm(event, text, translate=False, tag=True)

        # 写 memo
        _tagging_memo_store.set((text, max_tags, target_lang), result)
        return result

    async def _gateway_and_tag(
        self, event: "AstrMessageEvent", text: str, tag: bool
    ) -> str:
        """网关翻译 + 可选打标，一次 LLM 调用。"""
        return await self._call_llm(event, text, translate=True, tag=tag)

    async def _call_llm(
        self,
        event: "AstrMessageEvent",
        text: str,
        *,
        translate: bool,
        tag: bool,
    ) -> str:
        """执行 LLM 调用。"""
        try:
            provider = self._get_provider(event)
            system_prompt, prompt = self._build_prompt(text, translate, tag)

            resp = await provider.text_chat(
                system_prompt=system_prompt,
                prompt=prompt,
            )
            result = resp.completion_text.strip()
            logger.debug(f"preprocess LLM 输出: {result[:80]}...")
            return result

        except Exception as e:
            logger.exception(f"preprocess LLM 调用失败: {e}")
            if self.cfg.fish_tagging.fail_policy == "raw":
                logger.warning("fail_policy=raw，原文直送合成")
                return text
            raise

    def _get_provider(self, event: "AstrMessageEvent"):
        """获取预处理用的 LLM 提供商。"""
        tagging_cfg = self.cfg.fish_tagging
        provider_id = tagging_cfg.provider_id
        return self.cfg.get_judge_provider(
            event.unified_msg_origin if provider_id else None
        )

    def _build_prompt(
        self, text: str, translate: bool, tag: bool
    ) -> tuple[str, str]:
        """构建 system_prompt 和 prompt。"""
        tagging_cfg = self.cfg.fish_tagging
        gateway_cfg = self.cfg.fish_gateway

        if not tag:
            # 仅翻译不打标
            system_prompt = (
                f"你是一个翻译助手。将文本翻译为{gateway_cfg.target_lang}。"
                "只输出翻译结果，不要输出任何解释。"
            )
            prompt = text
        else:
            # 翻译 + 打标（或仅打标）
            template = tagging_cfg.custom_prompt or _BUILTIN_TAGGING_PROMPT
            prompt = template.format(
                text=text,
                max_tags=tagging_cfg.max_tags,
                target_lang=gateway_cfg.target_lang,
            )
            system_prompt = (
                "你是一个 TTS 文本预处理助手。只输出处理后的文本，不要输出任何解释。"
            )

        return system_prompt, prompt
