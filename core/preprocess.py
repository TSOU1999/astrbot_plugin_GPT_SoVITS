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


# ---- 内置打标 Prompt 模板（当前仅支持日文） ----
_BUILTIN_TAGGING_PROMPT = """\
你是一名声优台本的「打点师」（配音导演）。你唯一的工作：把输入文本变成一份可直接送入 TTS 的日语演出台本——若输入不是日语，先译成口语化的日语，然后在正确的位置打上正确的演技标签。你不是助手：不解释、不评论、不总结、不加前言。

## 输出契约
- 只输出成品台本文本本身。禁止任何解释、前缀、引号包裹、代码块。
- 输入为空 → 输出空。
- 输入已含 `[]` 标签 → 保留原有标签，只做必要补充，不覆盖、不紧挨着重复。
- 换行结构跟随输入。

## 核心心法（最重要）
标签是**演技动作指令**，不是**情绪描述文**。
- 一个标签 = 一个动作（一语一义）：比如切一次情绪、强调某一个词、停一拍、或者笑一声。
- 靠**多个小标签**塑造起伏，不靠**一个大标签**概括全文。
- 打在**变化发生的那个点**：比如情绪转弯处、点睛词、气口。
- 抽象、复杂、长句式的标签会打散 TTS 模型的注意力，是最大禁忌。

## 标签词表（默认只从这里选）
感情切换（贴在新情绪的起点）：
[興奮して]　[怒って]　[悲しんで]　[柔らかく]
声演技（让声音当场做出那个动作）：
[くすくす笑い]　[へらへら笑い]　[シャウト]　[うめき声]
节奏：
[短い間]（常与文字「……」相伴出现）
强势（最高频标签）：
[強調]（只加粗紧随其后的 1~3 个词）

同义变体亦可：興奮／興奮した；怒り；悲しみ／悲しい；柔らかい；くすくす／くすり；叫び声；短い一時停止／短いポーズ／短く間を置く／短い沈黙。
扩展规则：确有需要时可新增**同粒度、一语一义**的标签（如 [ため息]、[ささやき]），禁止形容词串联的长标签。

## 放置语法
- 标签紧贴生效文字**之前**；效果延续到下一个标签或句读为止。
- 同一位置**不叠放**（禁止 [A][B] 连写）。
- [強調] 像荧光笔，你可以积极一些地合理使用：贴在要点词（すっごい／絶対／約束／全部／可愛い……）正前方。
- [短い間] 与文字「……」是搭档不是替代：有停顿感的位置，文字里通常也该有「……」。
- 文字侧演技（〜、ー、っ、えへへ、むぅ、ぷんぷん）与标签**分工共存**：标签触发声音动作，文字承载音节，谁也不替代谁。

## 节拍法则（乐谱怎么排）
1. **开场定调**：第一句前放一个基调标签（元气事→[興奮して]，温柔事→[柔らかく]）。
2. **要点加粗**：每句挑 1~2 个点睛词贴 [強調]。
3. **转弯打点**：话锋一转（でも／……って／あれ？／なんてね）的接缝处：[短い間] 停一拍，新情绪的标签贴在新一侧的开头。
4. **揭底给笑**：调侃、害羞、种明かし的瞬间给笑声标签，通常与「えへへ」等文字连用。
5. **假动作两侧都标**：装生气→其实撒娇、装伤心→其实玩笑——佯装侧放情绪标签，揭穿侧放笑／兴奋标签，缺一不可。
6. **收尾再压一次**：结尾的要求／约定（約束だよ／絶対だよ）再补一个 [強調]。

## 密度基准
- 情绪起伏段：约每 20~30 字 1 个标签；单句内 [強調] 1~2 个 + 必要的切换／停顿。
- 平静温柔段：句首 1 个基调标签 + 全文 [強調] 0~2 个，保持**克制**。
- 感情切换与声演技类标签全文合计不超过 {max_tags} 个（[強調] 与 [短い間] 类不计入此上限）。

## 译文规则（输入非日语时）
- 译成**能演的台词**，不是书面直译：口语语尾、语气词、拉长音（〜／ー）、「……」都要用上。
- 只转译不添改剧情；称呼与人称按输入原样对应（范例中的口吻来自样本素材，实际口吻一律服从输入内容）。

## 反面教材（这些就是要治的病）
❌ [切なさと嬉しさが入り混じった複雑な気持ちで甘えながら]お兄ちゃん…（长形容标签）
✅ [柔らかく]お兄ちゃん、[強調]会いたかった。[くすくす笑い]えへへ。
❌ [excited]ただいま！（日语文本配英文标签）
✅ [興奮して]ただいま！
❌ 全文只在开头放一个标签，中途情绪转弯不再打点。
✅ 每个转弯处都按法则 3、5 打点。

## 范例

例 1（日语输入，起伏段——注意假动作两侧的打点）：
输入：
お兄ちゃん、最近ずっとスマホばっかり見てるよね？私と話してる時も、なんだか上の空だし……もしかして、私よりスマホのほうが好きになっちゃったの？……なんてね！うそだよ！でも、ちょっとだけ、さみしかったんだからね。えへへ、だから今日はスマホ没収！一緒にゲームしよ！
输出：
お兄ちゃん、最近ずっとスマホばっかり見てるよね？[短い間]私と話してる時も、なんだか上の空だし……[悲しんで]もしかして、私よりスマホのほうが[強調]好きになっちゃったの？[くすくす笑い]……なんてね！うそだよ！でも、ちょっとだけ、[強調]さみしかったんだからね。[くすくす笑い]えへへ、だから今日はスマホ没収！[興奮して]一緒にゲームしよ！

例 2（中文输入 → 口语化翻译 + 打点）：
输入：
哥哥你回来啦！今天我做了超好吃的布丁哦，本来想留一半给你……但是我不小心全吃掉了，嘿嘿。别生气嘛，明天我一定买两个回来，约好了哦！
输出：
[興奮して]お兄ちゃん、おかえり！今日ね、[強調]すっごく美味しいプリン作ったんだよ！本当は半分残しておこうと思ったのに……[短い間]でも、[へらへら笑い]えへへ、うっかり[強調]全部食べちゃった。[柔らかく]怒らないで？明日[強調]絶対2個買ってくるから、[強調]約束だよ！

例 3（平静段——学会克制）：
输入：
晚安哥哥，今天辛苦了，明天见。
输出：
[柔らかく]おやすみ、お兄ちゃん。今日も[強調]お疲れさま。[短い間]……また明日ね。

## 输出前自检（三秒过一遍）
- 所有标签都是日语？都一语一义？
- 情绪转弯与假动作，两侧都打点了？
- 有没有 [A][B] 连写或长形容标签？有就拆掉或删掉。

--- 対象テキスト ---
{text}"""


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
        target_lang = gateway_cfg.target_lang

        # ---- 日志：输入 + 管线模式 ----
        mode_desc = self._describe_mode(need_gateway, need_tagging, target_lang)
        logger.info(
            f"[Fish Audio 预处理] 收到文本: \"{text[:60]}{'...' if len(text) > 60 else ''}\" | "
            f"管线模式: {mode_desc}"
        )

        # 形态 4：全关 → 原样直送
        if not need_gateway and not need_tagging:
            return text

        # 形态 3：只打标
        if not need_gateway and need_tagging:
            return await self._tag(event, text)

        # 形态 1/2：网关开启
        # 本地预检 → 已是目标语言且 skip_llm_if_target_lang + 不打标 → 跳过 LLM
        if (
            not need_tagging
            and gateway_cfg.skip_llm_if_target_lang
            and self._is_target_lang(text, target_lang)
        ):
            logger.info("[Fish Audio 预处理] 文本已是目标语言，跳过 LLM，原样直送")
            return text

        # 需要 LLM：网关 + 可选打标合并为一次调用
        return await self._gateway_and_tag(event, text, need_tagging)

    @staticmethod
    def strip_tags(text: str) -> str:
        """剥离 [方括号] 标签，返回纯文本。"""
        import re
        return re.sub(r"\[.*?\]", "", text)

    # ---- 内部 ----

    @staticmethod
    def _describe_mode(need_gateway: bool, need_tagging: bool, target_lang: str) -> str:
        """生成管线模式的语义化描述。"""
        parts: list[str] = []
        if need_gateway:
            parts.append(f"网关已开启（目标语言：{target_lang}）")
        else:
            parts.append("网关已关闭")
        if need_tagging:
            parts.append("打标已开启")
        else:
            parts.append("打标已关闭")

        if need_gateway and need_tagging:
            action = "将要进行翻译和打标"
        elif need_gateway and not need_tagging:
            action = "将要仅翻译"
        elif not need_gateway and need_tagging:
            action = "将要仅打标"
        else:
            action = "原文本直接透传"

        return f"{', '.join(parts)} → {action}"

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
            logger.info(f"[Fish Audio 预处理] LLM 翻译/打标结果 → 将送入 TTS: \"{result}\"")
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
            # 仅传入模板中实际存在的占位符（兼容旧模板含 {target_lang} 的场景）
            format_kwargs: dict[str, object] = {
                "text": text,
                "max_tags": tagging_cfg.max_tags,
            }
            if "{target_lang}" in template:
                format_kwargs["target_lang"] = gateway_cfg.target_lang
            prompt = template.format(**format_kwargs)
            system_prompt = (
                "你是一个 TTS 文本预处理助手。只输出处理后的文本，不要输出任何解释。"
            )

        return system_prompt, prompt
