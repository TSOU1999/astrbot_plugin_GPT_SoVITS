import base64
import random
import time

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Plain, Record
from astrbot.core.platform import AstrMessageEvent

from .core.config import PluginConfig
from .core.local_data import LocalDataManager
from .core.preprocess import TextPreprocessor
from .core.providers.base import PROVIDER_REGISTRY, BaseTTSProvider
from .core.result import TTSResult
from .core.service import GPTSoVITSService

# 触发提供商注册
from .core.providers import gsv  # noqa: F401
from .core.providers import fish_audio  # noqa: F401


class GPTSoVITSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(config, context)
        self.local_data = LocalDataManager(self.cfg)

        # ---- 提供商实例化 ----
        provider_cls = PROVIDER_REGISTRY.get(self.cfg.provider)
        if provider_cls is None:
            raise ValueError(f"未知的 TTS 提供商: {self.cfg.provider}")
        self.provider: BaseTTSProvider = provider_cls(self.cfg, context)

        self.service = GPTSoVITSService(self.cfg, self.provider, self.local_data)

        # ---- 护栏状态（纯内存，重载清零已知限制） ----
        self._cooldowns: dict[str, float] = {}       # session_id → 上次调用时间戳
        self._daily_counts: dict[str, int] = {}       # date_key → 当日调用次数
        self._daily_date: str = ""                    # 当前日期（用于跨日重置）

        # ---- 工具激活态（装饰器双注册 + 运行时拨开关） ----
        tool_mgr = context.get_llm_tool_manager()
        if self.cfg.provider == "gpt_sovits":
            tool_mgr.activate_llm_tool("gsv_tts")
            tool_mgr.deactivate_llm_tool("send_voice")
        elif self.cfg.provider == "fish_audio":
            tool_mgr.deactivate_llm_tool("gsv_tts")
            tool_mgr.activate_llm_tool("send_voice")

    async def initialize(self):
        if self.cfg.enabled:
            await self.provider.initialize()

    async def terminate(self):
        await self.provider.close()

    # ========== 工具方法 ==========

    @staticmethod
    def _to_record(res: TTSResult) -> Record:
        """TTSResult → Record 组件（原 _to_record 逻辑不变）。"""
        if res.file_path:
            try:
                return Record.fromFileSystem(res.file_path)
            except Exception:
                logger.warning(f"无法读取文件：{res.file_path}, 已忽略")

        if not res.data:
            raise ValueError("无法获取结果数据")

        b64 = base64.urlsafe_b64encode(res.data).decode()
        return Record.fromBase64(b64)

    # ========== 自动转语音 ==========

    @filter.on_decorating_result(priority=14)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """消息入口：按概率自动将文本回复转为语音。"""
        if not self.cfg.enabled:
            return
        cfg = self.cfg.auto

        result = event.get_result()
        if not result:
            return
        chain = result.chain
        if not chain:
            return
        if cfg.only_llm_result and not result.is_llm_result():
            return
        if random.random() > cfg.tts_prob:
            return

        # 收集所有 Plain 文本片段
        plain_texts = []
        for seg in chain:
            if isinstance(seg, Plain):
                plain_texts.append(seg.text)

        # 仅允许只含 Plain 的消息链通过
        if len(plain_texts) != len(chain):
            return

        # 合并所有 Plain 文本
        combined_text = "\n".join(plain_texts)

        # 仅允许一定长度以下的文本通过
        if len(combined_text) > cfg.max_msg_len:
            return

        final_text, extra = await self.provider.prepare(event, combined_text)
        res = await self.service.inference(final_text, extra)
        if not bool(res):
            return
        chain.clear()
        chain.append(self._to_record(res))

    # ========== 指令 ==========

    @filter.command("说", alias={"gsv", "GSV"})
    async def on_command(self, event: AstrMessageEvent):
        """说 <内容>，直接调用 TTS 合成语音。"""
        if not self.cfg.enabled:
            return

        text = event.message_str.partition(" ")[2]
        final_text, extra = await self.provider.prepare(event, text)
        res = await self.service.inference(final_text, extra)

        if not bool(res):
            yield event.plain_result(res.error)
            return

        yield event.chain_result([self._to_record(res)])

    @filter.command("重启GSV", alias={"重启gsv"})
    async def tts_control(self, event: AstrMessageEvent):
        """重启 GPT-SoVITS（仅 GSV 提供商有效）。"""
        if not self.cfg.enabled:
            return

        if self.cfg.provider != "gpt_sovits":
            yield event.plain_result("当前 TTS 提供商不支持此操作")
            return

        yield event.plain_result("重启 TTS 中...(报错信息请忽略，等待一会即可完成重启)")
        await self.service.restart()

    # ========== LLM 工具 ==========

    @filter.llm_tool()
    async def gsv_tts(self, event: AstrMessageEvent, message: str = ""):
        """用语音输出要讲的话
        Args:
            message(string): 要讲的话
        """
        # 防御性检查（配置热改但插件未重载的边界情况）
        if self.cfg.provider != "gpt_sovits":
            return "当前 TTS 提供商未启用此工具"

        try:
            final_text, extra = await self.provider.prepare(event, message)
            res = await self.service.inference(final_text, extra)
            if not bool(res):
                return res.error
            seg = self._to_record(res)
            await event.send(event.chain_result([seg]))
        except Exception as e:
            return str(e)

    @filter.llm_tool()
    async def send_voice(self, event: AstrMessageEvent, text: str = ""):
        """当你觉得此刻适合用语音表达时调用，用你的声音说一句日语。
        适合的时机：打招呼、道晚安、表达强烈情绪、被要求说话时等。
        语音是点缀，正文仍应以文字回复。
        Args:
            text(string): 要说出的话，请使用日语，简短为佳（一两句）。
        """
        # 防御性检查（配置热改但插件未重载的边界情况）
        if self.cfg.provider != "fish_audio":
            return "当前 TTS 提供商未启用此工具"

        tool_cfg = self.cfg.fish_tool
        if not tool_cfg.enabled:
            return "语音工具当前已被管理员关闭"

        # ---- 护栏 1：文本长度 ----
        if tool_cfg.length_limit_enabled and len(text) > tool_cfg.max_text_len:
            if tool_cfg.overflow_policy == "reject":
                return f"文本过长（{len(text)} 字符），上限为 {tool_cfg.max_text_len} 字符，请缩短后再试"
            else:
                text = text[:tool_cfg.max_text_len]
                logger.debug(f"文本已截断至 {tool_cfg.max_text_len} 字符")

        # ---- 护栏 2：冷却间隔 ----
        if tool_cfg.cooldown_seconds > 0:
            session_id = event.unified_msg_origin or "default"
            now = time.time()
            last = self._cooldowns.get(session_id, 0)
            if now - last < tool_cfg.cooldown_seconds:
                remaining = int(tool_cfg.cooldown_seconds - (now - last))
                return f"语音发送冷却中，请 {remaining} 秒后再试"
            self._cooldowns[session_id] = now

        # ---- 护栏 3：每日上限 ----
        if tool_cfg.daily_max_calls > 0:
            today = time.strftime("%Y-%m-%d")
            if self._daily_date != today:
                self._daily_date = today
                self._daily_counts.clear()
            count = self._daily_counts.get(today, 0)
            if count >= tool_cfg.daily_max_calls:
                return f"今日语音发送已达上限（{tool_cfg.daily_max_calls} 次），请明天再试"
            self._daily_counts[today] = count + 1

        # ---- 预处理（网关 + 打标）→ 推理 → 发送 ----
        try:
            final_text, extra = await self.provider.prepare(event, text)
            res = await self.service.inference(final_text, extra)

            if not bool(res):
                return res.error

            seg = self._to_record(res)
            await event.send(event.chain_result([seg]))

            # 可选：附带剥离标签的原文
            if tool_cfg.send_text_too:
                plain_text = TextPreprocessor.strip_tags(final_text)
                await event.send(event.chain_result([Plain(plain_text)]))

        except Exception as e:
            return str(e)
