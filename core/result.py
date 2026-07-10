from dataclasses import dataclass


@dataclass
class TTSResult:
    """TTS 请求结果（提供商无关）。从 GSVRequestResult 更名平移，字段与 __bool__ 语义不变。"""

    ok: bool
    data: bytes | None = None
    error: str = ""
    text: str = ""
    file_path: str = ""

    @property
    def size(self) -> int:
        """音频数据大小（字节）"""
        return len(self.data) if self.data else 0

    @property
    def is_empty(self) -> bool:
        """是否无数据"""
        return self.size == 0

    def __bool__(self) -> bool:
        return self.ok and not self.is_empty
