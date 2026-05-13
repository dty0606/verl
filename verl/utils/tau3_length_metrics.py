import re
from typing import Any


def count_text_tokens(tokenizer: Any, text: str) -> int:
    text = text or ""
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())


def assistant_response_part_lengths(tokenizer: Any, text: str) -> dict[str, int]:
    text = text or ""
    think_spans = list(re.finditer(r"<think\b[^>]*>.*?(?:</think>|$)", text, flags=re.DOTALL | re.IGNORECASE))
    tool_spans = list(re.finditer(r"<tool_call\b[^>]*>.*?</tool_call>", text, flags=re.DOTALL | re.IGNORECASE))
    consumed = [False] * len(text)
    think_text = []
    tool_text = []
    for match in think_spans:
        think_text.append(match.group(0))
        for idx in range(match.start(), min(match.end(), len(consumed))):
            consumed[idx] = True
    for match in tool_spans:
        if any(consumed[idx] for idx in range(match.start(), min(match.end(), len(consumed)))):
            continue
        tool_text.append(match.group(0))
        for idx in range(match.start(), min(match.end(), len(consumed))):
            consumed[idx] = True
    final_text = "".join(ch for idx, ch in enumerate(text) if not consumed[idx])
    return {
        "thinking": count_text_tokens(tokenizer, "\n".join(think_text)),
        "tool_call": count_text_tokens(tokenizer, "\n".join(tool_text)),
        "final_text": count_text_tokens(tokenizer, final_text),
    }
