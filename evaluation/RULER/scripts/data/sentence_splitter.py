import re


def sent_tokenize(text: str):
    text = text.strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [part.strip() for part in parts if part.strip()]
