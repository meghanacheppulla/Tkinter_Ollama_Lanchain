"""Simple session memory for the Tkinter chatbot."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage


class ChatMemory:
    """Keep the current conversation in memory until the user clears it."""

    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def add_ai_message(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))

    def as_text(self, limit: int = 12) -> str:
        """Return recent turns in a prompt-friendly format."""
        recent = self.messages[-limit:]
        lines = []
        for message in recent:
            speaker = "User" if isinstance(message, HumanMessage) else "AI"
            lines.append(f"{speaker}: {message.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.messages.clear()
