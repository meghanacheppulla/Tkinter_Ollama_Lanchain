"""WELCOME 🤖: a Tkinter desktop chatbot for college rules PDFs.

Answers come from the uploaded PDF first (LangChain + FAISS + Ollama). If the
PDF does not cover the question, the app falls back to Wikipedia and compares
that text with Ollama's own answer.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from chat_memory import ChatMemory
from rag_pipeline import CampusRAG, ChatResult

# ---------------------------------------------------------------- palette ---
# Midnight indigo + amber + rose + mint on a soft lilac canvas.
COLORS = {
    "canvas": "#6B8E15",      # window background
    "header": "#2A1B54",      # deep indigo header
    "header_sub": "#461AB7",  # subtitle on header
    "ink": "#221A3D",         # main text
    "muted": "#5A4A8A",       # secondary text
    "chat_bg": "#348FB6",     # chat area
    "input_bg": "#146A92",    # question box
    "amber": "#2B1D89",       # primary buttons
    "amber_dark": "#120998",
    "lilac": "#561662",       # quiet buttons
    "lilac_dark": "#821F89",
    "user_label": "#C2255C",  # rose
    "user_bg": "#BF2DB8",
    "ai_label": "#0B7A75",    # teal
    "ai_bg": "#6E1359",
    "ok": "#2F855A",
    "busy": "#3E0775",
    "error": "#C53030",
    "info": "#5A4A8A",
}

FONT = "Segoe UI"
WELCOME = (
    "WELCOME TO TKINTER"
)


def friendly_error(exc: Exception) -> str:
    """Turn common failures into a message a student can act on."""
    text = str(exc) or exc.__class__.__name__
    low = text.lower()
    if "wikipedia" in low:
        return text
    if "not found" in low and "model" in low:
        return (
            f"{text}\n\nInstall the missing model with 'ollama pull <model-name>'. "
            "This app needs llama3.2:latest and nomic-embed-text."
        )
    if any(word in low for word in ("connect", "refused", "11434", "timed out")):
        return (
            "Cannot reach Ollama. Open the Ollama app (or run 'ollama serve') "
            "and try again."
        )
    return text


class CampusCompassApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Tkinter Ollama Assistant")
        self.root.geometry("960x720")
        self.root.minsize(760, 560)
        self.root.configure(bg=COLORS["canvas"])
        self.memory = ChatMemory()
        self.rag = CampusRAG()
        self.busy = False
        self._build_ui()
        self._welcome()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=COLORS["canvas"])
        style.configure("Header.TFrame", background=COLORS["header"])
        style.configure(
            "Hint.TLabel", background=COLORS["canvas"], foreground=COLORS["muted"],
            font=(FONT, 9),
        )
        style.configure(
            "Field.TLabel", background=COLORS["canvas"], foreground=COLORS["ink"],
            font=(FONT, 10, "bold"),
        )
        style.configure(
            "Accent.TButton", background=COLORS["amber"], foreground=COLORS["header"],
            font=(FONT, 10, "bold"), padding=(16, 9), borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", "#E9D9B0"), ("active", COLORS["amber_dark"])],
            foreground=[("disabled", "#8C7F5E")],
        )
        style.configure(
            "Quiet.TButton", background=COLORS["lilac"], foreground=COLORS["header"],
            font=(FONT, 10), padding=(14, 9), borderwidth=0,
        )
        style.map(
            "Quiet.TButton",
            background=[("active", COLORS["lilac_dark"])],
        )
        style.configure(
            "Vertical.TScrollbar", background=COLORS["lilac"],
            troughcolor=COLORS["canvas"], borderwidth=0, arrowcolor=COLORS["header"],
        )

        # Header
        header = ttk.Frame(self.root, padding=(16, 12), style="Header.TFrame")
        header.pack(fill=tk.X)
        titles = tk.Frame(header, bg=COLORS["header"])
        titles.pack(side=tk.LEFT)
        tk.Label(
            titles, text="Campus Compass", bg=COLORS["header"], fg="#FFFFFF",
            font=(FONT, 18, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            titles, text="Tkinter Ollama Assistant",
            bg=COLORS["header"], fg=COLORS["header_sub"], font=(FONT, 9),
        ).pack(anchor=tk.W)

        self.upload_button = ttk.Button(
            header, text="Upload PDF", command=self.upload_pdf,
            style="Accent.TButton",
        )
        self.upload_button.pack(side=tk.RIGHT)
        ttk.Button(
            header, text="Clear Chat", command=self.clear_chat, style="Quiet.TButton"
        ).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(
            header, text="Save Chat", command=self.save_chat, style="Quiet.TButton"
        ).pack(side=tk.RIGHT, padx=(0, 8))

        # Status line
        status_bar = tk.Frame(self.root, bg=COLORS["canvas"])
        status_bar.pack(fill=tk.X, padx=16, pady=(10, 0))
        self.status_dot = tk.Label(
            status_bar, text="\u25CF", bg=COLORS["canvas"], fg=COLORS["info"],
            font=(FONT, 11),
        )
        self.status_dot.pack(side=tk.LEFT)
        self.status_text = tk.Label(
            status_bar, text="", bg=COLORS["canvas"], fg=COLORS["ink"],
            font=(FONT, 10), anchor=tk.W,
        )
        self.status_text.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)
        self._set_status("Upload a PDF to begin.", "info")

        # Question box
        bottom = ttk.Frame(self.root, padding=(16, 4, 16, 12), style="App.TFrame")
        bottom.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(bottom, text="Your question", style="Field.TLabel").pack(
            anchor=tk.W, pady=(0, 5)
        )
        row = tk.Frame(bottom, bg=COLORS["canvas"])
        row.pack(fill=tk.X)
        self.entry = tk.Text(
            row, height=4, wrap=tk.WORD, font=(FONT, 11),
            background=COLORS["input_bg"], foreground=COLORS["ink"],
            insertbackground=COLORS["ink"], relief=tk.FLAT, padx=10, pady=8,
            highlightthickness=1, highlightbackground=COLORS["lilac_dark"],
            highlightcolor=COLORS["amber"],
        )
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<Control-Return>", self._on_enter)
        self.send_button = ttk.Button(
            row, text="Send", command=self.send_question, style="Accent.TButton"
        )
        self.send_button.pack(side=tk.LEFT, padx=(10, 0), fill=tk.Y)
        ttk.Label(
            bottom, text="send", style="Hint.TLabel"
        ).pack(anchor=tk.W, pady=(6, 0))

        # Chat area (expands with the window)
        chat_frame = ttk.Frame(self.root, padding=(16, 8), style="App.TFrame")
        chat_frame.pack(fill=tk.BOTH, expand=True)
        self.chat = tk.Text(
            chat_frame, height=8, wrap=tk.WORD, state=tk.DISABLED, font=(FONT, 11),
            background=COLORS["chat_bg"], foreground=COLORS["ink"],
            insertbackground=COLORS["ink"], relief=tk.FLAT, padx=14, pady=14,
            highlightthickness=1, highlightbackground=COLORS["lilac_dark"],
            highlightcolor=COLORS["lilac_dark"], spacing1=2, spacing3=2,
        )
        self.chat.tag_configure(
            "user_name", foreground=COLORS["user_label"], background=COLORS["user_bg"],
            font=(FONT, 10, "bold"), lmargin1=70, lmargin2=70, rmargin=8,
        )
        self.chat.tag_configure(
            "user_body", foreground=COLORS["ink"], background=COLORS["user_bg"],
            lmargin1=70, lmargin2=70, rmargin=8,
        )
        self.chat.tag_configure(
            "ai_name", foreground=COLORS["ai_label"], background=COLORS["ai_bg"],
            font=(FONT, 10, "bold"), lmargin1=8, lmargin2=8, rmargin=70,
        )
        self.chat.tag_configure(
            "ai_body", foreground=COLORS["ink"], background=COLORS["ai_bg"],
            lmargin1=8, lmargin2=8, rmargin=70,
        )
        self.chat.tag_configure(
            "note", foreground=COLORS["muted"], font=(FONT, 10, "italic"),
        )
        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat.yview)
        self.chat.configure(yscrollcommand=scrollbar.set)
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------- helpers --
    def _set_status(self, text: str, kind: str = "info") -> None:
        self.status_dot.configure(fg=COLORS.get(kind, COLORS["info"]))
        self.status_text.configure(text=text)

    def _write(self, chunks: list[tuple[str, str]]) -> None:
        self.chat.configure(state=tk.NORMAL)
        for text, tag in chunks:
            if tag:
                self.chat.insert(tk.END, text, tag)
            else:
                self.chat.insert(tk.END, text)
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _welcome(self) -> None:
        self._write([(WELCOME + "\n\n", "note")])

    def _append(self, speaker: str, text: str) -> None:
        prefix = "user" if speaker == "You" else "ai"
        self._write([
            (f" {speaker}\n", f"{prefix}_name"),
            (f"{text}\n", f"{prefix}_body"),
            ("\n", ""),
        ])

    def _on_enter(self, _event: tk.Event) -> str:
        self.send_question()
        return "break"

    # --------------------------------------------------------------- upload --
    def upload_pdf(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Select college PDF", filetypes=[("PDF files", "*.pdf")]
        )
        if not path:
            return
        self._set_busy(True, "Indexing PDF with Ollama embeddings...")
        threading.Thread(target=self._index_pdf, args=(path,), daemon=True).start()

    def _index_pdf(self, path: str) -> None:
        try:
            chunks = self.rag.load_pdf(path)
        except Exception as exc:
            message = friendly_error(exc)
            self.root.after(0, lambda: self._index_failed(message))
        else:
            name = Path(path).name
            self.root.after(0, lambda: self._index_done(name, chunks))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def _index_done(self, name: str, chunks: int) -> None:
        self._set_status(f"Ready: {name} ({chunks} searchable chunks)", "ok")

    def _index_failed(self, message: str) -> None:
        self._set_status("PDF indexing failed.", "error")
        messagebox.showerror("PDF error", message)

    # ----------------------------------------------------------------- chat --
    def send_question(self) -> None:
        question = self.entry.get("1.0", tk.END).strip()
        if not question or self.busy:
            return
        if self.rag.vector_store is None:
            messagebox.showinfo("Upload required", "upload pdf.")
            return
        self.entry.delete("1.0", tk.END)
        self._append("You", question)
        self.memory.add_user_message(question)
        self._set_busy(True, "Searching the PDF...")
        threading.Thread(target=self._answer, args=(question,), daemon=True).start()

    def _answer(self, question: str) -> None:
        try:
            result: ChatResult = self.rag.ask(question, self.memory)
        except Exception as exc:
            message = friendly_error(exc)
            self.root.after(0, lambda: self._answer_failed(message))
        else:
            text = f"[{result.source}]\n{result.answer}"
            if result.comparison:
                text += f"\n\nWikipedia vs Ollama check:\n{result.comparison}"
            self.root.after(0, lambda: self._show_answer(text))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def _show_answer(self, text: str) -> None:
        self._append("Compass", text)
        self.memory.add_ai_message(text)
        self._set_status("Ready for the next question.", "ok")

    def _answer_failed(self, message: str) -> None:
        self._set_status("Could not answer that question.", "error")
        messagebox.showerror("Answer error", message)

    def clear_chat(self) -> None:
        self.memory.clear()
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)
        self._welcome()
        self._set_status("Chat cleared. The indexed PDF is still available.", "info")

    def save_chat(self) -> None:
        content = self.chat.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("Nothing to save", "The chat is empty.")
            return
        path = filedialog.asksaveasfilename(
            title="Save chat", defaultextension=".txt",
            filetypes=[("Text files", "*.txt")], initialfile="campus_compass_chat.txt",
        )
        if not path:
            return
        try:
            Path(path).write_text(content + "\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save error", str(exc))
            return
        self._set_status(f"Chat saved to {Path(path).name}", "ok")

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.upload_button.configure(state=state)
        self.send_button.configure(state=state)
        self.entry.configure(state=state)
        if status:
            self._set_status(status, "busy")
        if not busy:
            self.entry.focus_set()


def main() -> None:
    root = tk.Tk()
    CampusCompassApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
