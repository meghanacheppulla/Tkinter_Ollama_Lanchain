# Campus Compass (ollama_langchain)

A desktop chatbot (Tkinter) for college rules and regulations. Upload a college PDF and ask questions.

- Answers come from your PDF first (LangChain + FAISS + Ollama embeddings).
- If the PDF does not cover the question, the app looks it up on Wikipedia and compares Wikipedia's text with Ollama's answer.
- Remembers the conversation (for example "My name is ...") until you press **Clear Chat**.
- **Save Chat** exports the conversation to a `.txt` file.

## Requirements

- Python 3.10+ (Tkinter is included with Python on Windows and macOS; on Linux run `sudo apt install python3-tk`)
- [Ollama](https://ollama.com) installed and running

## First-time setup (needs internet once)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

macOS / Linux: activate with `source .venv/bin/activate`.

## Run

```powershell
cd path\to\ollama_langchain
.\.venv\Scripts\Activate.ps1
python app.py
```

If Ollama is not already running in the background, start it first with `ollama serve` in another window.

## Use

1. Click **Upload College PDF** and wait for "Ready".
2. Type a question. **Enter** sends, **Shift+Enter** adds a new line.
3. Look at the label above each answer: `[PDF: ...]`, `[Wikipedia fallback]` or `[Conversation memory]`.

## Offline use

PDF questions work fully offline. Only the Wikipedia fallback needs internet.

## Build a Windows .exe (optional)

```powershell
pip install pyinstaller
pyinstaller CampusCompass.spec
```

## Colours

All colours are in the `COLORS` dictionary at the top of `app.py` (indigo header, amber buttons, rose and mint chat bubbles on a lilac background).
