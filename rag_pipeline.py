"""PDF-first RAG pipeline with Wikipedia fallback and answer comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chat_memory import ChatMemory
from wikipedia_tool import create_wikipedia_tool


@dataclass
class ChatResult:
    answer: str
    source: str
    wikipedia_answer: str = ""
    comparison: str = ""


class CampusRAG:
    """Index one PDF and answer questions against it before Wikipedia."""

    def __init__(
        self,
        model_name: str = "llama3.2:latest",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store: FAISS | None = None
        self.pdf_name = ""
        self.wikipedia = create_wikipedia_tool()

    def load_pdf(self, pdf_path: str) -> int:
        """Load, split, embed, and index a PDF. Return the chunk count."""
        path = Path(pdf_path)
        documents = PyPDFLoader(str(path)).load()
        if not documents:
            raise ValueError("The selected PDF does not contain readable text.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1_000, chunk_overlap=150, add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError("The selected PDF does not contain readable text.")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.pdf_name = path.name
        return len(chunks)

    def _pdf_context(self, question: str) -> tuple[str, float]:
        if self.vector_store is None:
            return "", 0.0
        matches = self.vector_store.similarity_search_with_score(question, k=6)
        if not matches:
            return "", 0.0
        # FAISS returns squared distance: lower means more relevant.
        best_distance = float(matches[0][1])
        context = "\n\n".join(doc.page_content for doc, _ in matches)
        return context, best_distance

    def _answer_from_pdf(self, question: str, context: str, memory: str) -> str:
        prompt = f"""You are a college rules assistant. Answer only from the PDF context below.
If the context does not answer the question, reply exactly: NOT_IN_PDF.
Use the conversation only to resolve references such as 'my name'; never invent college rules.

Conversation:
{memory or '(none)'}

PDF context:
{context}

Question: {question}
Answer:"""
        return str(self.llm.invoke(prompt)).strip()

    @staticmethod
    def _has_question_terms(question: str, context: str) -> bool:
        """Avoid a slow PDF LLM call when the retrieved text is clearly unrelated."""
        stop_words = {
            "what", "when", "where", "which", "who", "why", "how", "is", "are",
            "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "explain",
        }
        terms = {
            term for term in re.findall(r"[a-z0-9]+", question.lower())
            if len(term) > 2 and term not in stop_words
        }
        context_terms = set(re.findall(r"[a-z0-9]+", context.lower()))
        return bool(terms & context_terms)

    @staticmethod
    def _has_exact_rule_reference(question: str, context: str) -> bool:
        """Keep numbered college rules in the PDF path even with a high FAISS distance."""
        references = re.findall(r"\b(?:rule\s*)?(\d+\s*[.]\s*\d+)\b", question, re.I)
        if not references:
            return False
        normalized_context = re.sub(r"(?<=\d)\s*[.]\s*(?=\d)", ".", context.lower())
        for reference in references:
            normalized_reference = re.sub(r"\s+", "", reference)
            if normalized_reference in normalized_context:
                return True
        return False

    @staticmethod
    def _remembered_name(question: str, memory: ChatMemory) -> str | None:
        """Answer a name question from the current conversation, never Wikipedia."""
        if not re.search(r"\bwhat(?:'s| is) my name\b|\bwho am i\b", question, re.I):
            return None
        matches = re.findall(
            r"(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z .'-]{0,40})",
            memory.as_text(),
            re.I,
        )
        if not matches:
            return "I do not know your name yet. Tell me by saying, 'My name is ...'."
        name = matches[-1].strip(" .,!?:;")
        return f"Your name is {name}."

    def _compare_answers(self, question: str, wikipedia_answer: str) -> tuple[str, str]:
        ollama_prompt = f"""Answer this question using only the Wikipedia text below.
Question: {question}
Wikipedia text: {wikipedia_answer}
Give a concise factual answer."""
        ollama_answer = str(self.llm.invoke(ollama_prompt)).strip()
        judge_prompt = f"""Compare these two answers to the same question.
Question: {question}
Answer A (Wikipedia): {wikipedia_answer}
Answer B (Ollama using Wikipedia): {ollama_answer}
Reply in one short sentence beginning with either 'Same' or 'Different', then explain why."""
        comparison = str(self.llm.invoke(judge_prompt)).strip()
        return ollama_answer, comparison

    def ask(self, question: str, memory: ChatMemory) -> ChatResult:
        """Use the PDF when relevant; otherwise retrieve and compare Wikipedia."""
        remembered_answer = self._remembered_name(question, memory)
        if remembered_answer:
            return ChatResult(answer=remembered_answer, source="Conversation memory")

        context, distance = self._pdf_context(question)
        if (
            context
            and (
                self._has_exact_rule_reference(question, context)
                or (distance < 1.25 and self._has_question_terms(question, context))
            )
        ):
            answer = self._answer_from_pdf(question, context, memory.as_text())
            if answer and not re.search(r"NOT_IN_PDF", answer, re.IGNORECASE):
                return ChatResult(answer=answer, source=f"PDF: {self.pdf_name}")

        try:
            wikipedia_answer = str(self.wikipedia.invoke(question)).strip()
        except Exception as exc:
            raise RuntimeError(
                "Wikipedia lookup failed. Check your internet connection and try again."
            ) from exc
        if not wikipedia_answer:
            wikipedia_answer = "Wikipedia did not return an answer."
        ollama_answer, comparison = self._compare_answers(question, wikipedia_answer)
        answer = (
            f"{wikipedia_answer}\n\n"
            f"Ollama's answer from the same Wikipedia text:\n{ollama_answer}"
        )
        return ChatResult(
            answer=answer,
            source="Wikipedia fallback",
            wikipedia_answer=wikipedia_answer,
            comparison=comparison,
        )
