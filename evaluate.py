import logging
import json
import sys
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from rag.loader import process_pdf
from rag.embedder import create_vectorstore
from rag.retriever import get_retriever
from rag.chain import build_rag_chain, ask_question, get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_questions(chunks: list, num_questions: int = 10) -> list[dict]:
    """
    Use the local LLM to auto-generate question-answer pairs from document chunks.
    This saves you from writing test cases by hand.
    """
    llm = get_llm()
    qa_pairs = []

    # Sample chunks spread across the document
    step = max(1, len(chunks) // num_questions)
    sampled = chunks[::step][:num_questions]

    logger.info(f"Generating {len(sampled)} Q&A pairs from document chunks...")

    for i, chunk in enumerate(sampled):
        prompt = f"""Based on the following text, write one clear factual question and its answer.

Text:
{chunk.page_content}

Respond ONLY in this exact JSON format (no extra text):
{{"question": "...", "answer": "..."}}"""

        try:
            raw = llm.invoke(prompt)
            # Find JSON in response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                qa_pairs.append({
                    "question": data["question"],
                    "ground_truth": data["answer"],
                    "context_chunk": chunk.page_content
                })
                logger.info(f"  Generated Q{i+1}: {data['question'][:60]}...")
        except Exception as e:
            logger.warning(f"  Skipped chunk {i+1} — could not parse LLM response: {e}")

    return qa_pairs


def build_ragas_dataset(qa_pairs: list[dict], chain, retriever) -> Dataset:
    """
    Run each question through the RAG pipeline and collect:
    - question
    - generated answer
    - retrieved contexts
    - ground truth answer
    This is the format RAGAS expects.
    """
    questions, answers, contexts, ground_truths = [], [], [], []

    for pair in qa_pairs:
        q = pair["question"]
        result = ask_question(chain, retriever, q)

        # Get raw context chunks (as strings) for RAGAS
        raw_contexts = retriever.invoke(q)
        context_texts = [doc.page_content for doc in raw_contexts]

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(context_texts)
        ground_truths.append(pair["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })


def run_evaluation(pdf_path: str, num_questions: int = 10) -> dict:
    """
    Full evaluation pipeline:
    1. Process the PDF
    2. Auto-generate test Q&A pairs
    3. Run questions through RAG
    4. Score with RAGAS metrics
    5. Return scores dict
    """
    logger.info(f"\n=== DocMind RAG Evaluation ===")
    logger.info(f"PDF: {pdf_path}")

    # Build the RAG system
    chunks = process_pdf(pdf_path)
    if not chunks:
        logger.error("No chunks extracted. Evaluation aborted.")
        return {}

    vectorstore = create_vectorstore(chunks, persist_dir="vectorstore_eval")
    retriever = get_retriever(vectorstore, k=4)
    chain = build_rag_chain(retriever)

    # Generate test data
    qa_pairs = generate_test_questions(chunks, num_questions)
    if not qa_pairs:
        logger.error("Could not generate any Q&A pairs.")
        return {}

    # Build RAGAS dataset
    logger.info("\nRunning questions through RAG pipeline...")
    dataset = build_ragas_dataset(qa_pairs, chain, retriever)

    # Run RAGAS evaluation
    logger.info("\nScoring with RAGAS metrics...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
    )

    scores = {
        "faithfulness": round(results["faithfulness"], 3),
        "answer_relevancy": round(results["answer_relevancy"], 3),
        "context_recall": round(results["context_recall"], 3),
        "context_precision": round(results["context_precision"], 3),
    }

    return scores


def save_report(scores: dict, pdf_path: str, output_path: str = "eval_report.md"):
    """Save a readable markdown evaluation report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf_name = Path(pdf_path).name

    def grade(score):
        if score >= 0.8:
            return "✅ Good"
        elif score >= 0.6:
            return "⚠️ Acceptable"
        else:
            return "❌ Needs improvement"

    report = f"""# DocMind RAG Evaluation Report

**Date:** {now}
**Document:** {pdf_name}
**Model:** llama3.1 (Ollama)
**Embeddings:** nomic-embed-text

## Results

| Metric | Score | Status |
|---|---|---|
| Faithfulness | {scores.get('faithfulness', 'N/A')} | {grade(scores.get('faithfulness', 0))} |
| Answer Relevancy | {scores.get('answer_relevancy', 'N/A')} | {grade(scores.get('answer_relevancy', 0))} |
| Context Recall | {scores.get('context_recall', 'N/A')} | {grade(scores.get('context_recall', 0))} |
| Context Precision | {scores.get('context_precision', 'N/A')} | {grade(scores.get('context_precision', 0))} |

## What these metrics mean

- **Faithfulness**: Are answers grounded in the retrieved context? (No hallucination)
- **Answer Relevancy**: Is the answer actually addressing the question?
- **Context Recall**: Did the retriever find the chunks needed to answer correctly?
- **Context Precision**: Were the retrieved chunks relevant (no noise)?

## Overall

Average score: {round(sum(scores.values()) / len(scores), 3) if scores else 'N/A'}
"""

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "uploads/sample.pdf"
    num_q = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    scores = run_evaluation(pdf, num_questions=num_q)

    if scores:
        print("\n=== RAGAS Scores ===")
        for metric, score in scores.items():
            print(f"  {metric}: {score}")

        save_report(scores, pdf)
        print("\nReport saved to eval_report.md")