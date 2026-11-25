"""Task 5: ablation study for Homework 4.

Replays the pipeline on public_test from a naive RAG baseline (B0), turning on one
component at a time, and measures each configuration with the provided scoring
functions. Prints a Markdown table for the README report.

    python3 ablation.py
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hw4

DATA = Path("public_test")

# Each row changes one component relative to the row above it.
VARIANTS = [
    ("B0", "naive RAG (Tutorial 7): get_text, recursive split, whitespace-token BM25 top-5",
     dict(table_text=False, parse=False, parse_llm=False, year_boost=False,
          tokens=False, per_company=False, abstain=False, abstain_llm=False)),
    ("V1", "+ table_to_text and section-aware chunking",
     dict(table_text=True, parse=False, parse_llm=False, year_boost=False,
          tokens=False, per_company=False, abstain=False, abstain_llm=False)),
    ("V2", "+ query parsing (rules) and company/doc-type metadata filtering",
     dict(table_text=True, parse=True, parse_llm=False, year_boost=False,
          tokens=False, per_company=False, abstain=False, abstain_llm=False)),
    ("V3", "+ number-aware tokens, idf coverage bonus, per-company retrieval, LLM company extraction",
     dict(table_text=True, parse=True, parse_llm=True, year_boost=False,
          tokens=True, per_company=True, abstain=False, abstain_llm=False)),
    ("V4", "+ keep only chunks of documents that state the named year",
     dict(table_text=True, parse=True, parse_llm=True, year_boost=True,
          tokens=True, per_company=True, abstain=False, abstain_llm=False)),
    ("V5", "+ abstention: unknown-company/year rules and the LLM answerability check",
     dict(table_text=True, parse=True, parse_llm=True, year_boost=True,
          tokens=True, per_company=True, abstain=True, abstain_llm=True)),
]


def run_variant(cfg: dict, corpus: dict, questions: dict, ground_truth: dict) -> dict:
    hw4.CFG.update(cfg)
    hw4._PARSE_CACHE.clear()
    hw4._CORPUS["companies"].clear()
    hw4._CORPUS["orgs"].clear()

    chunks_by_file = {f: hw4.chunk_document(f, html) for f, html in corpus.items()}
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]]

    parsed = {qid: hw4.parse_query(q) for qid, q in questions.items()}
    index = hw4.build_index(all_chunks)
    retrieved = {qid: hw4.retrieve(index, q) for qid, q in questions.items()}

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(lambda qid: hw4.generate(questions[qid], retrieved[qid]),
                                               questions)))

    graded = [qid for qid in questions if qid in ground_truth]
    answerable = [qid for qid in graded if ground_truth[qid]["evidence"]]
    unanswerable = [qid for qid in graded if not ground_truth[qid]["evidence"]]
    return {
        "parse": sum(hw4.parse_score(parsed[qid], ground_truth[qid]) for qid in graded) / len(graded),
        "recall": sum(hw4.retrieval_hit(retrieved[qid], ground_truth[qid]) for qid in answerable)
                  / max(len(answerable), 1),
        "kept": sum(bool(retrieved[qid]) for qid in answerable) / max(len(answerable), 1),
        "rejected": sum(not retrieved[qid] for qid in unanswerable) / max(len(unanswerable), 1),
        "accuracy": sum(hw4.answer_correct(answers[qid], ground_truth[qid]) for qid in graded)
                    / max(len(graded), 1),
        "usage": dict(hw4.USAGE.summary()),
        "gen_usage": dict(hw4.GENERATOR_USAGE.summary()),
    }


def main() -> None:
    corpus = hw4.load_corpus(DATA / "corpus")
    questions = hw4.load_json(DATA / "questions.json")
    ground_truth = hw4.load_json(DATA / "ground_truth.json")

    print("| Variant | Change | Recall@5 | Kept | Rejected | Accuracy |")
    print("|---------|--------|----------|------|----------|----------|")
    rows = []
    for name, change, cfg in VARIANTS:
        metrics = run_variant(cfg, corpus, questions, ground_truth)
        rows.append((name, change, metrics))
        print(f"| {name} | {change} | {metrics['recall']:.1%} | {metrics['kept']:.1%} "
              f"| {metrics['rejected']:.1%} | {metrics['accuracy']:.1%} |")

    Path("ablation_results.json").write_text(json.dumps(
        {name: {"change": change, **metrics} for name, change, metrics in rows}, indent=2))
    print("\nwrote ablation_results.json")


if __name__ == "__main__":
    main()
