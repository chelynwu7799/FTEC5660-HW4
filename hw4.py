"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Design switches. The submission runs with all of them on; ablation.py (Task 5)
# replays the pipeline from a naive baseline (all off) and turns on one at a time.
CFG = {
    "table_text": True,    # V1: table_to_text + section-aware chunking (off: naive get_text)
    "parse": True,         # V2: parse the query, filter chunks on company and doc_type
    "parse_llm": True,     # V4: LLM extraction of the companies a question names
    "year_boost": True,    # V3: restrict to chunks of documents that state the named year
    "tokens": True,        # V4: number-aware tokens, stopwords off, idf coverage bonus
    "per_company": True,   # V4: retrieve per company for comparison questions
    "abstain": True,       # V5: return [] for companies/years the corpus does not cover
    "abstain_llm": True,   # V5: LLM check for figures that are not disclosed at all
}

CHUNK_TARGET = 1300   # characters per chunk, safely below MAX_CHUNK_CHARS (1500)
YEAR_BOOST = 1.35     # score multiplier for chunks of a fiscal year that states the fact
PER_COMPANY_MIN = 2   # chunks guaranteed per company in a comparison
COVERAGE_BONUS = 1.5  # per unit of idf of the distinct query terms a chunk covers

TOKEN_RE = re.compile(r"[a-z]+|\d+(?:,\d{3})*(?:\.\d+)?%?")

# Question words and boilerplate that should not drive retrieval.
STOPWORDS = set("""
a an the and or of at in on for to from by with as is are was were be been being do does did done
have has had having what which who whom whose when where why how much many more most its it s
usd millions million percent per share shares value end fy
""".split())


def _tokens(text: str) -> list[str]:
    """Lower-cased tokens that keep figures ('2,710.4', '8.5%') as single tokens."""
    return TOKEN_RE.findall(str(text).lower())


def _content_tokens(text: str) -> list[str]:
    """Retrieval tokens: figures and words, without stopwords."""
    return [t for t in _tokens(text) if t not in STOPWORDS]


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def _cell_text(cell: Any) -> str:
    return _clean(cell.get_text(" ", strip=True))


# ------------------------------------------------------------------ Task 1.1

def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    soup = BeautifulSoup(str(table_html), "lxml")
    table = soup.find("table") or soup

    caption_tag = table.find("caption")
    caption = _cell_text(caption_tag) if caption_tag else ""

    # Column headers, one entry per column; several header rows are joined per column
    # and a colspan repeats its header over the columns it spans.
    header_parts: list[list[str]] = []
    thead = table.find("thead")
    if thead:
        for tr in thead.find_all("tr"):
            col = 0
            for cell in tr.find_all(["th", "td"], recursive=False):
                span = int(cell.get("colspan") or 1)
                text = _cell_text(cell)
                for _ in range(span):
                    while len(header_parts) <= col:
                        header_parts.append([])
                    if text and text not in header_parts[col]:
                        header_parts[col].append(text)
                    col += 1
    columns = [" ".join(parts).strip() for parts in header_parts]

    data_rows = [tr for tr in table.find_all("tr") if tr.find_parent("thead") is None]
    if not columns and data_rows:  # no <thead>: a first row of <th> is the header
        first = data_rows[0].find_all(["th", "td"], recursive=False)
        if first and all(c.name == "th" for c in first):
            columns = [_cell_text(c) for c in first]
            data_rows = data_rows[1:]

    lines = [caption] if caption else []
    for tr in data_rows:
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        texts = [_cell_text(c) for c in cells]
        if len(texts) == 1:  # a section row ("Assets", "Operating activities", ...)
            if texts[0]:
                lines.append(texts[0])
            continue
        label, values = texts[0], texts[1:]
        heads = columns[1:1 + len(values)]
        heads += [""] * (len(values) - len(heads))
        parts = [f"{head}: {value}" if head else value for head, value in zip(heads, values)]
        lines.append(f"{label} | " + " | ".join(p for p in parts if p))
    return "\n".join(lines)


# ------------------------------------------------------------------ Task 1.2

_HEADING_LEVEL = {"h1": 0, "h2": 1, "h3": 2, "h4": 3}
_SKIP_TAGS = {"nav", "style", "script", "head", "footer"}
_LEGAL_SUFFIX_RE = re.compile(r"\s+(inc\.?|corp\.?|corporation|ltd\.?|limited|plc|llc)$", re.I)


def _display_company(name: str) -> str:
    """'Northgate Robotics Inc. (NGR)' -> 'Northgate Robotics'."""
    name = re.sub(r"\s*\([^)]*\)\s*$", "", _clean(name))
    previous = None
    while previous != name:
        previous = name
        name = _LEGAL_SUFFIX_RE.sub("", name).strip().rstrip(",").strip()
    return name


def _doc_header_meta(soup: BeautifulSoup, file_name: str) -> tuple[str, int, str]:
    """company / fiscal_year / doc_type, from the first lines of the document."""
    h1 = soup.find("h1")
    company = _display_company(h1.get_text(" ", strip=True) if h1 else file_name)
    subtitle = soup.select_one("p.subtitle")
    subtitle_text = _cell_text(subtitle) if subtitle else ""
    h2_text = " ".join(_cell_text(h) for h in soup.find_all("h2")[:3])
    title_text = _cell_text(soup.title) if soup.title else ""

    if "annual report" in subtitle_text.lower() or "annual report" in title_text.lower():
        doc_type = "annual_report"
        m = re.search(r"(20\d{2})", subtitle_text) or re.search(r"(20\d{2})", title_text)
    elif soup.select_one("div.broker") or re.search(r"results review", h2_text, re.I):
        doc_type = "research_report"
        m = re.search(r"fy\s*(20\d{2})", h2_text, re.I) or re.search(r"(20\d{2})", h2_text)
    else:  # "Fourth Quarter and Full Year 2025 Earnings Conference Call"
        doc_type = "earnings_call"
        m = re.search(r"full year\s*(20\d{2})", h2_text, re.I) or re.search(r"(20\d{2})", h2_text)
    if m is None:  # last resort: the first year near the top of the document
        m = re.search(r"(20\d{2})", _cell_text(soup.body or soup)[:500])
    return company, int(m.group(1)) if m else 2025, doc_type


# What chunk_document learns about the corpus. The runner chunks every document
# before it parses any question, so parse_query may use this (Task 2.2).
_CORPUS: dict[str, Any] = {"companies": {}, "orgs": set()}

_ORG_WORDS = re.compile(r"securities|capital|research|partners|bank|brokerage|advisors|asset management", re.I)


def _register_doc(company: str, soup: BeautifulSoup) -> None:
    """Remember the company and the non-company organisations (brokers etc.) of a document."""
    if company:
        _CORPUS["companies"].setdefault(company_key(company), company)
    broker = soup.select_one("div.broker")  # "Kowloon Peak Securities | Equity Research | Energy"
    if broker:
        name = _cell_text(broker).split("|")[0].strip()
        if name:
            _CORPUS["orgs"].add(company_key(name))
    for li in soup.find_all("li"):  # "Fiona Vasquez - Meridian Equity Research"
        m = re.match(r"^[A-Z][\w .]+-\s*([A-Z][\w&. ]{2,40})$", _cell_text(li))
        if m and _ORG_WORDS.search(m.group(1)):
            _CORPUS["orgs"].add(company_key(m.group(1)))


def _iter_blocks(element: Any, path: list[str], out: list) -> None:
    """Depth-first walk: ('text', str, path) and ('table', html, path) blocks in document order."""
    for child in element.children:
        name = getattr(child, "name", None)
        if not name or name in _SKIP_TAGS:
            continue
        if name in _HEADING_LEVEL:
            text = _cell_text(child)
            if text:
                level = _HEADING_LEVEL[name]
                if level:
                    path[:] = path[:level - 1] + [text]
                out.append(("text", text, list(path)))
        elif name == "table":
            out.append(("table", str(child), list(path)))
        elif name in ("p", "blockquote"):
            text = _cell_text(child)
            if text:
                out.append(("text", text, list(path)))
        elif name in ("ul", "ol"):
            for li in child.find_all("li", recursive=False):
                text = _cell_text(li)
                if text:
                    out.append(("text", "- " + text, list(path)))
        else:
            _iter_blocks(child, path, out)


def _pack(units: list[tuple[list[str], str]], limit: int) -> list[str]:
    """Greedy packing of text blocks: every chunk starts with its section path."""
    chunks: list[str] = []
    lines: list[str] = []
    length, current_path = 0, ""
    for path, text in units:
        prefix = " > ".join(path)[:100]
        while len(text) > limit - len(prefix) - 2:  # a very long block: hard-split on spaces
            cut = text.rfind(" ", 0, limit - len(prefix) - 2)
            cut = cut if cut > limit // 2 else limit - len(prefix) - 2
            if lines:
                chunks.append("\n".join(lines))
                lines, length, current_path = [], 0, ""
            chunks.append((prefix + "\n" + text[:cut]).strip() if prefix else text[:cut])
            text = text[cut:].lstrip()
        if not text:
            continue
        extra = len(text) + 1 + (len(prefix) + 1 if prefix and prefix != current_path else 0)
        if lines and length + extra > limit:
            chunks.append("\n".join(lines))
            lines, length, current_path = [], 0, ""
        if prefix and prefix != current_path:
            lines.append(prefix)
            length += len(prefix) + 1
            current_path = prefix
        lines.append(text)
        length += len(text) + 1
    if lines:
        chunks.append("\n".join(lines))
    return chunks


def _table_chunks(table_html: str, path: list[str], limit: int) -> list[str]:
    """One chunk per table, or several that each repeat the caption if the text is too long."""
    text = table_to_text(table_html)
    prefix = " > ".join(path)[:100]
    headroom = len(prefix) + 2
    if len(text) + headroom <= limit:
        return [(prefix + "\n" + text).strip() if prefix else text]
    lines = text.splitlines()
    caption, rows = (lines[0], lines[1:]) if lines else ("", [])
    parts, current, length = [], [], 0
    for line in rows:
        if current and length + len(line) + 1 > limit - headroom:
            parts.append("\n".join([caption] + current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        parts.append("\n".join([caption] + current))
    return [(prefix + "\n" + c).strip() if prefix else c for c in parts]


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    soup = BeautifulSoup(html, "lxml")
    company, fiscal_year, doc_type = _doc_header_meta(soup, file_name)
    _register_doc(company, soup)
    metadata = {"source": file_name, "company": company,
                "fiscal_year": int(fiscal_year), "doc_type": doc_type}
    body = soup.body or soup

    if not CFG["table_text"]:  # ablation baseline B0: naive text + recursive splitting
        text = _cell_text(body)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_TARGET, chunk_overlap=150)
        return [Document(page_content=piece, metadata=dict(metadata))
                for piece in splitter.split_text(text) if piece.strip()]

    blocks: list = []
    _iter_blocks(body, [], blocks)
    chunks: list[str] = []
    text_units: list[tuple[list[str], str]] = []

    def flush_text() -> None:
        nonlocal text_units
        chunks.extend(_pack(text_units, CHUNK_TARGET))
        text_units = []

    for kind, text, path in blocks:
        if kind == "table":  # a table is its own chunk, next to the text that introduces it
            flush_text()
            chunks.extend(_table_chunks(text, path, CHUNK_TARGET))
        else:
            text_units.append((path, text))
    flush_text()
    return [Document(page_content=chunk, metadata=dict(metadata)) for chunk in chunks]


# ------------------------------------------------------------------ Task 2.2

_PARSE_CACHE: dict[str, dict] = {}

_RESEARCH_WORDS_RE = re.compile(
    r"\b(rating|target price|price target|forecast|results review|estimate|valuation"
    r"|upside|downside|initiate|coverage)\b", re.I)


def _question_year(question: str) -> int | None:
    """The fiscal year the question names ('FY2024', '31 December 2024'), or None."""
    fy = re.findall(r"fy\s*(20\d{2})", question, re.I)
    if fy:
        return int(fy[0])
    years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)
    return int(years[0]) if years else None


def _question_doc_type(question: str) -> str:
    ql = question.lower()
    if "earnings call" in ql or "conference call" in ql:
        return "earnings_call"
    nq = normalize(question)
    if _RESEARCH_WORDS_RE.search(ql) or any(org in nq for org in _CORPUS["orgs"]):
        return "research_report"
    return "annual_report"


_LLM_COMPANIES_PROMPT = """You extract company names from questions asked to an equity-research assistant.

Companies in the document library: {companies}
Brokers, research providers, audit firms and banks (never the subject of a question): {orgs}

Question: {question}

List every company whose report or figures the question asks about, spelled exactly as in the question, in the order they appear. A company that is not in the library list is still listed. Never list brokers, research providers, audit firms, banks or people.
Reply with JSON only, e.g. {{"companies": ["Alpha Industries", "Beta Foods"]}} or {{"companies": []}}."""


def _llm_companies(question: str) -> list[str]:
    """The companies the question names, by the LLM (catches companies not in the corpus)."""
    try:
        llm = make_llm(max_tokens=150)
        prompt = _LLM_COMPANIES_PROMPT.format(
            companies="; ".join(sorted(_CORPUS["companies"].values())) or "(unknown)",
            orgs="; ".join(sorted(_CORPUS["orgs"])) or "(unknown)",
            question=question)
        raw = str(llm.invoke(prompt).content)
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        names = data.get("companies") or []
        return [str(n).strip() for n in names if str(n).strip()]
    except Exception:
        return []


def _question_companies(question: str) -> list[str]:
    """Corpus companies matched by rules, plus anything extra the LLM finds."""
    nq = normalize(question)
    found: dict[str, str] = {}
    for key, display in _CORPUS["companies"].items():
        if key and key in nq:
            found[key] = display
    if CFG["parse_llm"]:
        for name in _llm_companies(question):
            key = company_key(name)
            if key and key not in found and key not in _CORPUS["orgs"]:
                found[key] = name
    return list(found.values())


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    if question not in _PARSE_CACHE:
        _PARSE_CACHE[question] = {
            "companies": _question_companies(question),
            "fiscal_year": _question_year(question),
            "doc_type": _question_doc_type(question),
        }
    return _PARSE_CACHE[question]


# ------------------------------------------------------------------ Tasks 2 & 3

# The years a document actually states: an annual report shows its year and the prior
# one, an earnings call reviews its year and guides the next, a research report
# publishes estimates two years out.
_COVERED_YEARS = {"annual_report": (0, -1), "earnings_call": (0, 1), "research_report": (-1, 0, 1, 2)}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    index: dict[str, Any] = {"chunks": list(chunks), "by_company": {}, "years": {}}
    tokenised = []
    for i, chunk in enumerate(index["chunks"]):
        meta = chunk.metadata or {}
        key = company_key(meta.get("company", ""))
        index["by_company"].setdefault(key, []).append(i)
        fy = as_year(meta.get("fiscal_year"))
        dtype = str(meta.get("doc_type", ""))
        if fy is not None:
            index["years"].setdefault(key, {}).setdefault(dtype, set()).add(fy)
        # Indexed (not stored) text: the metadata as search terms, including the years a
        # document states, so that 'FY2023' finds the FY2024 report that shows 2023.
        header = f"{meta.get('company', '')} {dtype.replace('_', ' ')}"
        if fy is not None:
            header += " " + " ".join(str(fy + d) for d in _COVERED_YEARS.get(dtype, (0,)))
        text = header + "\n" + chunk.page_content
        tokenised.append(_content_tokens(text) if CFG["tokens"] else str(text).lower().split())
    index["tokens"] = tokenised
    index["token_sets"] = [set(t) for t in tokenised]
    index["bm25"] = BM25Okapi(tokenised) if tokenised else None
    return index


def _stem_set(key: str) -> set[str]:
    return {t[:-1] if t.endswith("s") and len(t) > 3 else t for t in key.split()}


def _match_company(key: str, known: list[str]) -> str | None:
    """The known company key that `key` refers to, allowing a dropped plural 's' only."""
    if not key:
        return None
    stem = _stem_set(key)
    best, best_score = None, 0.0
    for other in known:
        both = stem & _stem_set(other)
        score = len(both) / max(len(stem | _stem_set(other)), 1)
        if score > best_score:
            best, best_score = other, score
    return best if best_score >= 0.8 else None


def _covered_years(index: dict, keys: list[str]) -> set[int]:
    covered: set[int] = set()
    for key in keys:
        for dtype, years in index["years"].get(key, {}).items():
            for fy in years:
                covered.update(fy + d for d in _COVERED_YEARS.get(dtype, (0,)))
    return covered


_ANSWERABLE_PROMPT = """You check whether passages from company reports contain the information a question asks for.

Question: {question}

Passages:
{context}

Reply YES if the passages contain the specific information the question asks for (the figure, name, rating or fact, for the company and year asked about). For a comparison, reply YES only if every company asked about is covered. Reply NO if that specific information is not in the passages (for example the figure is not disclosed). Answer with one word: YES or NO."""


def _llm_answerable(question: str, chunks: list[Document]) -> bool:
    """Last abstention check: does the retrieved context really state what is asked?"""
    try:
        llm = make_llm(max_tokens=5)
        prompt = _ANSWERABLE_PROMPT.format(question=question, context=format_context(chunks))
        return not str(llm.invoke(prompt).content).strip().lower().startswith("no")
    except Exception:
        return True  # fail open: a missed answerable question costs more than a missed rejection


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    chunks: list[Document] = index["chunks"]
    if not chunks:
        return []
    if CFG["parse"]:
        parsed = parse_query(question)
    else:
        parsed = {"companies": [], "fiscal_year": None, "doc_type": None}

    known = list(index["by_company"])
    resolved, unknown = [], []
    for name in parsed["companies"]:
        key = company_key(name)
        if key in index["by_company"]:
            resolved.append(key)
        else:
            match = _match_company(key, known)
            (resolved if match else unknown).append(match or name)
    if CFG["abstain"] and unknown:
        return []  # a company the corpus does not cover (e.g. a near-name of a real one)
    companies = resolved or known
    year = as_year(parsed.get("fiscal_year"))
    if CFG["abstain"] and year is not None and year not in _covered_years(index, companies):
        return []  # a year the corpus does not cover (e.g. FY2021)

    query_tokens = _content_tokens(question) if CFG["tokens"] else question.lower().split()
    scores = index["bm25"].get_scores(query_tokens) if index["bm25"] else [0.0] * len(chunks)
    distinct_terms = set(query_tokens)
    idf = getattr(index["bm25"], "idf", {}) or {}

    def states_year(i: int) -> bool:
        """Whether the chunk's document states the named year at all (e.g. FY2023 only
        appears in the FY2024 annual report, as the prior-year column)."""
        meta = chunks[i].metadata or {}
        fy = as_year(meta.get("fiscal_year"))
        return fy is not None and year in {fy + d for d in _COVERED_YEARS.get(str(meta.get("doc_type", "")), (0,))}

    def pool_for(key: str, filtered: bool = True) -> list[int]:
        positions = index["by_company"].get(key, [])
        if filtered and CFG["parse"] and parsed.get("doc_type"):
            typed = [i for i in positions if chunks[i].metadata.get("doc_type") == parsed["doc_type"]]
            if typed:
                positions = typed
        if filtered and CFG["year_boost"] and year is not None:
            stating = [i for i in positions if states_year(i)]
            if stating:
                positions = stating
        return positions

    def score(i: int) -> float:
        s = float(scores[i])
        if CFG["tokens"] and distinct_terms:  # reward chunks that cover the rare query terms
            s += COVERAGE_BONUS * sum(max(0.0, idf.get(t, 0.0))
                                      for t in distinct_terms if t in index["token_sets"][i])
        if CFG["year_boost"] and year is not None:
            fy = as_year(chunks[i].metadata.get("fiscal_year"))
            if fy in (year, year + 1):  # the FY2025 report also states the FY2024 figures
                s *= YEAR_BOOST
        return s

    def pick(filtered: bool) -> list[int]:
        if CFG["per_company"] and len(companies) > 1:
            got: list[int] = []
            rest: list[int] = []
            for key in companies:  # a comparison needs the fact for every company
                ranked = sorted(pool_for(key, filtered), key=score, reverse=True)
                got.extend(ranked[:PER_COMPANY_MIN])
                rest.extend(ranked[PER_COMPANY_MIN:])
            got.extend(sorted(rest, key=score, reverse=True)[:MAX_RETRIEVED - len(got)])
            return got
        pool = [i for key in companies for i in pool_for(key, filtered)] or list(range(len(chunks)))
        return sorted(set(pool), key=score, reverse=True)[:MAX_RETRIEVED]

    result = [chunks[i] for i in pick(filtered=True)[:MAX_RETRIEVED]]
    if CFG["abstain"] and CFG["abstain_llm"] and result and not _llm_answerable(question, result):
        wider = [chunks[i] for i in pick(filtered=False)[:MAX_RETRIEVED]]
        if not wider or not _llm_answerable(question, wider):
            return []  # the figure exists nowhere (e.g. lobbying expenses)
        result = wider
    return result


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
