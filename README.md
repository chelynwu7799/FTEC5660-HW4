# FTEC5660 Homework 4

Retrieval for a RAG assistant over company reports (annual reports, earnings calls and
broker research reports). All of the logic is in `hw4.py` (the five functions marked
`YOUR CODE HERE`); `ablation.py` reproduces the ablation study below.

## Setup

```bash
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw4.py --data-folder public_test   # writes results.csv
python3 ablation.py                        # Task 5, writes ablation_results.json
```

## Homework 4 solution

### The pipeline

```
corpus/*.html
    |
    v
chunk_document(file_name, html)                     -- no LLM allowed here
    |  cover page (h1 / subtitle / h2) -> metadata: company, fiscal_year, doc_type
    |  <table> -> table_to_text: caption, then one line per row:
    |             "Revenue | 2025: 2,710.4 | 2024: 2,629.1"
    |  narrative  -> blocks under their h2/h3 section path, packed to <= 1,300 chars;
    |               every chunk starts with its section path; a table is its own
    |               chunk (split by rows, caption repeated, if it is too long)
    v
build_index(chunks)
    |  BM25Okapi over number-aware tokens ("2,710.4", "8.5%" stay whole; stopwords
    |  dropped) of chunk text + metadata terms (company, doc_type, stated years:
    |  an annual report states fy and fy-1, a call fy and fy+1, a report fy-1..fy+2)
    v
question -> parse_query(question)
    |  fiscal_year / doc_type: rules ("FY2024" / "31 December 2024", "earnings call",
    |  "target price", ...);  companies: corpus matching + one LLM extraction call
    |  (the LLM gets the corpus company list and the broker list, so it also catches
    |  companies that are NOT in the corpus)
    v
retrieve(index, question)
    |  company not in the corpus ---------------> []        (abstain, Task 3)
    |  year not covered by any document ---------> []        (abstain, Task 3)
    |  pool = chunks of the asked company (and doc_type, restricted to documents
    |  that state the named year); rank = BM25 + idf coverage bonus over the
    |  distinct query terms; comparisons retrieve per company (>= 2 chunks each)
    v
    |  LLM answerability check on the top-5 --NO--> retry once on a wider pool
    |                                          --still NO--> []  (figure not disclosed)
    v
generate(question, chunks)   -- the fixed generator
```

The generator is fixed, so every design choice serves one goal: the five chunks it sees
must come from the right company, the right year and the right document, with figures
printed in a way that keeps row, column and unit together. Tables are therefore never
flattened; every value keeps its row label, its column header and the caption (the
unit). Retrieval is lexical (BM25), and the hard part — telling Northgate Robotics from
Northgate Payments, and FY2024 from the same figure restated in the FY2025 report — is
done by metadata, where it is exact, instead of by the scorer, where it is not. A
question is first parsed into companies / fiscal year / doc_type; the candidate pool is
then filtered on exactly those fields, so BM25 only ranks chunks that can possibly
state the answer, and a comparison retrieves for each company separately so that one
company can never take all five slots. Abstention is layered: cheap deterministic rules
reject companies or years the corpus does not cover, and one short LLM check ("do these
passages state the exact thing asked?") catches figures that are simply not disclosed,
with a fail-open retry on a wider pool so that an answerable question is almost never
dropped by mistake.

### Ablation study (public_test, 60 questions; `python3 ablation.py`)

| Variant | Change relative to the row above | Recall@5 | Kept | Rejected | Accuracy |
|---------|--------------------------------|----------|------|----------|----------|
| B0 | naive RAG as in Tutorial 7: `get_text()`, `RecursiveCharacterTextSplitter`, whitespace-token BM25 top-5 over the whole corpus, no parsing, never `[]` | 65.2% | 100% | 0% | 66.7% |
| V1 | + `table_to_text` and section-aware chunking | 80.4% | 100% | 0% | 78.3% |
| V2 | + query parsing (rules) and company / doc_type metadata filtering | 82.6% | 100% | 0% | 85.0% |
| V3 | + number-aware tokens, idf coverage bonus, per-company retrieval, LLM company extraction | 100.0% | 100% | 0% | 96.7% |
| V4 | + keep only chunks of documents that state the named year | 100.0% | 100% | 0% | 100.0% |
| V5 | + abstention: unknown-company/year rules + LLM answerability check | 100.0% | 100% | 100% | 100.0% |

- **V1 (+15 points of recall).** With `get_text()` a table becomes a stream of numbers
  that loses the link between a value, its row and its column, so most "table" misses in
  B0 are chunks in which the evidence string never even appears. Turning each row into
  `label | column: value` fixes the majority of them, and keeping the caption preserves
  the unit for the generator.
- **V2 (+7 accuracy).** Filtering the candidate pool by company and doc_type removes
  the cross-company noise that BM25 cannot handle (Northgate Robotics vs Northgate
  Payments share most of their vocabulary). Recall rises only a little because the
  naive whitespace tokeniser still dilutes the distinctive query terms.
- **V3 (+17 recall, +12 accuracy).** Three changes that share one idea — make the
  distinctive terms count: tokens keep figures whole and drop stopwords, a coverage
  bonus rewards chunks that contain the rare query terms (e.g. "inventories", which a
  short cover-page chunk can otherwise outscore without containing), and comparison
  questions retrieve per company. This closes all remaining recall gaps, including the
  comparisons that B0–V2 failed because one company took all five slots.
- **V4 (+3 accuracy, the FY2023 questions).** Recall was already 100%, but the FY2025
  report restates FY2024 figures and vice versa, so a "FY2023" question could surface
  the FY2025 report (which does not show 2023) next to the FY2024 report (which does),
  and the generator picked the wrong column. Restricting the pool to documents that
  actually state the named year (the FY2024 report only, for FY2023) removes the
  ambiguity: 96.7% -> 100%.
- **V5 (rejected 0% -> 100%).** With recall at 100%, accuracy on answerable questions
  cannot improve further; the remaining errors are the unanswerable questions that a
  non-abstaining retriever forces through the generator. Rules catch companies not in
  the corpus and years not covered; the LLM check catches undisclosed figures
  (lobbying expenses, churn rate). The same questions are now NOT_FOUND by design
  instead of by luck.

### Error analysis

The errors along the way, newest last:

- **B0, comparisons (7 of 12 wrong).** Five similar balance-sheet chunks, all of one
  company or of the wrong year — e.g. q021 (cash at 31 December 2024: Northgate
  Robotics vs Northgate Payments) retrieved five "cash" chunks and the generator
  answered NOT_FOUND because one company's figure was missing. Fixed by per-company
  retrieval (V3).
- **B0, flattened tables (9 of 22 table questions wrong).** The evidence number is
  printed in the document but not in any naive chunk with its label, so retrieval
  cannot match it. Fixed by `table_to_text` (V1).
- **B0, near-name companies.** "Northgate Robotic Systems" is not in the corpus, yet
  B0 returned Northgate Robotics' cash figure (241.6) with full confidence. Fixed by
  exact company resolution and the unknown-company rule (V5).
- **B0, wrong column.** q026 (long-term borrowing at 31 December 2023): the evidence
  chunk was retrieved but the generator answered the 2024 column (551.8 instead of
  446.9). Fixed by year-restricted pools (V4).
- **V3 -> V4, the last wrong answerable question.** q050 (Brightwater Software's EMEA
  revenue in FY2023): both the FY2024 and the FY2025 region tables were retrieved; the
  generator read 158.1 (the FY2024 column of the FY2025 report) instead of 493.5 (the
  FY2023 column of the FY2024 report). With the pool restricted to documents stating
  2023, only the FY2024 report remains and the answer is unambiguous.
- **Remaining known risks.** A question whose asked figure appears only in a document
  type the parser did not predict would be missed (the pool falls back to all of the
  company's chunks when the typed pool is empty, which softens this); and the LLM
  answerability check could in principle reject an answerable question — the fail-open
  retry on a wider pool is the mitigation, and on the public set no answerable question
  needed it.

## Task 6: Reflection

(to be added by the author)
