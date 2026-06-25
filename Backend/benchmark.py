import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

import requests
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK DATASET
# ══════════════════════════════════════════════════════════════════════════════

QUERY_TEST_CASES = [
    {"id": "Q01", "category": "factual",
     "question": "What is the Writ Petition number in the Kesavananda Bharati case?",
     "ground_truth": "The Writ Petition number is W.P.(C) 135 of 1970."},
    {"id": "Q02", "category": "factual",
     "question": "On what date was the Kesavananda Bharati judgment delivered?",
     "ground_truth": "The judgment was delivered on 24th April 1973."},
    {"id": "Q03", "category": "factual",
     "question": "Who was the petitioner in the Kesavananda Bharati case?",
     "ground_truth": "The petitioner was His Holiness Kesavananda Bharati Sripadagalvaru."},
    {"id": "Q04", "category": "factual",
     "question": "How many judges sat on the bench in the Kesavananda Bharati case?",
     "ground_truth": "13 judges sat on the bench."},
    {"id": "Q05", "category": "conceptual",
     "question": "What is the Basic Structure Doctrine established in Kesavananda Bharati?",
     "ground_truth": (
         "The Basic Structure Doctrine holds that while Parliament has wide powers to amend "
         "the Constitution under Article 368, it cannot amend or destroy the basic or essential "
         "features such as supremacy of the Constitution, republican and democratic form of "
         "government, secular character, separation of powers, and federal character.")},
    {"id": "Q06", "category": "conceptual",
     "question": "Which constitutional amendments were challenged in the Kesavananda Bharati case?",
     "ground_truth": "The 24th, 25th, and 29th Constitutional Amendment Acts were challenged."},
    {"id": "Q07", "category": "conceptual",
     "question": "What was the role of the Golaknath case in the Kesavananda Bharati judgment?",
     "ground_truth": (
         "Golaknath held Parliament cannot amend Fundamental Rights. Kesavananda Bharati "
         "overruled Golaknath but introduced the basic structure limitation instead.")},
    {"id": "Q08", "category": "synthesis",
     "question": "What was Justice H.R. Khanna's view on the amendability of fundamental rights?",
     "ground_truth": (
         "Justice Khanna held Parliament can amend Fundamental Rights but cannot abrogate "
         "the entire chapter as that would destroy the basic structure.")},
    {"id": "Q09", "category": "synthesis",
     "question": "What was A.N. Ray J.'s position on Parliament's amending power?",
     "ground_truth": (
         "A.N. Ray J. dissented, holding Parliament has unlimited power to amend any part "
         "of the Constitution including Fundamental Rights, with no implied limitations.")},
    {"id": "Q10", "category": "edge_case",
     "question": "What did Justice Smith say about land reforms in this case?",
     "ground_truth": "There is no Justice Smith in this case. None of the 13 judges is named Smith."},
    {"id": "Q11", "category": "edge_case",
     "question": "What is the punishment for violating the Kerala Land Reforms Act under this judgment?",
     "ground_truth": (
         "The judgment does not prescribe punishments. It only deals with the "
         "constitutional validity of the Kerala Land Reforms Act.")},
]

COMPARE_TEST_CASES = [
    {"id": "C01", "category": "compare_opposing",
     "question": "Compare the views of Sikri CJ and A.N. Ray J. on Parliament's power to amend the Constitution.",
     "ground_truth": (
         "Sikri CJ held Parliament's amending power is limited and cannot damage the basic structure. "
         "A.N. Ray J. dissented, holding Parliament has plenary amending power with no implied limitations.")},
    {"id": "C02", "category": "compare_theme",
     "question": "How do the majority and minority views differ on whether Fundamental Rights can be amended?",
     "ground_truth": (
         "Majority held Fundamental Rights can be amended but not abrogated. Minority "
         "(Ray, Palekar, Mathew, Beg, Dwivedi, Chandrachud) held Parliament has full unrestricted power.")},
    {"id": "C03", "category": "compare_nuanced",
     "question": "Compare how Shelat & Grover JJ. and Chandrachud J. approached implied limitations on amending power.",
     "ground_truth": (
         "Shelat and Grover JJ. strongly affirmed implied limitations from the Constitution's fundamental "
         "features and Preamble. Chandrachud J. rejected implied limitations, holding Article 368 "
         "confers unqualified amending power.")},
]


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION #2 — HASH-BASED JSON CACHE
# ══════════════════════════════════════════════════════════════════════════════

class ScoreCache:
    """
    Caches metric scores keyed on a hash of (question, answer, contexts, metric).
    Survives across reruns — stored as JSON on disk.
    """
    def __init__(self, path: str = "./benchmark_results/score_cache.json", enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._cache = {}
        if enabled and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    @staticmethod
    def _make_key(question: str, answer: str, contexts: list, metric: str) -> str:
        raw = json.dumps({"q": question, "a": answer, "c": contexts, "m": metric}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, question, answer, contexts, metric):
        if not self.enabled:
            return None
        key = self._make_key(question, answer, contexts, metric)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, question, answer, contexts, metric, value):
        if not self.enabled:
            return
        key = self._make_key(question, answer, contexts, metric)
        self._cache[key] = value

    def save(self):
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._cache, f)

    def clear(self):
        self._cache = {}
        if os.path.exists(self.path):
            os.remove(self.path)

    def stats(self):
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0
        return f"{self.hits} hits / {total} lookups ({rate:.0f}% hit rate)"


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION #3 — SENTENCE-TRANSFORMER SCORER (local, free, fast)
# ══════════════════════════════════════════════════════════════════════════════

class LocalScorer:
    """
    Uses the same all-MiniLM-L6-v2 model DocMind already runs locally for
    embeddings, to score answer_relevancy and context_precision via cosine
    similarity instead of burning Gemini calls on them.
    """
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            print("  📦 Loading SentenceTransformer (all-MiniLM-L6-v2)...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def cosine_sim(self, a: str, b: str) -> float:
        from sentence_transformers import util
        model = self._load()
        emb = model.encode([a, b], convert_to_tensor=True)
        score = util.cos_sim(emb[0], emb[1]).item()
        # Cosine similarity is [-1, 1]; clip to [0, 1] for RAGAS-style scoring
        return round(max(0.0, min(1.0, score)), 4)

    def answer_relevancy(self, question: str, answer: str) -> float:
        return self.cosine_sim(question, answer)

    def context_precision(self, question: str, contexts: list) -> float:
        if not contexts:
            return 0.0
        from sentence_transformers import util
        model = self._load()
        q_emb = model.encode(question, convert_to_tensor=True)
        ctx_embs = model.encode(contexts[:5], convert_to_tensor=True)
        sims = util.cos_sim(q_emb, ctx_embs)[0]
        # Threshold-based relevance: treat similarity > 0.3 as "relevant" (tunable)
        relevant = (sims > 0.3).sum().item()
        return round(relevant / len(contexts[:5]), 4)

    def context_precision_score(self, question: str, contexts: list) -> float:
        """Continuous version — mean similarity instead of threshold count. Use whichever fits your reporting."""
        if not contexts:
            return 0.0
        from sentence_transformers import util
        model = self._load()
        q_emb = model.encode(question, convert_to_tensor=True)
        ctx_embs = model.encode(contexts[:5], convert_to_tensor=True)
        sims = util.cos_sim(q_emb, ctx_embs)[0]
        return round(max(0.0, min(1.0, sims.mean().item())), 4)


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS METRICS — Gemini for faithfulness + context_recall, batched where possible
# ══════════════════════════════════════════════════════════════════════════════

class RagasEvaluator:
    def __init__(self, api_key: str, cache: ScoreCache, local_scorer: LocalScorer,
                 use_st_precision: bool = True):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        self.cache = cache
        self.local = local_scorer
        self.use_st_precision = use_st_precision  # OPT #3: ST instead of Gemini for precision

    def _ask(self, prompt: str) -> str:
        for attempt in range(6):
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt)
                return response.text.strip()
            except Exception as e:
                msg = str(e)
                match = re.search(r"retryDelay.*?(\d+)s", msg)
                wait = int(match.group(1)) + 3 if match else 10 * (attempt + 1)
                if attempt < 5 and ("429" in msg or "503" in msg or "RESOURCE_EXHAUSTED" in msg or "UNAVAILABLE" in msg):
                    print(f"\n    ⚠️  Rate limited. Waiting {wait}s then retrying ({attempt+1}/5)...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    def _parse_score(self, text: str) -> float:
        matches = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0*)?)\b", text)
        if matches:
            return round(float(matches[-1]), 4)
        nums = re.findall(r"\d+\.?\d*", text)
        if nums:
            val = float(nums[0])
            return round(min(max(val / 10 if val > 1 else val, 0.0), 1.0), 4)
        return 0.0

    # ── Faithfulness — stays on Gemini (needs real claim-checking reasoning) ──
    def faithfulness(self, question: str, answer: str, contexts: list) -> float:
        cached = self.cache.get(question, answer, contexts, "faithfulness")
        if cached is not None:
            return cached

        ctx_block = "\n\n".join(f"[Context {i+1}]:\n{c[:800]}" for i, c in enumerate(contexts[:5]))
        prompt = f"""You are a RAGAS faithfulness evaluator.

Identify each factual claim in the answer. Check if each claim is directly supported by the context.
Score = supported_claims / total_claims. Return ONLY a decimal between 0.0 and 1.0.

Question: {question}
Retrieved Context:
{ctx_block}
Answer: {answer}

Faithfulness Score:"""
        score = self._parse_score(self._ask(prompt))
        self.cache.set(question, answer, contexts, "faithfulness", score)
        return score

    # ── Answer Relevancy — OPT #3: local SentenceTransformer, no Gemini call ──
    def answer_relevancy(self, question: str, answer: str) -> float:
        cached = self.cache.get(question, answer, [], "answer_relevancy")
        if cached is not None:
            return cached
        score = self.local.answer_relevancy(question, answer)
        self.cache.set(question, answer, [], "answer_relevancy", score)
        return score

    # ── Context Precision — OPT #1 (batched Gemini) or OPT #3 (local ST) ──
    def context_precision(self, question: str, contexts: list) -> float:
        cached = self.cache.get(question, "", contexts, "context_precision")
        if cached is not None:
            return cached

        if self.use_st_precision:
            score = self.local.context_precision_score(question, contexts)
        else:
            score = self._context_precision_batched_gemini(question, contexts)

        self.cache.set(question, "", contexts, "context_precision", score)
        return score

    def _context_precision_batched_gemini(self, question: str, contexts: list) -> float:
        """OPT #1: ONE Gemini call scoring all chunks via JSON, instead of N calls."""
        if not contexts:
            return 0.0
        ctx_block = "\n\n".join(f"[{i+1}]: {c[:600]}" for i, c in enumerate(contexts[:5]))
        prompt = f"""You are a RAGAS context precision evaluator.

For EACH numbered passage below, decide if it is relevant to answering the question.
Return ONLY a JSON object mapping passage number (as string) to 1 (relevant) or 0 (not relevant).
No explanation, no markdown, just raw JSON.

Example output format:
{{"1": 1, "2": 0, "3": 1, "4": 1, "5": 0}}

Question: {question}

Passages:
{ctx_block}

JSON scores:"""
        raw = self._ask(prompt)
        try:
            cleaned = raw.strip().strip("```json").strip("```").strip()
            scores_dict = json.loads(cleaned)
            vals = [float(v) for v in scores_dict.values()]
            return round(sum(vals) / len(vals), 4) if vals else 0.0
        except (json.JSONDecodeError, ValueError, ZeroDivisionError):
            # Fallback: try to extract any 0/1 sequence
            nums = re.findall(r"[01]", raw)
            if nums:
                vals = [float(n) for n in nums[:len(contexts[:5])]]
                return round(sum(vals) / len(vals), 4) if vals else 0.0
            return 0.0

    # ── Context Recall — stays on Gemini (needs reasoning about ground truth coverage) ──
    def context_recall(self, question: str, contexts: list, ground_truth: str) -> float:
        cached = self.cache.get(question, ground_truth, contexts, "context_recall")
        if cached is not None:
            return cached

        ctx_block = "\n\n".join(f"[Context {i+1}]:\n{c[:600]}" for i, c in enumerate(contexts[:5]))
        prompt = f"""You are a RAGAS context recall evaluator.

Check whether the retrieved context contains enough information to derive the ground truth answer.
Score = proportion of ground truth statements attributable to context.
Return ONLY a decimal between 0.0 and 1.0.

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context:
{ctx_block}

Context Recall Score:"""
        score = self._parse_score(self._ask(prompt))
        self.cache.set(question, ground_truth, contexts, "context_recall", score)
        return score

    def score_query(self, question, answer, contexts, ground_truth):
        print("    → faithfulness      [Gemini]", end="", flush=True)
        f = self.faithfulness(question, answer, contexts);              print(f" {f:.3f}")
        print("    → answer_relevancy  [local] ", end="", flush=True)
        ar = self.answer_relevancy(question, answer);                   print(f" {ar:.3f}")
        print(f"    → context_precision [{'local' if self.use_st_precision else 'Gemini-batch'}]", end="", flush=True)
        cp = self.context_precision(question, contexts);                print(f" {cp:.3f}")
        print("    → context_recall    [Gemini]", end="", flush=True)
        cr = self.context_recall(question, contexts, ground_truth);     print(f" {cr:.3f}")
        return {"faithfulness": f, "answer_relevancy": ar, "context_precision": cp, "context_recall": cr}

    def score_compare(self, question, answer, contexts):
        print("    → faithfulness      [Gemini]", end="", flush=True)
        f = self.faithfulness(question, answer, contexts);   print(f" {f:.3f}")
        print("    → answer_relevancy  [local] ", end="", flush=True)
        ar = self.answer_relevancy(question, answer);        print(f" {ar:.3f}")
        print(f"    → context_precision [{'local' if self.use_st_precision else 'Gemini-batch'}]", end="", flush=True)
        cp = self.context_precision(question, contexts);     print(f" {cp:.3f}")
        return {"faithfulness": f, "answer_relevancy": ar, "context_precision": cp}


# ══════════════════════════════════════════════════════════════════════════════
# DOCMIND CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class DocMindClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._all_doc_ids = []

    def list_documents(self):
        r = self.session.get(f"{self.base_url}/documents"); r.raise_for_status()
        docs = r.json().get("documents", [])
        # IMPORTANT: /compare requires the actual doc_id (the key used in store.metadata),
        # NOT the filename. /documents returns both — doc_id is the one /compare needs.
        self._all_doc_ids = [
            d if isinstance(d, str) else d.get("doc_id", d.get("id", str(d)))
            for d in docs
        ]
        return docs

    def _extract_contexts(self, data):
        raw = data.get("contexts", data.get("retrieved_chunks", data.get("chunks", [])))
        return [c if isinstance(c, str) else c.get("text", c.get("content", str(c))) for c in raw]

    def _post_with_retry(self, url, payload, retries=5):
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = self.session.post(url, json=payload, timeout=300)
                latency = round((time.time() - t0) * 1000, 1)
                if r.status_code in (429, 500, 503):
                    wait = 10 * (attempt + 1)
                    print(f"\n    ⚠️  Server returned {r.status_code}. Waiting {wait}s then retrying ({attempt+1}/{retries})...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json(), latency
            except requests.exceptions.ConnectionError:
                wait = 10 * (attempt + 1)
                print(f"\n    ⚠️  Connection error. Waiting {wait}s then retrying ({attempt+1}/{retries})...")
                time.sleep(wait)
        raise RuntimeError(f"Failed after {retries} retries: {url}")

    def query(self, question, top_k=5):
        d, latency = self._post_with_retry(f"{self.base_url}/query", {"question": question})
        return {
            "answer": d.get("answer", d.get("response", "")),
            "contexts": self._extract_contexts(d),
            "latency_ms": latency,
        }

    def compare(self, question, doc_ids=None):
        ids = doc_ids or self._all_doc_ids
        d, latency = self._post_with_retry(f"{self.base_url}/compare", {"query": question, "doc_ids": ids})
        return {
            "answer": d.get("answer", d.get("comparison", d.get("response", ""))),
            "contexts": self._extract_contexts(d),
            "latency_ms": latency,
        }

    def upload_pdf(self, filepath: str, retries: int = 3):
        """POST a single PDF to /upload. Returns the response JSON (includes doc_id)."""
        filename = os.path.basename(filepath)
        for attempt in range(retries):
            try:
                with open(filepath, "rb") as f:
                    files = {"file": (filename, f, "application/pdf")}
                    r = self.session.post(f"{self.base_url}/upload", files=files, timeout=300)
                if r.status_code in (429, 500, 503):
                    wait = 10 * (attempt + 1)
                    print(f"    ⚠️  Upload got {r.status_code}. Waiting {wait}s, retry {attempt+1}/{retries}...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.ConnectionError:
                wait = 10 * (attempt + 1)
                print(f"    ⚠️  Connection error on upload. Waiting {wait}s, retry {attempt+1}/{retries}...")
                time.sleep(wait)
        raise RuntimeError(f"Failed to upload {filename} after {retries} retries")

    def delete_document(self, doc_id: str):
        r = self.session.delete(f"{self.base_url}/documents/{doc_id}", timeout=60)
        r.raise_for_status()
        return r.json()

    def delete_all_documents(self):
        """Wipe every currently uploaded document, so the next chunker starts from a clean slate."""
        docs = self.list_documents()
        deleted = 0
        for d in docs:
            doc_id = d.get("doc_id") if isinstance(d, dict) else d
            if not doc_id:
                continue
            try:
                self.delete_document(doc_id)
                deleted += 1
            except Exception as e:
                print(f"    ⚠️  Could not delete {doc_id}: {e}")
        return deleted

    def upload_folder(self, folder: str):
        """Upload every PDF in a folder. Returns list of (filename, doc_id) tuples."""
        pdfs = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(".pdf")
        )
        if not pdfs:
            raise RuntimeError(f"No PDFs found in {folder}")

        uploaded = []
        for i, fname in enumerate(pdfs):
            path = os.path.join(folder, fname)
            print(f"  📤 [{i+1}/{len(pdfs)}] Uploading {fname}...")
            result = self.upload_pdf(path)
            doc_id = result.get("doc_id", "?")
            print(f"     ✅ doc_id={doc_id}")
            uploaded.append((fname, doc_id))
        return uploaded


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION #4 — TWO-STAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_two_stage_query_benchmark(client, evaluator, local_scorer, top_n=5):
    """
    Stage 1: score ALL queries with cheap local (ST) metrics only — answer_relevancy
             and context_precision. No Gemini calls at all in this stage.
    Stage 2: only the top_n best-performing queries (by combined local score) get
             the expensive Gemini metrics (faithfulness, context_recall).

    This mirrors the chunker-elimination use case: cheap scoring filters out the
    weak performers before spending Gemini quota only on the survivors.
    """
    print("\n" + "═"*60 + "\n  STAGE 1 — Local scoring (no Gemini calls)\n" + "═"*60)
    stage1_rows = []
    for tc in QUERY_TEST_CASES:
        print(f"  [{tc['id']}] {tc['question'][:60]}...")
        try:
            res = client.query(tc["question"])
            ar = local_scorer.answer_relevancy(tc["question"], res["answer"])
            cp = local_scorer.context_precision_score(tc["question"], res["contexts"])
            combined = round((ar + cp) / 2, 4)
            stage1_rows.append({
                "id": tc["id"], "category": tc["category"], "question": tc["question"],
                "ground_truth": tc["ground_truth"], "answer": res["answer"],
                "contexts": res["contexts"], "latency_ms": res["latency_ms"],
                "answer_relevancy": ar, "context_precision": cp, "stage1_score": combined,
            })
            print(f"    local: relevancy={ar:.3f} precision={cp:.3f} → combined={combined:.3f}")
        except Exception as e:
            print(f"    ✗ {e}")
            stage1_rows.append({
                "id": tc["id"], "category": tc["category"], "question": tc["question"],
                "ground_truth": tc["ground_truth"], "answer": "ERROR", "contexts": [],
                "latency_ms": -1, "answer_relevancy": 0, "context_precision": 0, "stage1_score": -1,
            })

    stage1_rows.sort(key=lambda r: r["stage1_score"], reverse=True)
    survivors = stage1_rows[:top_n]
    eliminated = stage1_rows[top_n:]

    print(f"\n  ✅ Stage 1 complete. Top {top_n} survivors advance to Stage 2 (Gemini):")
    for s in survivors:
        print(f"     {s['id']} (score={s['stage1_score']:.3f})")
    print(f"  ❌ Eliminated (no Gemini calls spent): {[e['id'] for e in eliminated]}")

    print("\n" + "═"*60 + "\n  STAGE 2 — Gemini scoring (survivors only)\n" + "═"*60)
    final_rows = []
    for i, s in enumerate(survivors):
        print(f"\n  [{s['id']}] Gemini scoring...")
        f = evaluator.faithfulness(s["question"], s["answer"], s["contexts"])
        cr = evaluator.context_recall(s["question"], s["contexts"], s["ground_truth"])
        print(f"    → faithfulness={f:.3f}  context_recall={cr:.3f}")
        final_rows.append({**s, "faithfulness": f, "context_recall": cr})

    for e in eliminated:
        final_rows.append({**e, "faithfulness": None, "context_recall": None})

    df = pd.DataFrame(final_rows)
    df = df.drop(columns=["contexts", "ground_truth"], errors="ignore")
    return df.sort_values("id").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD (NON-STAGED) RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_query_benchmark(client, evaluator):
    print("\n" + "═"*60 + "\n  /QUERY BENCHMARK\n" + "═"*60)
    rows = []
    for i, tc in enumerate(QUERY_TEST_CASES):
        print(f"\n[{tc['id']}] {tc['category'].upper()}: {tc['question'][:75]}...")
        row = {"id": tc["id"], "category": tc["category"]}
        try:
            res = client.query(tc["question"])
            row.update({"answer": res["answer"], "latency_ms": res["latency_ms"],
                        "contexts_n": len(res["contexts"])})
            print(f"  ({res['latency_ms']}ms, {len(res['contexts'])} chunks) {res['answer'][:90]}...")
            row.update(evaluator.score_query(tc["question"], res["answer"],
                                              res["contexts"], tc["ground_truth"]))
        except Exception as e:
            print(f"  ✗ {e}")
            row.update({"answer": "ERROR", "latency_ms": -1, "contexts_n": 0,
                        "faithfulness": None, "answer_relevancy": None,
                        "context_precision": None, "context_recall": None})
        rows.append(row)
    return pd.DataFrame(rows)


def run_compare_benchmark(client, evaluator):
    print("\n" + "═"*60 + "\n  /COMPARE BENCHMARK\n" + "═"*60)
    rows = []
    for i, tc in enumerate(COMPARE_TEST_CASES):
        print(f"\n[{tc['id']}] {tc['category'].upper()}: {tc['question'][:75]}...")
        row = {"id": tc["id"], "category": tc["category"]}
        try:
            res = client.compare(tc["question"])
            row.update({"answer": res["answer"], "latency_ms": res["latency_ms"],
                        "contexts_n": len(res["contexts"])})
            print(f"  ({res['latency_ms']}ms, {len(res['contexts'])} chunks) {res['answer'][:90]}...")
            row.update(evaluator.score_compare(tc["question"], res["answer"], res["contexts"]))
        except Exception as e:
            print(f"  ✗ {e}")
            row.update({"answer": "ERROR", "latency_ms": -1, "contexts_n": 0,
                        "faithfulness": None, "answer_relevancy": None, "context_precision": None})
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(dq, dc):
    qm = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    cm = ["faithfulness", "answer_relevancy", "context_precision"]
    print("\n" + "═"*60 + "\n  RESULTS SUMMARY\n" + "═"*60)

    if not dq.empty:
        print("\n📊 /QUERY — Per-question:")
        cols = ["id", "category", "latency_ms"] + [m for m in qm if m in dq.columns]
        print(dq[cols].to_string(index=False))
        print(f"\n📈 /QUERY — Aggregate (n={len(dq)}):")
        for m in qm:
            if m in dq.columns:
                v = dq[m].dropna()
                if len(v): print(f"  {m:<25} mean={v.mean():.3f}  min={v.min():.3f}  max={v.max():.3f}")
        gl = dq[dq["latency_ms"] > 0]["latency_ms"]
        if len(gl): print(f"  {'avg latency':<25} {gl.mean():.0f} ms")

    if not dc.empty:
        print("\n📊 /COMPARE — Per-question:")
        cols = ["id", "category", "latency_ms"] + [m for m in cm if m in dc.columns]
        print(dc[cols].to_string(index=False))
        print(f"\n📈 /COMPARE — Aggregate (n={len(dc)}):")
        for m in cm:
            if m in dc.columns:
                v = dc[m].dropna()
                if len(v): print(f"  {m:<25} mean={v.mean():.3f}  min={v.min():.3f}  max={v.max():.3f}")
        gl = dc[dc["latency_ms"] > 0]["latency_ms"]
        if len(gl): print(f"  {'avg latency':<25} {gl.mean():.0f} ms")

    if not dq.empty and "category" in dq.columns:
        print("\n📋 /QUERY — By category:")
        for cat in dq["category"].unique():
            sub = dq[dq["category"] == cat]
            parts = [f"{m[:8]}={sub[m].mean():.2f}" for m in qm if m in sub.columns and sub[m].notna().any()]
            print(f"  {cat:<20} {'  '.join(parts)}")


def save_report(dq, dc, output_dir, cache: ScoreCache, chunker_label: str = "default"):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    qm = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    cm = ["faithfulness", "answer_relevancy", "context_precision"]

    if not dq.empty: dq.to_csv(os.path.join(output_dir, f"query_{ts}.csv"), index=False)
    if not dc.empty: dc.to_csv(os.path.join(output_dir, f"compare_{ts}.csv"), index=False)

    summary = {
        "chunker": chunker_label,
        "timestamp": ts,
        "cache_stats": cache.stats(),
        "query": {
            "n": len(dq),
            "avg_latency_ms": round(dq[dq["latency_ms"] > 0]["latency_ms"].mean(), 1) if not dq.empty else 0,
            "metrics": {m: round(dq[m].mean(), 4) for m in qm if not dq.empty and m in dq.columns and dq[m].notna().any()},
        },
        "compare": {
            "n": len(dc),
            "avg_latency_ms": round(dc[dc["latency_ms"] > 0]["latency_ms"].mean(), 1) if not dc.empty else 0,
            "metrics": {m: round(dc[m].mean(), 4) for m in cm if not dc.empty and m in dc.columns and dc[m].notna().any()},
        },
    }
    sp = os.path.join(output_dir, f"summary_{ts}.json")
    with open(sp, "w") as f: json.dump(summary, f, indent=2)
    print(f"\n💾 Results saved to: {output_dir}/")
    print(f"💾 Cache: {cache.stats()}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-CHUNKER COMPARISON UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def compare_chunkers(output_dir: str):
    """
    Scans benchmark_results/<chunker>/summary_*.json for every chunker folder
    and prints a side-by-side comparison table. Run with --compare-chunkers
    after you've benchmarked naive, hierarchical, semantic, and sentence.
    """
    if not os.path.isdir(output_dir):
        print(f"❌ {output_dir} does not exist yet — run a benchmark first.")
        return

    chunker_summaries = {}
    for entry in sorted(os.listdir(output_dir)):
        folder = os.path.join(output_dir, entry)
        if not os.path.isdir(folder):
            continue
        summary_files = sorted(
            [f for f in os.listdir(folder) if f.startswith("summary_") and f.endswith(".json")]
        )
        if not summary_files:
            continue
        # Use the most recent summary for this chunker
        latest = summary_files[-1]
        with open(os.path.join(folder, latest), "r") as f:
            chunker_summaries[entry] = json.load(f)

    if not chunker_summaries:
        print(f"❌ No summary_*.json files found under {output_dir}/<chunker>/")
        return

    print("\n" + "═"*70 + "\n  CHUNKER COMPARISON\n" + "═"*70)

    q_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n📊 /QUERY metrics by chunker:")
    header = f"  {'chunker':<15}" + "".join(f"{m[:10]:>13}" for m in q_metrics) + f"{'latency_ms':>13}"
    print(header)
    for chunker, summary in chunker_summaries.items():
        qm_vals = summary.get("query", {}).get("metrics", {})
        row = f"  {chunker:<15}"
        for m in q_metrics:
            v = qm_vals.get(m)
            row += f"{v:>13.3f}" if v is not None else f"{'—':>13}"
        lat = summary.get("query", {}).get("avg_latency_ms", 0)
        row += f"{lat:>13.0f}"
        print(row)

    c_metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    print("\n📊 /COMPARE metrics by chunker:")
    header = f"  {'chunker':<15}" + "".join(f"{m[:10]:>13}" for m in c_metrics) + f"{'latency_ms':>13}"
    print(header)
    for chunker, summary in chunker_summaries.items():
        cm_vals = summary.get("compare", {}).get("metrics", {})
        row = f"  {chunker:<15}"
        for m in c_metrics:
            v = cm_vals.get(m)
            row += f"{v:>13.3f}" if v is not None else f"{'—':>13}"
        lat = summary.get("compare", {}).get("avg_latency_ms", 0)
        row += f"{lat:>13.0f}"
        print(row)

    # Save a combined CSV too
    rows = []
    for chunker, summary in chunker_summaries.items():
        row = {"chunker": chunker}
        for m in q_metrics:
            row[f"query_{m}"] = summary.get("query", {}).get("metrics", {}).get(m)
        row["query_avg_latency_ms"] = summary.get("query", {}).get("avg_latency_ms")
        for m in c_metrics:
            row[f"compare_{m}"] = summary.get("compare", {}).get("metrics", {}).get(m)
        row["compare_avg_latency_ms"] = summary.get("compare", {}).get("avg_latency_ms")
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "chunker_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\n💾 Combined comparison saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key")
    parser.add_argument("--output-dir", default="./benchmark_results",
                         help="Root results folder. Actual output goes to <output-dir>/<chunker>/")
    parser.add_argument("--chunker", default="default",
                         help="Label for the chunking strategy currently uploaded in DocMind "
                              "(e.g. naive, hierarchical, semantic, sentence). Results are saved "
                              "to <output-dir>/<chunker>/ so runs for different chunkers never collide.")
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--use-gemini-precision", action="store_true",
                         help="Use batched Gemini call for context_precision instead of local SentenceTransformer (default: local)")
    parser.add_argument("--two-stage", action="store_true",
                         help="Run /query benchmark as two-stage pipeline: ST filters, Gemini only scores survivors")
    parser.add_argument("--top-n", type=int, default=5, help="Survivors advancing to Stage 2 in --two-stage mode")
    parser.add_argument("--no-cache", action="store_true", help="Disable the score cache")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the score cache before running")
    parser.add_argument("--compare-chunkers", action="store_true",
                         help="Skip running a new benchmark — just scan <output-dir>/<chunker>/ "
                              "folders and print a side-by-side comparison table across all chunkers tested so far.")
    parser.add_argument("--test-data-dir", default="./Test_Data",
                         help="Folder containing PDFs to auto-upload before benchmarking (default: ./Test_Data)")
    parser.add_argument("--auto-upload", action="store_true",
                         help="Before benchmarking: wipe all currently uploaded docs, then upload every "
                              "PDF found in --test-data-dir. Use this for the fully automated workflow — "
                              "just restart your server with the new chunker active in upload.py, then run "
                              "this script with --auto-upload and a matching --chunker label.")
    parser.add_argument("--no-wipe", action="store_true",
                         help="With --auto-upload, skip deleting existing documents first (only adds new ones).")
    args = parser.parse_args()

    if args.compare_chunkers:
        compare_chunkers(args.output_dir)
        return

    # Namespace everything under benchmark_results/<chunker>/
    run_output_dir = os.path.join(args.output_dir, args.chunker)
    os.makedirs(run_output_dir, exist_ok=True)

    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No API key. Pass --api-key or set GEMINI_API_KEY."); sys.exit(1)

    client = DocMindClient(args.base_url)
    print(f"🔗 Connecting to DocMind at {args.base_url}")
    print(f"🏷️  Chunker label: {args.chunker}  →  results will be saved to: {run_output_dir}/")
    try:
        docs = client.list_documents()
        print(f"✅ {len(docs)} document(s) currently uploaded")
        for d in docs[:5]:
            if isinstance(d, str):
                print(f"   • {d}")
            else:
                print(f"   • {d.get('filename', '?')}  (doc_id={d.get('doc_id', '?')}, chunks={d.get('chunk_count', '?')})")
    except Exception as e:
        print(f"❌ Cannot reach DocMind: {e}"); sys.exit(1)

    # ── Fully automated upload workflow ──────────────────────────────────────
    if args.auto_upload:
        if not args.no_wipe and docs:
            print(f"\n🗑️  Wiping {len(docs)} existing document(s) before fresh upload...")
            deleted = client.delete_all_documents()
            print(f"   Removed {deleted} document(s)")

        if not os.path.isdir(args.test_data_dir):
            print(f"❌ Test data folder not found: {args.test_data_dir}")
            print(f"   Create it and drop your PDFs there, e.g.: {os.path.abspath(args.test_data_dir)}")
            sys.exit(1)

        print(f"\n📁 Auto-uploading PDFs from {args.test_data_dir}/ ...")
        try:
            uploaded = client.upload_folder(args.test_data_dir)
            print(f"✅ Uploaded {len(uploaded)} document(s) for chunker='{args.chunker}'")
        except Exception as e:
            print(f"❌ Upload failed: {e}"); sys.exit(1)

        # Refresh the doc list / doc_ids cache now that uploads are in
        docs = client.list_documents()
        print(f"📊 DocMind now has {len(docs)} document(s) ready for benchmarking")

    if not docs:
        print("⚠️  No documents uploaded. Either add PDFs to Test_Data/ and rerun with --auto-upload, "
              "or upload manually first.")
        sys.exit(1)

    cache = ScoreCache(
        path=os.path.join(run_output_dir, "score_cache.json"),
        enabled=not args.no_cache,
    )
    if args.clear_cache:
        cache.clear()
        print("🗑️  Cache cleared")

    local_scorer = LocalScorer()
    print("🤖 Gemini evaluator ready (faithfulness + context_recall only)")
    print(f"💾 Cache: {'enabled' if not args.no_cache else 'disabled'}")
    print(f"⚡ Context precision: {'Gemini (batched)' if args.use_gemini_precision else 'local SentenceTransformer'}")

    evaluator = RagasEvaluator(
        api_key, cache, local_scorer,
        use_st_precision=not args.use_gemini_precision,
    )

    dq = pd.DataFrame()
    dc = pd.DataFrame()

    if not args.skip_query:
        if args.two_stage:
            dq = run_two_stage_query_benchmark(client, evaluator, local_scorer, top_n=args.top_n)
        else:
            dq = run_query_benchmark(client, evaluator)

    if not args.skip_compare:
        dc = run_compare_benchmark(client, evaluator)

    cache.save()
    print_summary(dq, dc)
    summary = save_report(dq, dc, run_output_dir, cache, chunker_label=args.chunker)

    print(f"\n✅ Benchmark complete for chunker: {args.chunker}")
    if summary["query"]["metrics"]:
        q = summary["query"]["metrics"]
        print(f"  /query   faith={q.get('faithfulness','?')}  rel={q.get('answer_relevancy','?')}  "
              f"prec={q.get('context_precision','?')}  recall={q.get('context_recall','?')}")
    if summary["compare"]["metrics"]:
        c = summary["compare"]["metrics"]
        print(f"  /compare faith={c.get('faithfulness','?')}  rel={c.get('answer_relevancy','?')}  "
              f"prec={c.get('context_precision','?')}")


if __name__ == "__main__":
    main()