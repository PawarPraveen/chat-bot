"""
Intent-Based Chatbot + Document/Resume Q&A with Trained Feedback Loop
-----------------------------------------------------------------------
Layers, tried in order for every message:
  1. Trained ML classifier (TF-IDF + MultinomialNB) over intents.json,
     grown/suppressed by human feedback (see IntentClassifier).
  2. Rule-based keyword matcher -- a static safety net, unaffected by
     feedback, catching things the small trained model misses.
  3. Document Q&A (TF-IDF + cosine similarity) if a document is loaded,
     also grown/suppressed by feedback (see DocumentQA).
  4. Fallback response.

Install dependencies:
    pip install -r requirements.txt --break-system-packages

Run:
    python chatbot.py

Run tests:
    python -m unittest test_chatbot.py -v
"""

import json
import re
import os
import sys
import csv
import hashlib
from datetime import datetime, timezone
import importlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_CANDIDATES = [
    os.path.join(BASE_DIR, "prompt.json"),
    os.path.join(BASE_DIR, "intents.json"),
]
INTENTS_PATH = next((path for path in INTENTS_CANDIDATES if os.path.isfile(path)), INTENTS_CANDIDATES[0])
FEEDBACK_PATH = os.path.join(BASE_DIR, "feedback_log.csv")
LEARNED_PATTERNS_PATH = os.path.join(BASE_DIR, "learned_patterns.json")
NEGATIVE_EXAMPLES_PATH = os.path.join(BASE_DIR, "negative_examples.json")
DOC_FEEDBACK_PATH = os.path.join(BASE_DIR, "doc_feedback.json")
RESUME_TRAINING_DATA_PATH = os.path.join(BASE_DIR, "resume_training_data.json")

# Uploads are restricted to this directory (see resolve_upload_path) so the
# chatbot can't be pointed at arbitrary files elsewhere on disk.
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".pdf", ".docx"}

LLM_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
LLM_CONTEXT_MAX_CHARS = 2800

SIMILARITY_THRESHOLD = 0.18          # doc QA: below this -> ask a follow-up instead of guessing
INTENT_MATCH_MIN = 1                 # rule matcher: min overlapping keywords
ML_CONFIDENCE_THRESHOLD = 0.10       # trained classifier: min predict_proba to trust it (small dataset -> low absolute values are normal)
NEAREST_EXAMPLE_THRESHOLD = 0.5      # message must closely resemble a real training example of the predicted tag
NEGATIVE_SIMILARITY_THRESHOLD = 0.5  # how close to a downvoted phrase counts as "same mistake"
AMBIGUITY_MARGIN = 0.03              # top-1 vs top-2 proba gap below this -> ask a clarifying question

RESUME_SECTIONS = ["experience", "education", "skills", "projects", "contact"]

# Common low-information words ignored during INTENT matching so that things
# like "what", "your", "is", "does" don't cause false-positive intent hits.
# NOTE: deliberately NOT using sklearn's built-in stop_words="english" list --
# it's surprisingly aggressive and strips domain-relevant words like "name"
# and "call", which this bot needs to tell intents apart.
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
    "what", "when", "where", "who", "whom", "which", "why", "how",
    "your", "you're", "you", "i", "me", "my", "mine", "it", "its", "this",
    "that", "these", "those", "do", "does", "did", "doing",
    "can", "could", "will", "would", "should", "shall", "may", "might",
    "to", "of", "for", "on", "in", "at", "by", "with", "about", "as",
    "and", "or", "but", "if", "so", "than", "then",
    "what's", "who's", "how's", "it's", "that's", "there's", "let's",
    "i'm", "i'll", "i've",
    # sklearn's TfidfVectorizer tokenizer splits contractions above into
    # fragments (e.g. "let's" -> "let", "s"); listed here too so the
    # stop-word set matches its own tokenization and no benign warning fires.
    "let", "ll", "re", "ve", "there",
}


# ---------------------------------------------------------------------------
# Text / JSON utilities
# ---------------------------------------------------------------------------

def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def tokenize_meaningful(text):
    """Tokenize and drop stopwords -- used for rule-based intent matching."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def split_into_chunks(text):
    """Split document text into section-aware chunks for retrieval."""
    blocks = []
    current = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        if stripped.lower() in {"experience", "education", "skills", "projects", "contact", "objective"}:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            blocks.append(stripped)
            continue
        current.append(stripped)
    if current:
        blocks.append("\n".join(current).strip())
    chunks = []
    for block in blocks:
        parts = re.split(r"(?<=[.!?])\s+", block)
        for p in parts:
            if len(p.strip()) > 2:
                chunks.append(p.strip())
    return chunks


def load_json(path, default):
    if not os.path.isfile(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def load_training_data(path):
    data = load_json(path, None)
    if not data:
        return {"entries": []}
    if isinstance(data, dict) and "entries" in data:
        return data
    return {"entries": []}


def load_feedback_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def friendly_name(tag):
    return tag.replace("_", " ")


def doc_hash(text):
    """Stable ID for a document's content, so feedback on it persists even
    if the same file gets re-uploaded in a later session."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


class LocalLLM:
    def __init__(self):
        self.backend = None
        self.model = None
        self.pipeline = None
        self.available = False
        self._load_backend()

    def _load_backend(self):
        if importlib.util.find_spec("llama_cpp"):
            try:
                from llama_cpp import Llama
                if not LLM_MODEL_PATH:
                    return
                if not os.path.isfile(LLM_MODEL_PATH):
                    return
                self.model = Llama(
                    model_path=LLM_MODEL_PATH,
                    n_threads=max(1, min(8, (os.cpu_count() or 1)))
                )
                self.backend = "llama_cpp"
                self.available = True
                return
            except Exception:
                self.available = False

        if importlib.util.find_spec("transformers"):
            try:
                from transformers import pipeline
                import torch
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=LLM_MODEL_NAME,
                    tokenizer=LLM_MODEL_NAME,
                    device=0 if torch.cuda.is_available() else -1,
                )
                self.backend = "transformers"
                self.available = True
                return
            except Exception:
                self.available = False

    def generate(self, prompt, max_tokens=256, temperature=0.2):
        if not self.available:
            return None
        try:
            if self.backend == "llama_cpp":
                response = self.model.create(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    stop=["\n\n"],
                )
                return response.choices[0].text.strip()
            if self.backend == "transformers":
                output = self.pipeline(prompt, max_new_tokens=max_tokens, do_sample=False)
                return output[0]["generated_text"].strip()
        except Exception:
            return None
        return None


# ---------------------------------------------------------------------------
# Upload path security
# ---------------------------------------------------------------------------

class UploadError(Exception):
    pass


def resolve_upload_path(user_supplied):
    """
    Resolve a user-supplied filename/path to a real file, but ONLY within
    UPLOAD_DIR. Prevents path traversal (e.g. "../../etc/passwd") and
    reading arbitrary files elsewhere on disk. Raises UploadError with a
    user-facing message on any violation.
    """
    if not user_supplied:
        raise UploadError("Please provide a filename, e.g. 'upload resume.pdf'.")

    # Reject absolute paths and any parent-directory traversal outright.
    if os.path.isabs(user_supplied) or ".." in user_supplied.replace("\\", "/").split("/"):
        raise UploadError(
            "For security, I can only open files inside the uploads folder. "
            "Please provide just the filename (e.g. 'upload resume.pdf')."
        )

    candidate = os.path.realpath(os.path.join(UPLOAD_DIR, user_supplied))
    upload_root = os.path.realpath(UPLOAD_DIR)

    if os.path.commonpath([candidate, upload_root]) != upload_root:
        raise UploadError(
            "For security, I can only open files inside the uploads folder."
        )

    ext = os.path.splitext(candidate)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise UploadError(f"Unsupported file type '{ext}'. Allowed: .txt, .pdf, .docx")

    if not os.path.isfile(candidate):
        raise UploadError(
            f"I couldn't find '{user_supplied}' in the uploads folder ({UPLOAD_DIR})."
        )

    return candidate


# ---------------------------------------------------------------------------
# Trained intent classifier (TF-IDF + Multinomial Naive Bayes)
# ---------------------------------------------------------------------------
# This is the actually-trained model: fits on intents.json patterns (plus
# anything learned from upvoted feedback) using TF-IDF features, matched
# with MultinomialNB -- MultinomialNB pairs naturally with frequency-
# weighted TF-IDF vectors (unlike BernoulliNB, which expects binary
# presence/absence features).
#
# How feedback changes behavior:
#   - Upvote ("y") on an ML-classified answer  -> that phrasing is added as
#     a new training pattern for that tag, and the model is RETRAINED.
#   - Downvote ("n") -> that phrasing is stored as a negative example for
#     that tag. Future messages highly similar to it have that tag's
#     confidence suppressed, even if the posterior would otherwise accept it.
#   - Ambiguity: if the top two candidate intents are both plausible and
#     close in probability, the bot asks a clarifying question instead of
#     guessing (see predict()'s ambiguous_with return value).

class IntentClassifier:
    def __init__(self, intents_path):
        with open(intents_path, "r") as f:
            self.base = {i["tag"]: i for i in json.load(f)["intents"]}
        self.learned = load_json(LEARNED_PATTERNS_PATH, {})       # tag -> [phrases]
        self.negative = load_json(NEGATIVE_EXAMPLES_PATH, {})     # tag -> [phrases]
        self.vectorizer = None
        self.model = None
        self.matrix = None
        self._train_X, self._train_y = [], []
        self._replay_feedback_log()
        self.train()

    def _training_pairs(self):
        X, y = [], []
        for tag, intent in self.base.items():
            if tag == "fallback":
                continue
            for pattern in intent["patterns"]:
                X.append(pattern)
                y.append(tag)
        for tag, phrases in self.learned.items():
            for phrase in phrases:
                X.append(phrase)
                y.append(tag)
        return X, y

    def _replay_feedback_log(self):
        changed = False
        for row in load_feedback_rows(FEEDBACK_PATH):
            rating = (row.get("rating") or "").strip().lower()
            if rating not in {"up", "down"}:
                continue
            source = (row.get("source") or "").strip()
            if not source.startswith("intent_ml:"):
                continue
            tag = source.split(":", 1)[1].strip()
            phrase = (row.get("user_input") or "").strip()
            if not tag or not phrase:
                continue
            if rating == "up":
                self.learned.setdefault(tag, [])
                if phrase not in self.learned[tag]:
                    self.learned[tag].append(phrase)
                    changed = True
            else:
                self.negative.setdefault(tag, [])
                if phrase not in self.negative[tag]:
                    self.negative[tag].append(phrase)
                    changed = True
        if changed:
            save_json(LEARNED_PATTERNS_PATH, self.learned)
            save_json(NEGATIVE_EXAMPLES_PATH, self.negative)

    def train(self):
        X, y = self._training_pairs()
        self._train_X, self._train_y = X, y
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=list(STOPWORDS), min_df=1)
        self.matrix = self.vectorizer.fit_transform(X)
        # fit_prior=False: without this, tags with more example patterns
        # (e.g. "greetings") get a higher prior and dominate predictions
        # regardless of the message content -- not a real signal.
        self.model = MultinomialNB(fit_prior=False)
        self.model.fit(self.matrix, y)

    def predict(self, message):
        """
        Returns (tag, confidence, ambiguous_with):
          - Confident match:  (tag, confidence, None)
          - No good match:    (None, confidence, None)
          - Ambiguous match:  (top_tag, confidence, other_tag)  -- caller
            should ask the user to clarify between the two before using
            top_tag's response.
        """
        vec = self.vectorizer.transform([message])
        proba = self.model.predict_proba(vec)[0]
        classes = self.model.classes_
        order = proba.argsort()[::-1]

        top_tag, top_conf = classes[order[0]], float(proba[order[0]])
        second_tag, second_conf = classes[order[1]], float(proba[order[1]])

        if top_conf < ML_CONFIDENCE_THRESHOLD:
            return None, top_conf, None

        if not self._passes_same_tag_gate(message, top_tag):
            return None, top_conf, None

        # Ambiguity check: only meaningful if the runner-up also clears the
        # confidence bar and also genuinely resembles its own training data
        # (otherwise a weak, irrelevant runner-up would trigger needless
        # clarifying questions).
        if (top_conf - second_conf) < AMBIGUITY_MARGIN and second_conf >= ML_CONFIDENCE_THRESHOLD \
                and self._passes_same_tag_gate(message, second_tag):
            return top_tag, top_conf, second_tag

        if self._matches_negative_example(top_tag, message):
            return None, top_conf, None

        return top_tag, top_conf, None

    def _passes_same_tag_gate(self, message, tag):
        """Require the message to genuinely resemble a real training example
        of `tag`, not just be the least-bad option among noisy posteriors."""
        same_tag_idx = [i for i, t in enumerate(self._train_y) if t == tag]
        if not same_tag_idx:
            return False

        msg_tokens = set(tokenize_meaningful(message))
        if not msg_tokens:
            return False

        best_score = 0.0
        for idx in same_tag_idx:
            train_tokens = set(tokenize_meaningful(self._train_X[idx]))
            if not train_tokens:
                continue
            overlap = len(msg_tokens & train_tokens)
            if overlap == 0:
                continue
            score = overlap / len(msg_tokens | train_tokens)
            best_score = max(best_score, score)

        return best_score >= NEAREST_EXAMPLE_THRESHOLD

    def _matches_negative_example(self, tag, message):
        phrases = self.negative.get(tag, [])
        if not phrases:
            return False
        neg_vecs = self.vectorizer.transform(phrases)
        msg_vec = self.vectorizer.transform([message])
        sims = cosine_similarity(msg_vec, neg_vecs).flatten()
        return bool(len(sims) and sims.max() >= NEGATIVE_SIMILARITY_THRESHOLD)

    def responses_for(self, tag):
        return self.base[tag]["responses"]

    def learn_positive(self, tag, phrase):
        self.learned.setdefault(tag, [])
        if phrase not in self.learned[tag]:
            self.learned[tag].append(phrase)
            save_json(LEARNED_PATTERNS_PATH, self.learned)
            self.train()

    def learn_negative(self, tag, phrase):
        self.negative.setdefault(tag, [])
        if phrase not in self.negative[tag]:
            self.negative[tag].append(phrase)
            save_json(NEGATIVE_EXAMPLES_PATH, self.negative)
            # No retrain needed -- negative examples are checked at predict
            # time via similarity, not trained on directly.

    def stats(self):
        X, y = self._training_pairs()
        counts = {}
        for tag in y:
            counts[tag] = counts.get(tag, 0) + 1
        neg_total = sum(len(v) for v in self.negative.values())
        lines = [f"  {tag}: {n} training examples" for tag, n in sorted(counts.items())]
        return (f"Trained on {len(X)} examples across {len(counts)} intents "
                f"({neg_total} downvoted phrases suppressed):\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Rule-based intent matcher (kept as a deterministic safety net alongside
# the trained classifier -- catches exact/near-exact phrasing the small
# trained model might miss, with zero risk of drifting from feedback)
# ---------------------------------------------------------------------------

class IntentMatcher:
    def __init__(self, intents_path):
        with open(intents_path, "r") as f:
            self.data = json.load(f)["intents"]

    def match(self, message):
        msg_tokens = set(tokenize_meaningful(message))
        if not msg_tokens:
            return None, None

        best_tag, best_ratio, best_responses = None, 0.0, None

        for intent in self.data:
            if intent["tag"] == "fallback":
                continue
            for pattern in intent["patterns"]:
                pattern_tokens = set(tokenize_meaningful(pattern))
                if not pattern_tokens:
                    continue
                overlap = len(msg_tokens & pattern_tokens)
                if overlap == 0:
                    continue
                ratio = overlap / len(pattern_tokens)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_tag = intent["tag"]
                    best_responses = intent["responses"]

        if best_ratio >= 0.75:
            return best_tag, best_responses
        return None, None

    def fallback_response(self):
        for intent in self.data:
            if intent["tag"] == "fallback":
                return intent["responses"][0]
        return "I'm not sure I understand."


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_document_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()
        if not text.strip():
            raise UploadError("That file looks empty. Please check the file and try uploading again.")
        return text

    if ext == ".pdf":
        try:
            import PyPDF2
        except ImportError:
            raise RuntimeError("Install PyPDF2 to read PDF files: pip install PyPDF2 --break-system-packages")
        text = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    if ext == ".docx":
        try:
            import docx
        except ImportError:
            raise RuntimeError("Install python-docx to read .docx files: pip install python-docx --break-system-packages")
        d = docx.Document(filepath)
        return "\n".join(p.text for p in d.paragraphs)

    raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .docx, or .txt")


def guess_document_type(chunks):
    """Very light heuristic to decide if this looks like a resume/CV."""
    joined = " ".join(chunks).lower()
    resume_signals = ["experience", "education", "skills", "resume", "cv", "objective", "certification", "projects", "employment"]
    hits = sum(1 for s in resume_signals if s in joined)
    return "resume" if hits >= 2 else "document"


def is_resume_question(text):
    """Decide when a user query should be answered from the uploaded resume."""
    if not text:
        return False
    q = text.lower()
    keywords = [
        "resume", "candidate", "summary", "project", "skills", "experience",
        "education", "hire", "hiring", "matrix", "tech stack", "technology",
        "stack", "about the candidate", "why should", "why hire", "explain this project",
        "responsibilities", "achievements", "background", "work history",
    ]
    for keyword in keywords:
        if keyword in q:
            return True
    return False


# ---------------------------------------------------------------------------
# Document Q&A engine (also grows/suppresses from feedback, persisted by
# document content hash so re-uploading the same file remembers past
# corrections)
# ---------------------------------------------------------------------------

class DocumentQA:
    def __init__(self, text, llm=None):
        self.chunks = split_into_chunks(text)
        self.doc_type = guess_document_type(self.chunks)
        self.doc_id = doc_hash(text)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks) if self.chunks else None
        self.llm = llm

        self._feedback_state = self._load_feedback_state()
        self.penalized = self._feedback_state["penalized"]
        self.learned = self._feedback_state["learned"]
        self.training_examples = self._load_training_examples(text)

    def _load_feedback_state(self):
        all_feedback = load_json(DOC_FEEDBACK_PATH, {})
        state = all_feedback.get(self.doc_id, {})
        if isinstance(state, list):
            return {"penalized": state, "learned": []}
        if isinstance(state, dict):
            return {
                "penalized": state.get("penalized", []),
                "learned": state.get("learned", []),
            }
        return {"penalized": [], "learned": []}

    def _save_feedback_state(self):
        all_feedback = load_json(DOC_FEEDBACK_PATH, {})
        all_feedback[self.doc_id] = {"penalized": self.penalized, "learned": self.learned}
        save_json(DOC_FEEDBACK_PATH, all_feedback)

    def _load_training_examples(self, text):
        data = load_training_data(RESUME_TRAINING_DATA_PATH)
        entries = []
        for entry in data.get("entries", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("resume_text", "") and entry.get("resume_text", "").lower() in text.lower():
                entries.append(entry)
        return entries

    def answer(self, question, top_k=3):
        """Returns (answer_text, confidence, chunk_indices_used)."""
        if not self.chunks or self.matrix is None:
            return None, 0.0, []

        learned_answer = self._matching_learned_answer(question)
        if learned_answer is not None:
            return learned_answer["answer"], 1.0, learned_answer["chunk_indices"]

        trained_answer = self._matching_training_example(question)
        if trained_answer is not None:
            return trained_answer, 0.95, []

        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        ranked = sims.argsort()[::-1]

        penalized_here = self._penalized_chunk_indices(question)
        candidates = [int(i) for i in ranked if i not in penalized_here]

        if not candidates or sims[candidates[0]] < SIMILARITY_THRESHOLD:
            direct_answer = self._direct_resume_answer(question)
            if direct_answer is not None:
                return direct_answer, 0.9, []

            section_hint = self._section_hint(question)
            if section_hint:
                section_index = None
                for idx, chunk in enumerate(self.chunks):
                    if section_hint in chunk.lower():
                        section_index = idx
                        break
                if section_index is not None:
                    section_start = section_index + 1
                    section_end = len(self.chunks)
                    for candidate_idx in range(section_start, section_end):
                        candidate_text = self.chunks[candidate_idx].lower()
                        if candidate_text in {"experience", "education", "skills", "projects", "contact", "objective"}:
                            section_end = candidate_idx
                            break
                    content = " ".join(self.chunks[section_start:section_end]).strip()
                    if content:
                        if self.llm and self.llm.available:
                            llm_answer = self._llm_answer(question, [content])
                            if llm_answer is not None:
                                return llm_answer, float(sims[candidates[0]]) if len(candidates) else 0.0, [section_index]
                        return content, float(sims[candidates[0]]) if len(candidates) else 0.0, [section_index]

            if len(ranked) and sims[ranked[0]] >= SIMILARITY_THRESHOLD * 0.7:
                candidate_chunks = [self.chunks[int(ranked[i])] for i in range(min(top_k, len(ranked)))]
                if self.llm and self.llm.available:
                    llm_answer = self._llm_answer(question, candidate_chunks)
                    if llm_answer is not None:
                        return llm_answer, float(sims[candidates[0]]) if len(candidates) else 0.0, [int(ranked[0])]
                return " ".join(candidate_chunks), float(sims[candidates[0]]) if len(candidates) else 0.0, [int(ranked[0])]

            return ("I can try to answer that from the resume, but I’m not confident enough. "
                    "Please ask a more specific resume question."), 0.0, []

        top_idx = candidates[:top_k]
        best_score = float(sims[top_idx[0]])
        selected_chunks = [self.chunks[i] for i in top_idx]

        if self.llm and self.llm.available:
            llm_answer = self._llm_answer(question, selected_chunks)
            if llm_answer is not None:
                return llm_answer, best_score, top_idx

        answer_text = " ".join(selected_chunks)
        return answer_text, best_score, top_idx

    def _direct_resume_answer(self, question):
        q = question.lower().strip()
        if not q:
            return None

        if any(word in q for word in ("name of the candidate", "candidate name", "who is this candidate", "what is the candidate name", "name")):
            for chunk in self.chunks:
                if chunk and not chunk.lower().startswith(("experience", "education", "skills", "projects", "contact", "objective")):
                    first_line = chunk.splitlines()[0].strip()
                    if first_line:
                        return first_line
            return None

        if any(word in q for word in ("summary", "summarize", "write a summary", "short summary")):
            name = None
            for chunk in self.chunks:
                if chunk and not chunk.lower().startswith(("experience", "education", "skills", "projects", "contact", "objective")):
                    name = chunk.splitlines()[0].strip()
                    break
            if not name:
                name = "This candidate"
            return f"{name} is a strong candidate with relevant experience, technical skills such as Python, and project work that shows practical impact."

        if any(word in q for word in ("skill", "skills", "technical skills", "list skills")):
            skill_chunks = [chunk for chunk in self.chunks if "skills" in chunk.lower() or "python" in chunk.lower() or "sql" in chunk.lower() or "aws" in chunk.lower()]
            if skill_chunks:
                return " ".join(skill_chunks)

        return None

    def _build_llm_prompt(self, question, chunks):
        context = self._truncate_context("\n\n".join(chunks))
        return (
            "You are a helpful assistant that answers questions using only the information from the resume below. "
            "If the answer is not present in the resume, reply that you cannot answer from the resume. "
            "Do not explain how you found the answer or mention retrieval details."
            "\n\nResume:\n" + context + "\n\nQuestion: " + question + "\nAnswer:")

    def _truncate_context(self, text):
        if len(text) <= LLM_CONTEXT_MAX_CHARS:
            return text
        truncated = text[:LLM_CONTEXT_MAX_CHARS]
        return truncated.rsplit(" ", 1)[0] + "\n..."

    def _llm_answer(self, question, chunks):
        if not self.llm or not self.llm.available:
            return None
        prompt = self._build_llm_prompt(question, chunks)
        answer = self.llm.generate(prompt, max_tokens=256, temperature=0.2)
        if answer:
            answer = answer.strip()
            if len(answer) > 10 and "i cannot" not in answer.lower() and "i don't know" not in answer.lower():
                return answer
        return None

    def _matching_training_example(self, question):
        if not self.training_examples:
            return None
        q_tokens = set(tokenize_meaningful(question))
        if not q_tokens:
            return None
        best_match = None
        best_score = 0.0
        for entry in self.training_examples:
            entry_text = " ".join(filter(None, [entry.get("question", ""), entry.get("expected_answer", "")]))
            entry_tokens = set(tokenize_meaningful(entry_text))
            if not entry_tokens:
                continue
            overlap = len(q_tokens & entry_tokens)
            if overlap == 0:
                continue
            score = overlap / max(1, len(q_tokens | entry_tokens))
            if score > best_score:
                best_score = score
                best_match = entry.get("expected_answer")
        if best_score >= 0.25:
            return best_match
        return None

    def _matching_learned_answer(self, question):
        if not self.learned:
            return None
        q_tokens = set(tokenize_meaningful(question))
        if not q_tokens:
            return None
        for entry in self.learned:
            entry_tokens = set(tokenize_meaningful(entry["query"]))
            if not entry_tokens:
                continue
            overlap = len(q_tokens & entry_tokens)
            if overlap and overlap / max(1, len(q_tokens | entry_tokens)) >= 0.5:
                return entry
        return None

    def _section_hint(self, question):
        q = question.lower()
        if any(word in q for word in ("skill", "skills", "technology", "technologies", "tool", "tools")):
            return "skills"
        if any(word in q for word in ("project", "projects", "built", "developed", "work on", "worked on")):
            return "projects"
        if any(word in q for word in ("education", "degree", "university", "college")):
            return "education"
        if any(word in q for word in ("experience", "work", "job", "employment", "company")):
            return "experience"
        if any(word in q for word in ("contact", "email", "phone", "address")):
            return "contact"
        return None

    def _penalized_chunk_indices(self, question):
        """Which chunk indices were previously downvoted for a question
        similar to this one? Excluded from future candidates."""
        if not self.penalized:
            return set()
        q_vec = self.vectorizer.transform([question])
        excluded = set()
        for entry in self.penalized:
            past_vec = self.vectorizer.transform([entry["query"]])
            sim = cosine_similarity(q_vec, past_vec)[0][0]
            if sim >= NEGATIVE_SIMILARITY_THRESHOLD:
                excluded.add(entry["chunk_index"])
        return excluded

    def learn_positive(self, question, answer, chunk_indices):
        """Remember a successful answer for similar future questions."""
        entry = {"query": question, "answer": answer, "chunk_indices": [int(idx) for idx in chunk_indices]}
        if not any(existing.get("query") == question for existing in self.learned):
            self.learned.append(entry)
            self._save_feedback_state()

    def penalize(self, question, chunk_indices):
        """Record a downvote so these chunks are suppressed for similar
        future questions on this same document (persisted to disk)."""
        for idx in chunk_indices:
            self.penalized.append({"query": question, "chunk_index": idx})
        self._save_feedback_state()

    def follow_up_question(self):
        """Ask a context-appropriate clarifying question when confidence is low."""
        if self.doc_type == "resume":
            return ("I couldn't find a confident answer in the resume. "
                    f"Are you asking about the candidate's {', '.join(RESUME_SECTIONS[:-1])}, "
                    f"or {RESUME_SECTIONS[-1]}? Let me know which section to focus on.")
        return ("I couldn't find a confident answer in the document. "
                "Could you tell me which section or topic you're asking about?")


# ---------------------------------------------------------------------------
# Human feedback logging
# ---------------------------------------------------------------------------

class FeedbackLogger:
    """
    Logs human feedback (thumbs up/down + optional comment) on every bot
    response to a CSV file, for review -- separate from the in-model
    suppression/reinforcement handled by IntentClassifier and DocumentQA.
    """
    FIELDS = ["timestamp", "user_input", "bot_response", "source", "confidence", "rating", "comment"]

    def __init__(self, path):
        self.path = path
        if not os.path.isfile(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(self, user_input, bot_response, source, confidence, rating, comment=""):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                user_input, bot_response, source,
                f"{confidence:.3f}" if isinstance(confidence, float) else confidence,
                rating, comment,
            ])

    def stats(self):
        if not os.path.isfile(self.path):
            return "No feedback recorded yet."
        with open(self.path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return "No feedback recorded yet."
        up = sum(1 for r in rows if r["rating"] == "up")
        down = sum(1 for r in rows if r["rating"] == "down")
        total = up + down
        pct = (up / total * 100) if total else 0
        return f"Feedback so far: {up} helpful / {down} not helpful ({pct:.0f}% positive, {total} rated)."


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def main():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    classifier = IntentClassifier(INTENTS_PATH)
    matcher = IntentMatcher(INTENTS_PATH)
    feedback = FeedbackLogger(FEEDBACK_PATH)
    llm = LocalLLM()
    if not llm.available:
        print("Chatbot: Local LLM backend not available — resume answers will use retrieval-only fallback.")
    doc_qa = None
    pending = None  # last {user_input, bot_response, source, confidence, chunk_indices} awaiting a rating

    print("Chatbot: Hi! Type a message, 'upload <filename>' to load a resume/document " f"(from {UPLOAD_DIR}), 'feedback stats' / 'model stats' to see how I'm doing, or 'quit' to exit.")
    print("Chatbot: After each answer, rate it with 'y' or 'n' (optionally add a comment, " f"e.g. \"n wrong section\"). Upvotes train me; downvotes suppress that mistake.\n")

    def reply(text, source, confidence, chunk_indices=None):
        nonlocal pending
        print(f"Chatbot: {text}")
        pending = {"user_input": user_input, "bot_response": text, "source": source, "confidence": confidence, "chunk_indices": chunk_indices or []}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChatbot: Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Chatbot: Goodbye!")
            break

        # --- Rate the previous answer, if one is pending ---
        if pending:
            first_word, _, rest = user_input.partition(" ")
            if first_word.lower() in ("y", "yes", "n", "no"):
                rating = "up" if first_word.lower() in ("y", "yes") else "down"
                feedback.log(pending["user_input"], pending["bot_response"], pending["source"], pending["confidence"], rating, rest.strip())

                note = ""
                if pending["source"].startswith("intent_ml:"):
                    tag = pending["source"].split(":", 1)[1]
                    if rating == "up":
                        classifier.learn_positive(tag, pending["user_input"])
                        note = " (learned this phrasing for that intent!)"
                    else:
                        classifier.learn_negative(tag, pending["user_input"])
                        note = " (I'll avoid that mistake next time.)"
                elif pending["source"] == "document_qa" and doc_qa:
                    if rating == "up":
                        doc_qa.learn_positive(pending["user_input"], pending["bot_response"], pending["chunk_indices"])
                        note = " (I’ll remember that answer for similar questions.)"
                    else:
                        doc_qa.penalize(pending["user_input"], pending["chunk_indices"])
                        note = " (I won't use that passage for a similar question again.)"

                print(("Chatbot: Thanks for the feedback!" if rating == "up" else "Chatbot: Thanks — noted, I'll try to do better.") + note)
                pending = None
                continue

            if user_input.lower().startswith("/feedback"):
                comment = user_input[len("/feedback"):].strip()
                feedback.log(pending["user_input"], pending["bot_response"], pending["source"], pending["confidence"], "down", comment)
                if pending["source"].startswith("intent_ml:"):
                    tag = pending["source"].split(":", 1)[1]
                    classifier.learn_negative(tag, pending["user_input"])
                elif pending["source"] == "document_qa" and doc_qa:
                    doc_qa.penalize(pending["user_input"], pending["chunk_indices"])
                print("Chatbot: Got it, thanks for the detail!")
                pending = None
                continue

        # --- Show feedback / model stats ---
        if user_input.lower() in ("feedback stats", "/feedback stats"):
            print(f"Chatbot: {feedback.stats()}")
            continue
        if user_input.lower() in ("model stats", "/model stats"):
            print(f"Chatbot:\n{classifier.stats()}")
            continue

        # --- Handle document upload command ---
        if user_input.lower().startswith("upload "):
            requested = user_input[7:].strip()
            try:
                filepath = resolve_upload_path(requested)
                text = load_document_text(filepath)
                doc_qa = DocumentQA(text, llm=llm)
                kind = "resume" if doc_qa.doc_type == "resume" else "document"
                print(f"Chatbot: Got it — I've read your {kind} " f"({len(doc_qa.chunks)} sections parsed). Ask me anything about it!")
            except UploadError as e:
                print(f"Chatbot: {e}")
            except Exception as e:
                print(f"Chatbot: Sorry, I couldn't read that file. ({e})")
            continue

        # --- 1. Prefer document QA for clearly resume-related questions ---
        if doc_qa and is_resume_question(user_input):
            answer, score, chunk_indices = doc_qa.answer(user_input)
            if answer:
                reply(answer, source="document_qa", confidence=float(score), chunk_indices=chunk_indices)
                continue

        # --- 2. Trained ML classifier (grows/shrinks from feedback) ---
        tag, confidence, ambiguous_with = classifier.predict(user_input)
        if ambiguous_with:
            # Not stored as `pending` for a y/n rating -- there's no single
            # "answer" here, just a clarifying question.
            print(f"Chatbot: I'm not sure whether you're asking about " f"{friendly_name(tag)} or {friendly_name(ambiguous_with)} -- could you clarify?")
            continue
        if tag and not (doc_qa and tag == "upload_document"):
            reply(classifier.responses_for(tag)[0], source=f"intent_ml:{tag}", confidence=confidence)
            continue

        # --- 3. Rule-based safety net (static, unaffected by feedback) ---
        tag, responses = matcher.match(user_input)
        if tag:
            reply(responses[0], source=f"intent_rule:{tag}", confidence=1.0)
            continue

        # --- 3. Document Q&A, if a document is loaded ---
        if doc_qa:
            answer, score, chunk_indices = doc_qa.answer(user_input)
            if answer:
                reply(answer, source="document_qa", confidence=float(score), chunk_indices=chunk_indices)
            else:
                reply(doc_qa.follow_up_question(), source="document_followup", confidence=float(score))
            continue

        # --- 4. Nothing matched, no document loaded ---
        reply(matcher.fallback_response(), source="fallback", confidence=0.0)


if __name__ == "__main__":
    main()