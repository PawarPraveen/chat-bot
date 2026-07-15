# Chatbot: Trained Intents + Resume/Document Q&A

A command-line chatbot with two capabilities:

1. **Small talk / FAQ answers** via a trained classifier (TF-IDF + Multinomial
   Naive Bayes) with a rule-based safety net, defined in `intents.json`.
2. **Document Q&A** — upload a resume or other document and ask questions
   about it (TF-IDF + cosine similarity retrieval over the document's text).

Both layers **learn from human feedback**: upvoting (`y`) or downvoting (`n`)
each answer actually changes future behavior, not just logs it.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
python chatbot.py
```

## Usage

```
You: hello
Chatbot: Hello!
You: y

You: upload sample_resume.txt
Chatbot: Got it — I've read your resume (16 sections parsed). Ask me anything about it!

You: What is the candidate's phone number?
Chatbot: Email: john.doe@example.com | Phone: 555-123-4567
You: y
```

**Commands:**
| Command | Effect |
|---|---|
| `upload <filename>` | Load a `.txt`, `.pdf`, or `.docx` file from the `uploads/` folder |
| `y` / `n` (after an answer) | Rate the last answer helpful / not helpful |
| `n <comment>` or `/feedback <comment>` | Rate not-helpful with a note |
| `feedback stats` | Show overall helpful/not-helpful counts |
| `model stats` | Show the classifier's training set size per intent |
| `quit` / `exit` | End the session |

**Uploading files:** for security, files must be placed in the `uploads/`
folder next to `chatbot.py` — the bot will not read files from arbitrary
paths on disk (see [Security](#security) below).

## Architecture

Every message is tried against four layers, in order, until one accepts it:

1. **Trained ML classifier** (`IntentClassifier`) — TF-IDF features +
   `MultinomialNB`, trained on the patterns in `intents.json`. This is the
   layer that actually learns from feedback.
2. **Rule-based keyword matcher** (`IntentMatcher`) — a static, deterministic
   safety net for phrasing the small trained model misses. Unaffected by
   feedback, so it can't drift or be broken by bad feedback.
3. **Document Q&A** (`DocumentQA`) — only tried if a document is loaded and
   neither intent layer matched. Also learns from feedback (see below).
4. **Fallback response** — if nothing above matched.

### How feedback changes behavior

| Layer | Upvote (`y`) | Downvote (`n`) |
|---|---|---|
| ML classifier | Phrase saved to `learned_patterns.json`, model **retrained** | Phrase saved to `negative_examples.json`; future similar messages have that tag suppressed |
| Rule matcher | Not affected (static safety net by design) | Not affected |
| Document Q&A | Not affected (no penalty needed) | The specific passage used is recorded (by document content hash) in `doc_feedback.json` and excluded for similar future questions **on that same document**, even across sessions |

### Files this generates
- `feedback_log.csv` — full audit log of every rating (for manual review)
- `learned_patterns.json` — phrases added to intents by upvotes
- `negative_examples.json` — phrases suppressed by downvotes
- `doc_feedback.json` — per-document penalized passages, keyed by content hash

Delete any of these to reset that piece of learned state; delete all four to
return the bot to its out-of-the-box behavior.

## Security

`upload <filename>` resolves the given name **only** within the `uploads/`
folder:
- Absolute paths and `..` traversal are rejected outright.
- The resolved real path must stay inside `uploads/` (symlink-safe check).
- Only `.txt`, `.pdf`, `.docx` extensions are accepted.

This prevents the chat command from being used to read arbitrary files
elsewhere on the filesystem.

## Testing

```bash
python -m unittest app.py -v
```

Covers: classifier accuracy on known phrasings, rejection of unrelated/
gibberish input, the ambiguous-intent clarifying-question path, feedback-
driven learning and suppression (both intents and document Q&A), JSON
serialization of numpy types, and all upload-security edge cases. Tests run
against a temporary scratch directory, so they never touch your real
`feedback_log.csv` / `learned_patterns.json` / etc.

## Known limitations

This is a small-scale demo, not a production system. Worth knowing:

- **Tiny training set.** `intents.json` has ~5–10 example patterns per
  intent. `MultinomialNB`'s confidence values are consequently low in
  absolute terms (0.10–0.25 is normal, not a bug) — thresholds were tuned
  empirically for this dataset size, not derived from theory.
- **No stemming/lemmatization.** "job" and "jobs" are unrelated tokens to
  the model. Fine at this scale; would need addressing if `intents.json`
  grows much larger.
- **`learned_patterns.json` / `negative_examples.json` grow unboundedly.**
  Dedup is exact-string-match only — no pruning or similarity-based merging
  over time.
- **Document Q&A is per-document**, keyed by a hash of its full text — a
  single-character edit to the source document is treated as a different
  document and loses accumulated feedback for it.
- **CLI only.** No web UI, no session persistence for the uploaded document
  itself (only the feedback about it persists).
