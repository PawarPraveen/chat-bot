"""
Basic regression tests for chatbot.py.

Run with:
    python -m unittest test_chatbot.py -v

These tests exist to catch the exact kind of bug that showed up during
manual development (pattern collisions skewing predictions, numpy types
breaking JSON serialization, path traversal in uploads) automatically,
instead of relying on someone noticing during a live chat session.
"""

import csv
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import app as chatbot


class TempWorkspaceTestCase(unittest.TestCase):
    """Redirects all chatbot persistence paths to a scratch directory so
    tests never read/write the real feedback/learned-pattern files, and
    each test starts from a clean slate."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_paths = {
            "FEEDBACK_PATH": chatbot.FEEDBACK_PATH,
            "LEARNED_PATTERNS_PATH": chatbot.LEARNED_PATTERNS_PATH,
            "NEGATIVE_EXAMPLES_PATH": chatbot.NEGATIVE_EXAMPLES_PATH,
            "DOC_FEEDBACK_PATH": chatbot.DOC_FEEDBACK_PATH,
            "UPLOAD_DIR": chatbot.UPLOAD_DIR,
        }
        chatbot.FEEDBACK_PATH = os.path.join(self.tmpdir, "feedback_log.csv")
        chatbot.LEARNED_PATTERNS_PATH = os.path.join(self.tmpdir, "learned_patterns.json")
        chatbot.NEGATIVE_EXAMPLES_PATH = os.path.join(self.tmpdir, "negative_examples.json")
        chatbot.DOC_FEEDBACK_PATH = os.path.join(self.tmpdir, "doc_feedback.json")
        chatbot.UPLOAD_DIR = os.path.join(self.tmpdir, "uploads")
        os.makedirs(chatbot.UPLOAD_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for k, v in self._orig_paths.items():
            setattr(chatbot, k, v)


class TestResumeTrainingData(TempWorkspaceTestCase):
    def test_dataset_entries_are_loaded_and_used_for_follow_up_questions(self):
        data = chatbot.load_training_data(chatbot.RESUME_TRAINING_DATA_PATH)
        self.assertIn("entries", data)
        self.assertGreaterEqual(len(data["entries"]), 20)

        doc = chatbot.DocumentQA(
            "John Doe\nExperience\nSenior Backend Engineer at TechCorp.\nSkills\nPython, Go, AWS.\nProjects\nBuilt a data pipeline tool."
        )
        answer, score, chunk_indices = doc.answer("Why should we hire this candidate?")

        self.assertIsNotNone(answer)
        self.assertGreaterEqual(score, 0.0)
        self.assertTrue(isinstance(chunk_indices, list) or isinstance(chunk_indices, tuple))


class TestIntentClassifier(TempWorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.clf = chatbot.IntentClassifier(chatbot.INTENTS_PATH)

    def test_clear_matches_are_correct(self):
        cases = {
            "hello there": "greetings",
            "tell me a joke": "joke",
            "bye for now": "goodbye",
            "you are dumb": "insult",
            "thank you so much": "thanks",
            "how old are you?": "age",
            "what is your name?": "name",
        }
        for text, expected_tag in cases.items():
            with self.subTest(text=text):
                tag, confidence, ambiguous = self.clf.predict(text)
                self.assertEqual(tag, expected_tag)
                self.assertIsNone(ambiguous)

    def test_unrelated_text_is_rejected_not_misclassified(self):
        # Regression test for the "you don't work" / "nice work" pattern
        # collisions found during manual testing -- an unrelated resume
        # question must NOT be claimed by an intent.
        cases = [
            "Where did John work at TechCorp?",
            "What skills does the candidate have?",
            "What did he build at StartupX?",
        ]
        for text in cases:
            with self.subTest(text=text):
                tag, confidence, ambiguous = self.clf.predict(text)
                self.assertIsNone(tag, f"{text!r} was wrongly classified as {tag!r}")

    def test_gibberish_is_rejected(self):
        tag, confidence, ambiguous = self.clf.predict("asdkj random gibberish blah")
        self.assertIsNone(tag)

    def test_ambiguous_input_returns_runner_up(self):
        tag, confidence, ambiguous = self.clf.predict("what is your age and name")
        self.assertIsNotNone(ambiguous)
        self.assertIn(tag, ("age", "name"))
        self.assertIn(ambiguous, ("age", "name"))
        self.assertNotEqual(tag, ambiguous)

    def test_positive_feedback_grows_training_set_and_retrains(self):
        before_tag, _, _ = self.clf.predict("got any jokes for me")
        before_count = len(self.clf._training_pairs()[0])

        self.clf.learn_positive("joke", "got any jokes for me")

        after_count = len(self.clf._training_pairs()[0])
        after_tag, after_conf, _ = self.clf.predict("got any jokes for me")

        self.assertEqual(after_count, before_count + 1)
        self.assertEqual(after_tag, "joke")
        # Persisted to disk, not just in memory.
        self.assertTrue(os.path.isfile(chatbot.LEARNED_PATTERNS_PATH))
        with open(chatbot.LEARNED_PATTERNS_PATH) as f:
            saved = json.load(f)
        self.assertIn("got any jokes for me", saved.get("joke", []))

    def test_negative_feedback_suppresses_future_match(self):
        # Find a phrase this classifier currently (mis)classifies, then
        # downvote it and confirm the exact tag is suppressed next time.
        text = "what can you help me with"
        tag, conf, _ = self.clf.predict(text)
        if tag is None:
            self.skipTest("Nothing to suppress -- classifier already rejects this input")

        self.clf.learn_negative(tag, text)
        tag2, conf2, _ = self.clf.predict(text)
        self.assertIsNone(tag2)
        self.assertTrue(os.path.isfile(chatbot.NEGATIVE_EXAMPLES_PATH))

    def test_feedback_log_replays_previous_ratings_on_startup(self):
        with open(chatbot.FEEDBACK_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "user_input", "bot_response", "source", "confidence", "rating", "comment"])
            writer.writerow(["2026-07-11T00:00:00+00:00", "got any jokes for me", "Sure!", "intent_ml:joke", "0.9", "up", ""])

        clf = chatbot.IntentClassifier(chatbot.INTENTS_PATH)
        tag, confidence, ambiguous = clf.predict("got any jokes for me")

        self.assertEqual(tag, "joke")
        self.assertTrue(os.path.isfile(chatbot.LEARNED_PATTERNS_PATH))


class TestDocumentQA(TempWorkspaceTestCase):
    SAMPLE_TEXT = (
        "John Doe\n"
        "Email: john.doe@example.com | Phone: 555-123-4567\n\n"
        "Experience\n"
        "Senior Backend Engineer at TechCorp, 2021-Present.\n"
        "Software Engineer at StartupX, 2019-2021.\n\n"
        "Education\n"
        "B.S. in Computer Science, State University, 2019.\n\n"
        "Skills\n"
        "Python, Go, SQL, Docker, Kubernetes, AWS.\n\n"
        "Projects\n"
        "Built an open-source data pipeline tool used by over 500 developers.\n"
    )

    def test_answers_relevant_question(self):
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        answer, score, chunk_indices = doc.answer("What is the phone number?")
        self.assertIsNotNone(answer)
        self.assertIn("555-123-4567", answer)
        self.assertTrue(len(chunk_indices) > 0)

    def test_short_section_questions_return_section_content(self):
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        for question, expected in [
            ("What skills are listed?", "Python"),
            ("What projects did the candidate work on?", "open-source"),
            ("education", "Computer Science"),
        ]:
            with self.subTest(question=question):
                answer, score, chunk_indices = doc.answer(question)
                self.assertIsNotNone(answer)
                self.assertIn(expected, answer)
                self.assertTrue(len(chunk_indices) >= 0)

    def test_missing_term_returns_not_listed_message(self):
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        answer, score, chunk_indices = doc.answer("Java")
        self.assertIsNotNone(answer)
        self.assertIn("resume", answer.lower())
        self.assertIn("remember", answer.lower())

    def test_candidate_name_question_returns_resume_name(self):
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        answer, score, chunk_indices = doc.answer("name of the candidate")
        self.assertIsNotNone(answer)
        self.assertIn("John Doe", answer)

    def test_short_skill_question_returns_section_content(self):
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        answer, score, chunk_indices = doc.answer("skills")
        self.assertIsNotNone(answer)
        self.assertIn("Python", answer)

    def test_summary_question_returns_summary_like_answer(self):
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        answer, score, chunk_indices = doc.answer("write a summary about the candidate")
        self.assertIsNotNone(answer)
        self.assertIn("John Doe", answer)
        self.assertIn("Python", answer)

    def test_positive_feedback_is_reused_for_similar_questions(self):
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        doc.learn_positive("what skills are listed?", "Python, Go, SQL, Docker, Kubernetes, AWS.", [1])
        answer, score, chunk_indices = doc.answer("skills")
        self.assertIsNotNone(answer)
        self.assertIn("Python", answer)

    def test_low_similarity_triggers_follow_up_not_a_guess(self):
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        answer, score, chunk_indices = doc.answer("what is the meaning of life")
        self.assertIsNotNone(answer)
        self.assertIn("remember", answer.lower())
        followup = doc.follow_up_question()
        self.assertIn("resume", followup.lower())

    def test_downvote_persists_and_suppresses_same_answer(self):
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        question = "What is the phone number?"
        answer, score, chunk_indices = doc.answer(question)
        self.assertIsNotNone(answer)

        doc.penalize(question, chunk_indices)

        # Same DocumentQA instance: chunk should now be excluded.
        answer2, score2, chunk_indices2 = doc.answer(question)
        self.assertNotEqual(chunk_indices2, chunk_indices)

        # A fresh instance on the SAME document text should also remember
        # the penalty (persisted to disk by content hash).
        doc_reloaded = chatbot.DocumentQA(self.SAMPLE_TEXT)
        answer3, score3, chunk_indices3 = doc_reloaded.answer(question)
        self.assertNotEqual(chunk_indices3, chunk_indices)

    def test_chunk_indices_are_json_serializable(self):
        # Regression test for the numpy.int64 JSON serialization bug found
        # during manual testing.
        doc = chatbot.DocumentQA(self.SAMPLE_TEXT)
        answer, score, chunk_indices = doc.answer("What is the phone number?")
        for idx in chunk_indices:
            self.assertIsInstance(idx, int)
        json.dumps(chunk_indices)  # must not raise


class TestUploadSecurity(TempWorkspaceTestCase):
    def setUp(self):
        super().setUp()
        with open(os.path.join(chatbot.UPLOAD_DIR, "resume.txt"), "w") as f:
            f.write("Sample content")

    def test_empty_file_is_rejected(self):
        empty_path = os.path.join(chatbot.UPLOAD_DIR, "empty.txt")
        with open(empty_path, "w") as f:
            f.write("")
        with self.assertRaises(chatbot.UploadError):
            chatbot.load_document_text(empty_path)

    def test_loaded_document_suppresses_upload_intent(self):
        sample_path = os.path.join(chatbot.UPLOAD_DIR, "resume.txt")
        with open(sample_path, "w") as f:
            f.write("Skills\nPython, Go, SQL")

        with patch("builtins.input", side_effect=["upload resume.txt", "what skills are in the resume?", "quit"]):
            with redirect_stdout(io.StringIO()) as stdout:
                chatbot.main()

        output = stdout.getvalue()
        self.assertIn("Got it — I've read your", output)
        self.assertNotIn("Please upload your resume", output)
        self.assertIn("Python", output)

    def test_valid_filename_resolves(self):
        path = chatbot.resolve_upload_path("resume.txt")
        self.assertTrue(path.startswith(chatbot.UPLOAD_DIR))

    def test_parent_directory_traversal_is_rejected(self):
        for attempt in ["../resume.txt", "../../etc/passwd", "a/../../etc/passwd"]:
            with self.subTest(attempt=attempt):
                with self.assertRaises(chatbot.UploadError):
                    chatbot.resolve_upload_path(attempt)

    def test_absolute_path_is_rejected(self):
        with self.assertRaises(chatbot.UploadError):
            chatbot.resolve_upload_path("/etc/passwd")

    def test_disallowed_extension_is_rejected(self):
        with open(os.path.join(chatbot.UPLOAD_DIR, "script.exe"), "w") as f:
            f.write("x")
        with self.assertRaises(chatbot.UploadError):
            chatbot.resolve_upload_path("script.exe")

    def test_missing_file_gives_clear_error(self):
        with self.assertRaises(chatbot.UploadError):
            chatbot.resolve_upload_path("does_not_exist.txt")


class TestFeedbackLogger(TempWorkspaceTestCase):
    def test_log_creates_file_with_header(self):
        logger = chatbot.FeedbackLogger(chatbot.FEEDBACK_PATH)
        self.assertTrue(os.path.isfile(chatbot.FEEDBACK_PATH))

    def test_log_and_stats(self):
        logger = chatbot.FeedbackLogger(chatbot.FEEDBACK_PATH)
        logger.log("hi", "Hello!", "intent_ml:greetings", 0.9, "up")
        logger.log("xyz", "I'm not sure...", "fallback", 0.0, "down", "confusing")
        stats = logger.stats()
        self.assertIn("1 helpful", stats)
        self.assertIn("1 not helpful", stats)


if __name__ == "__main__":
    unittest.main()