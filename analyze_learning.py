import csv
import json
import os
import app

print('=== Document QA learning benchmark ===')
resumes = [
    ('resume_alpha.txt', 'What skills are listed?'),
    ('resume_beta.txt', 'What skills are listed?'),
    ('resume_gamma.txt', 'What skills are listed?'),
]

for filename, question in resumes:
    path = os.path.join(app.UPLOAD_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    doc = app.DocumentQA(text)
    answer, score, chunks = doc.answer(question)
    print(f'[{filename}] initial answer -> {answer[:120] if answer else None}')
    if answer:
        doc.learn_positive(question, answer, chunks)
    reloaded = app.DocumentQA(text)
    followup_answer, followup_score, followup_chunks = reloaded.answer('skills')
    print(f'[{filename}] follow-up answer -> {followup_answer[:120] if followup_answer else None}')
    print('---')

print('=== Intent feedback replay benchmark ===')
with open(app.FEEDBACK_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp','user_input','bot_response','source','confidence','rating','comment'])
    writer.writerow(['2026-07-11T00:00:00+00:00','got any jokes for me','Sure!','intent_ml:joke','0.9','up',''])

clf = app.IntentClassifier(app.INTENTS_PATH)
tag, confidence, ambiguous = clf.predict('got any jokes for me')
print('predicted tag:', tag)
print('confidence:', confidence)
print('learned phrases for joke:', clf.learned.get('joke'))
print('learned_patterns.json exists:', os.path.isfile(app.LEARNED_PATTERNS_PATH))
print('negative_examples.json exists:', os.path.isfile(app.NEGATIVE_EXAMPLES_PATH))
