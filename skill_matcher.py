#!/usr/bin/env python
"""
Zero-Shot Resume ↔ JD Skill-Match
---------------------------------
Given a resume and a job-description (plain-text files),
ranks how strongly the resume *entails* each skill required by the JD.

Usage
-----
$ python skill_match.py path/to/resume.txt path/to/jd.txt --top 15

Outputs a CSV-style table to stdout and saves `results_<timestamp>.csv`.
"""

import argparse, csv, re, pathlib, time
from typing import List, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm
import spacy
from transformers import pipeline

# ─────────────────────────────── Helpers ────────────────────────────────

STOP_SKILL_WORDS = {
    "the", "and", "with", "using", "use", "knowledge", "experience",
    "skills", "skill", "ability", "develop", "development"
}

def extract_skill_phrases(text: str, nlp) -> List[str]:
    """
    Very lightweight skill-phrase extractor:
    • Returns noun-chunks (1-3 tokens, alphabetical) minus common stop terms.
    """
    doc = nlp(text.lower())
    phrases = set()
    for chunk in doc.noun_chunks:
        tok_text = chunk.text.strip()
        # filter: 1-3 words, alphabetic, no stop filler
        if 1 <= len(tok_text.split()) <= 3 and re.match(r"^[a-zA-Z ]+$", tok_text):
            if not any(w in STOP_SKILL_WORDS for w in tok_text.split()):
                phrases.add(tok_text)
    # crude duplicates clean-up (e.g. "python" vs "python programming")
    phrases = sorted(phrases, key=len)
    deduped = []
    for p in phrases:
        if not any(p in longer for longer in deduped):
            deduped.append(p)
    return deduped

def zero_shot_scores(resume: str, skills: List[str], pipe) -> List[Tuple[str, float]]:
    """
    Runs HF zero-shot classification once per skill (fast on CPU).
    Hypothesis template: “This person is skilled in <skill>.”
    Returns list of (skill, entailment_score).
    """
    results = []
    template = "This person is skilled in {}."
    for skill in tqdm(skills, desc="Scoring"):
        hypothesis = template.format(skill)
        res = pipe(resume, hypothesis, entailment=True)
        # pipeline returns {'labels': [...], 'scores': [...], ...}
        entail_score = res["scores"][0]
        results.append((skill, round(entail_score, 3)))
    return sorted(results, key=lambda x: x[1], reverse=True)

# ──────────────────────────────── Main ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("resume", type=pathlib.Path, help="Plain-text resume file")
    parser.add_argument("jd", type=pathlib.Path, help="Plain-text job-description file")
    parser.add_argument("--top", type=int, default=20, help="Top-N skills to print")
    args = parser.parse_args()

    resume_text = args.resume.read_text(encoding="utf-8")
    jd_text = args.jd.read_text(encoding="utf-8")

    print("xtracting candidate skills from JD …")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    skills = extract_skill_phrases(jd_text, nlp)
    print(f"   → {len(skills)} unique skill phrases found.")

    print("\Running zero-shot entailment …")
    zshot = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli",
                     device_map="auto" if torch.cuda.is_available() else None)

    scored = zero_shot_scores(resume_text, skills, zshot)

    # Save CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"results_{ts}.csv"
    pd.DataFrame(scored, columns=["skill", "score"]).to_csv(out_path, index=False)
    print(f"\nSaved full results → {out_path}")

    # Print top-N nicely
    print("\nTop matches")
    print("-" * 30)
    for s, sc in scored[:args.top]:
        print(f"{s:<25} {sc:.3f}")

if __name__ == "__main__":
    main()
