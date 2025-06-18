# Zero-Shot Resume JD Skill-Matcher

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![HF Transformers](https://img.shields.io/badge/🤗_Transformers-4.40+-ff66cc)


Tiny utility (~150 LOC) that **ranks how well a résumé covers the skills
listed in a job-description** – no training, just instant zero-shot inference.

---

## Why?

Recruiters and candidates both waste hours eyeballing overlaps between
résumés and JDs. Modern NLI (Natural-Language Inference) models already
understand *“sentence A entails sentence B”*.  
We abuse that super-power:

> *Hypothesis* → “This person is skilled in `<skill>`.”

If the model says “entailment = 0.93”, the skill is probably present.
That’s it—fast, explainable, and fun.

---

## How it works

1. **spaCy** w/ `en_core_web_sm` extracts concise noun-phrases
   (candidate skills) from the JD.
2. For each phrase we pose the hypothesis above to
   **facebook/bart-large-mnli** via the 🤗 *zero-shot* pipeline.
3. We collect entailment scores, sort, print the top-N, and save everything
   to `results_<timestamp>.csv`.

---

## 🏃‍♂️ Quick start

```
git clone https://github.com/<your-handle>/resume-jd-skill-match.git
cd resume-jd-skill-match

# create & activate a virtual-env (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# install deps
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```


![License](https://img.shields.io/badge/License-MIT-green)
