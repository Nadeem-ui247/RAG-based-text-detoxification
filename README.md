# DetoxLLM-RAG: Explainable Toxic Comment Detoxification using Retrieval-Augmented Generation

A text detoxification pipeline that combines **Chain-of-Thought (CoT) reasoning** with **Retrieval-Augmented Generation (RAG)** to rewrite toxic comments into non-toxic alternatives while preserving the original meaning.

Built on top of [DetoxLLM-7B](https://huggingface.co/UBC-NLP/DetoxLLM-7B) (Khondaker et al., 2024), this project extends the base model with a three-step inference pipeline: rationale generation → example retrieval → grounded rewrite.

---

## Overview

Although DetoxLLM produces high-quality detoxified text, its generation process is not explicitly grounded in external examples. This project augments DetoxLLM with Retrieval-Augmented Generation (RAG), enabling retrieval-guided detoxification while improving interpretability through rationale generation

## Contributions 

This work extends DetoxLLM-7B by introducing:

- Chain-of-Thought rationale generation
- Retrieval-Augmented Generation (RAG)
- Semantic reranking of retrieved examples
- Similarity thresholding
- Comparative evaluation against the baseline DetoxLLM
  
## Methodology

1. **Generating a rationale** — the model first explains *why* the comment is toxic (CoT)
2. **Retrieving relevant examples** — the rationale is used as a semantic query to fetch non-toxic examples from a FAISS index built on ParaDetox (RAG)
3. **Generating a grounded rewrite** — the final rewrite is conditioned on the original comment, the rationale, and the retrieved examples

This forces the model to reason explicitly before rewriting, and grounds the output in verified non-toxic language.

---

## Pipeline Architecture

```
Toxic Comment
      │
      ▼
┌─────────────────────┐
│  Rationale Generation│  ← DetoxLLM-7B (CoT prompt)
│  "Why is this toxic?"│
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Example Retrieval  │  ← MiniLM embeddings + FAISS index
│  (RAG via FAISS)    │     Query = toxic comment + rationale
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Grounded Rewrite   │  ← DetoxLLM-7B (rewrite prompt)
│  Generation         │     Context = rationale + retrieved examples
└─────────────────────┘
      │
      ▼
 Detoxified Output
```

---

## Features

- **CoT-guided reasoning** — explicit rationale generated before rewriting, improving interpretability
- **RAG grounding** — rewrites anchored in verified non-toxic examples from ParaDetox
- **Reranking** — retrieval candidates reranked by semantic similarity to the original comment for better topical relevance
- **4-bit quantization** — runs on a single consumer GPU via QLoRA / BitsAndBytes
- **Safety gate** — cosine similarity check flags rewrites with excessive semantic drift
- **Interactive loop** — test any comment interactively in Colab

---

## Dataset

**[ParaDetox](https://huggingface.co/datasets/s-nlp/paradetox)** (Logacheva et al., ACL 2022)

A parallel detoxification corpus of ~12,000 toxic sentences paired with human-written non-toxic paraphrases. Used for:
- Building the FAISS retrieval index (non-toxic side, 5,000 sampled sentences)
- Evaluation against human reference rewrites (100 sampled sentences)

---

## Model

**[UBC-NLP/DetoxLLM-7B](https://huggingface.co/UBC-NLP/DetoxLLM-7B)** (Khondaker et al., 2024)

A LLaMA-2 based 7B parameter model pre-trained on a cross-platform detoxification corpus with explanation generation capability. Loaded in 4-bit NF4 quantization via BitsAndBytes.

> **Note:** DetoxLLM-7B is a gated model. You must accept the terms on its Hugging Face page and authenticate via `huggingface_hub.login()` before use.

---

## Retrieval: FAISS + MiniLM

| Component | Detail |
|-----------|--------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Index type | `faiss.IndexFlatIP` (inner product = cosine on normalized vectors) |
| Retrieval corpus | ParaDetox non-toxic sentences (2,000 sampled) |
| Query | Concatenation of toxic comment + generated rationale |
| Retrieval k | 10 candidates |
| Reranking | Top 10 reranked by cosine similarity to original toxic comment |
| Final returned | Top 3 above similarity threshold of 0.25 |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU** | N-gram overlap between rewrite and human reference |
| **BERT F1** | Semantic similarity to human reference via RoBERTa embeddings |
| **SIM** | Cosine similarity between SBERT embeddings of input and rewrite — measures meaning preservation |
| **STA (%)** | Percentage of rewrites classified as non-toxic by Detoxify |
| **Toxicity Delta** | Average reduction in Detoxify toxicity score (input − rewrite) |

---

## Results

Evaluated on 100 randomly sampled ParaDetox sentences (seed = 42).

| Metric | DetoxLLM only | DetoxLLM + RAG | Human Reference |
|--------|:-------------:|:--------------:|:---------------:|
| BLEU | 0.0739 | 0.0595 | 0.9800 |
| BERT F1 | 0.9145 | 0.8950 | 1.0000 |
| SIM | 0.6517 | 0.5530 | 0.8072 |
| STA (%) | 97.00 | **100.00** | 95.00 |
| Toxicity Delta | 0.8674 | **0.8841** | 0.8178 |

**Key finding:** DetoxLLM + RAG achieves stronger detoxification effectiveness (STA 100%, higher Toxicity Delta) while DetoxLLM alone better preserves semantic content (higher SIM and BERT F1). This reflects the classic detoxification tradeoff between toxicity removal and meaning preservation.

---

## Installation

```bash
pip install transformers accelerate bitsandbytes sentence-transformers \
            faiss-cpu datasets evaluate bert-score detoxify
```

Authenticate with Hugging Face:

```python
from huggingface_hub import login
login()  # paste your HF token when prompted
```

---

## Usage

Run all cells in order in Google Colab (T4 GPU recommended):

| Cell | Description |
|------|-------------|
| Cell 1 | Install dependencies + HF login |
| Cell 2 | Load ParaDetox, build FAISS index |
| Cell 3 | Load DetoxLLM-7B in 4-bit quantization |
| Cell 4 | Define pipeline functions |
| Cell 5 | Interactive detoxification loop |
| Cell 6 | Evaluation on 100 ParaDetox samples |

**Interactive testing:**

```
Enter toxic comment (or 'exit'): Shut the fuck up. Nobody cares about your opinion.

Rationale:
The comment contains profanity ("fuck") and a personal attack that dismisses the other person's opinion in a hostile manner.

Retrieved Examples:
  1. I'd appreciate it if we could keep the discussion respectful.
  2. Everyone is entitled to express their opinion.
  3. Let's keep the conversation constructive.

Detoxified Rewrite:
I disagree with your opinion, but let's keep the discussion respectful.
```

---

## Future Work

- Extend retrieval corpus beyond 2,000 samples for better topical coverage
- Explore iterative RAG (multi-hop retrieval) for complex toxic comments
- Fine-tune with rationale supervision on the full ParaDetox + ToxiGen corpus
- Multilingual detoxification support
- Replace cosine similarity safety gate with a learned semantic drift classifier

---

## References

- Khondaker et al. (2024). *DetoxLLM: A Framework for Detoxification with Explanations.* arXiv:2402.15951
- Logacheva et al. (2022). *ParaDetox: Detoxification with Parallel Data.* ACL 2022
- Hartvigsen et al. (2022). *ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection.* arXiv:2203.09509
- Rehulka & Suppa (2024). *RAG Meets Detox: Enhancing Text Detoxification using Open LLMs with RAG.* CLEF 2024

---

## License

This project is for academic and research purposes. Model usage is subject to the [DetoxLLM-7B](https://huggingface.co/UBC-NLP/DetoxLLM-7B) and [LLaMA-2](https://ai.meta.com/llama/) license terms.
