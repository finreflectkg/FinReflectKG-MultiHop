# FinReflectKG-MultiHop: Financial QA Benchmark for Reasoning with Knowledge Graph Evidence

A comprehensive benchmark for evaluating Large Language Models (LLMs) on multi-hop question answering over SEC 10-K financial filings. The benchmark is constructed using a financial knowledge graph and tests model performance across six context retrieval strategies with systematic LLM-as-Judge evaluation.

## Overview

Multi-hop question answering over financial documents requires models to synthesize information scattered across multiple pages, documents, companies, and fiscal years. This benchmark provides:

- **15,600 curated QA pairs** spanning 2-hop and 3-hop reasoning patterns over SEC 10-K filings
- **6 experimental conditions** testing different context retrieval and presentation strategies
- **8 LLM configurations** evaluated across all conditions
- **Multi-judge evaluation** using structured LLM-as-Judge scoring across four dimensions

## Repository Structure

```
FinReflectKG-MultiHop/
├── data/                  # Benchmark QA dataset (18 files, 15,600 questions)
├── experiments/           # Model outputs across 6 experimental conditions (864 files)
├── evaluations/           # LLM-as-Judge evaluation scores (1,199 files)
└── README.md
```

## Benchmark Dataset

### Question Design

Questions are generated from a financial knowledge graph constructed over SEC 10-K filings. Each question requires reasoning across multiple knowledge graph hops, with ground-truth answers derived from source document evidence.

**Question Types:**
- **Qualitative:** Reasoning about business relationships, strategic decisions, risk factors, and operational details
- **Quantitative:** Numerical extraction and computation involving financial metrics, percentages, and monetary values

**Reasoning Patterns:**

| Hop Count | Pattern | Description | Example Relation |
|-----------|---------|-------------|-----------------|
| 2-hop | `cross_company` | Two different companies linked by a shared entity | `ORG1 -[Invests_In]-> GPE <-[Operates_In]- ORG2` |
| 2-hop | `cross_year` | Same company across different fiscal years | `ORG(y1) -[Discloses]-> FIN_METRIC <-[Discloses]- ORG(y2)` |
| 2-hop | `intra_doc` | Same company, same year, different pages | `ORG(p1) -[Has_Stake_In]-> COMP <-[Depends_On]- ORG(p2)` |
| 3-hop | `cross_company_type1` | Connector chain across two companies | `ORG1 -> C1 -> C2 <- ORG2` |
| 3-hop | `cross_company_type2` | Three companies sharing one connector | `ORG1 -> C <- ORG2, C <- ORG3` |
| 3-hop | `cross_year_type1` | Connector chain across years | `ORG(y1) -> C1 -> C2 <- ORG(y2)` |
| 3-hop | `cross_year_type2` | Three years sharing one connector | `ORG(y1) -> C <- ORG(y2), C <- ORG(y3)` |
| 3-hop | `intra_doc_type1` | Connector chain within document | `ORG(p1) -> C1 -> C2 <- ORG(p2)` |
| 3-hop | `intra_doc_type2` | Three pages sharing one connector | `ORG(p1) -> C <- ORG(p2), C <- ORG(p3)` |

### Dataset Statistics

| Split | 2-hop | 3-hop | Total |
|-------|-------|-------|-------|
| Qualitative | 3,000 | 4,800 | 7,800 |
| Quantitative | 3,000 | 4,800 | 7,800 |
| **Total** | **6,000** | **9,600** | **15,600** |

### Sampling Methodology

Questions are sampled from a larger generated pool using a deterministic quality-based selection:
- Sort by `quality_score` descending, then `question_id` ascending for tie-breaking
- Select top-N per pattern according to predefined quotas
- Quality scores range from 46-50 (on a 50-point rubric covering factual grounding, multi-hop necessity, answer completeness, reasoning clarity, and difficulty calibration)

### Data Schema

Each data file contains a metadata header and a list of questions:

```json
{
  "metadata": {
    "mode": "qualitative",
    "hop": "2hop",
    "pattern": "cross_company",
    "count": 1250,
    "sampling_method": "top_quality_score",
    "quality_stats": { "min": 46, "max": 49, "mean": 46.78 }
  },
  "questions": [
    {
      "question_id": 85,
      "question": "...",
      "answer": "...",
      "reasoning_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
      "difficulty": "easy|medium|hard",
      "idf_score": 3.68,
      "sector": "Financials",
      "sub_industry": "Multi-line Insurance",
      "pattern": "ORG -[Invests_In]-> GPE <-[Operates_In]- ORG",
      "entities": { "start": "AIG", "intermediate": "New York", "end": "GS" },
      "hop_count": 2,
      "document_relationship": "inter_document_cross_company",
      "quality_score": 48,
      "path_data": { ... }
    }
  ]
}
```

## Experiments

Six experiments evaluate how different context retrieval strategies affect LLM performance on the benchmark. All experiments use the same 15,600 questions; only the context provided to the model changes.

### E1: KG Minimal (Baseline)

**Strategy:** Provide only the raw evidence chunks extracted along the knowledge graph path, plus the KG triplets.

- Minimal context: only hop-specific text chunks and entity relationships
- No surrounding pages or document-level context
- Tests baseline reasoning ability with perfect but minimal evidence

### E2: Page Window

**Strategy:** Provide fixed-size page windows around each target page.

- Adaptive windowing: 2 pages before + target page + 2 pages after (5 pages total per hop)
- Handles document boundaries by expanding the opposite side
- Pages presented in sequential order
- Tests whether surrounding document context improves over E1

### E3: Page Window + Distractors

**Strategy:** Same page windows as E2, plus irrelevant distractor pages, all shuffled.

- Adds distractor pages from the same document (selected from pages at least 10 pages away from any target)
- All pages (relevant + distractor) shuffled randomly
- Tests model robustness to noisy, unordered context

### E4: Page Window + Relevance (Descending)

**Strategy:** Same page windows as E2, ranked by semantic similarity to the question (most relevant first).

- Pages scored using embedding-based similarity to the question
- Sorted in descending order: most relevant content appears first
- Tests whether placing relevant information early improves performance

### E5: Page Window + Relevance (Ascending)

**Strategy:** Same page windows as E2, ranked by semantic similarity (least relevant first).

- Inverse of E4: most relevant content appears last
- Combined with E4, measures the "lost in the middle" effect
- Tests recency bias and positional sensitivity

### E6: Semantic Retrieval

**Strategy:** For each hop target page, retrieve top-K semantically similar pages from the same document, then shuffle.

- Retrieves pages that are semantically similar to each target page (top-5 per hop)
- Always includes the actual target page
- Retrieved pages shuffled randomly
- Tests whether models can distinguish between semantically similar but factually different content

### Models Evaluated

| Model | Parameters | Reasoning Mode |
|-------|-----------|----------------|
| GPT-OSS-120B | 120B | Non-reasoning |
| GPT-OSS-120B-Reasoning | 120B | Reasoning (Harmony API) |
| GPT-OSS-20B-Reasoning | 20B | Reasoning (Harmony API) |
| Qwen3-32B | 32B | Non-reasoning |
| Qwen3-32B-Reasoning | 32B | Reasoning |
| Qwen3-8B-Reasoning | 8B | Reasoning |
| Nemotron-Nano-30B | 30B | Non-reasoning |
| Nemotron-Nano-30B-Reasoning | 30B | Reasoning |

### Experiment Output Schema

```json
{
  "experiment_metadata": {
    "model": "qwen3-32b-reasoning",
    "experiment": "e1_kg_minimal",
    "pattern": "qualitative_2hop_cross_company",
    "total_questions": 1250
  },
  "results": [
    {
      "question_id": 85,
      "question": "...",
      "original_answer": "...",
      "llm_answer": "...",
      "llm_thought": "...",
      "hop_count": 2,
      "document_relationship": "inter_document_cross_company",
      "pattern": "ORG -[Invests_In]-> GPE <-[Operates_In]- ORG",
      "triplets_used": ["AIG (ORG) -> New York (GPE)", "..."],
      "chunks_used": [{ "hop": "hop_1_rel", "source_file": "AIG_10k_2023.pdf", "page_id": "page_92" }],
      "context_stats": { "total_chunks": 2, "total_chars": 5146 },
      "token_usage": { "prompt_tokens": 1924, "completion_tokens": 610, "total_tokens": 2534 }
    }
  ]
}
```

## Evaluation

### LLM-as-Judge Framework

Model outputs are evaluated using an LLM-as-Judge approach. Judge models receive the original question, ground-truth answer, and the model-generated answer, then produce structured scores across four dimensions.

### Scoring Dimensions

| Dimension | Scale | Description |
|-----------|-------|-------------|
| **Correctness** | 0-10 | Overall factual accuracy, calculation correctness, and multi-hop reasoning soundness |
| **Quantitative Accuracy** | 1-10 | Precision of numbers, dates, percentages, dollar amounts, and derived calculations |
| **Qualitative Accuracy** | 1-10 | Reasoning quality, entity identification, and multi-hop synthesis correctness |
| **Contextual Relevance** | 1-10 | Appropriateness of the response and coverage of all question components |


The evaluation framework applies strict assessment on numerical precision while remaining flexible on semantic equivalence in wording.

### Judge Models

| Judge Model | Role |
|-------------|------|
| Qwen3-235B-Reasoning | Primary judge |
| Claude Haiku 4.5 | Secondary judge |

### Evaluation Output Schema

```json
{
  "evaluation_metadata": {
    "evaluator_name": "LLM-as-Judge Evaluator (Model: qwen3-235b-reasoning)",
    "evaluator_model": "qwen3-235b-reasoning",
    "model_evaluated": "qwen3-32b-reasoning",
    "experiment": "e1_kg_minimal",
    "pattern": "qualitative_2hop_cross_company",
    "total_questions_in_experiment": 1250
  },
  "evaluation_results": [
    {
      "question_id": 85,
      "question": "...",
      "original_answer": "...",
      "llm_answer": "...",
      "correctness_score": 9,
      "quantitative_accuracy": 10,
      "qualitative_accuracy": 9,
      "contextual_relevance": 10,
      "detailed_feedback": "...",
      "hop_count": 2,
      "document_relationship": "inter_document_cross_company",
      "pattern": "ORG -[Invests_In]-> GPE <-[Operates_In]- ORG",
      "token_usage": { "prompt_tokens": 1136, "completion_tokens": 343, "total_tokens": 1479 }
    }
  ]
}
```

## File Inventory

| Component | Files | Size | Description |
|-----------|-------|------|-------------|
| `data/` | 18 | ~182 MB | Benchmark QA dataset (9 qualitative + 9 quantitative patterns) |
| `experiments/` | 864 | ~7.1 GB | Model outputs (8 models x 6 experiments x 18 patterns) |
| `evaluations/` | 1,199 | ~4.4 GB | Judge scores (multiple judges x models x experiments x patterns) |



## License

This dataset is released for research purposes. The underlying financial data is sourced from publicly available SEC 10-K filings.
