# Multi-Hop QA Generation Pipeline

A flexible pipeline for generating multi-hop question-answer pairs from SEC 10-K filings stored in a Neo4j knowledge graph.

## Overview

This pipeline generates **true multi-hop questions** that require information from multiple document chunks to answer. Questions are Cypher-retrievable, meaning the reasoning path can be traced through the knowledge graph.

### Hop Patterns Supported

#### 2-Hop Patterns
```
cross_company:   ORG1 ──[rel1]──> Connector <──[rel2]── ORG2
                 (Same year, different companies)

cross_year:      ORG(year1) ──[rel1]──> Connector <──[rel2]── ORG(year2)
                 (Same company, different years)

intra_doc:       ORG ──[rel1]──> Connector <──[rel2]── ORG
                 (Same company, same year, different pages/chunks)
```

#### 3-Hop Patterns

**Type 1: Connector Chain** (2 connectors linked together)
```
cross_company_type1:  ORG1 ──> C1 ──> C2 <── ORG2
                      (Different companies, same year)

cross_year_type1:     ORG(y1) ──> C1 ──> C2 <── ORG(y2)
                      (Same company, different years)

intra_doc_type1:      ORG(p1) ──> C1 ──> C2 <── ORG(p2)
                      (Same company, same year, different pages)
```

**Type 2: Multi-Branch** (3 anchors → 1 connector)
```
cross_company_type2:  ORG1 ──> Connector <── ORG2
                              Connector <── ORG3
                      (3 different companies, same year)

cross_year_type2:     ORG(y1) ──> Connector <── ORG(y2)
                                  Connector <── ORG(y3)
                      (Same company, 3 different years)

intra_doc_type2:      ORG(p1) ──> Connector <── ORG(p2)
                                  Connector <── ORG(p3)
                      (Same company, same year, 3 different pages)
```

**3-Hop Pattern Summary Table**
| Pattern | Type | Category | ORGs | Years | Pages | Structure |
|---------|------|----------|------|-------|-------|-----------|
| `cross_company_type1` | 1 | Cross Company | 2 diff | Same | N/A | `O1─C1─C2─O2` |
| `cross_year_type1` | 1 | Cross Year | Same | 2 diff | N/A | `O(y1)─C1─C2─O(y2)` |
| `intra_doc_type1` | 1 | Intra Doc | Same | Same | 2 diff | `O(p1)─C1─C2─O(p2)` |
| `cross_company_type2` | 2 | Cross Company | 3 diff | Same | N/A | `O1→C←O2, C←O3` |
| `cross_year_type2` | 2 | Cross Year | Same | 3 diff | N/A | `O(y1)→C←O(y2), C←O(y3)` |
| `intra_doc_type2` | 2 | Intra Doc | Same | Same | 3 diff | `O(p1)→C←O(p2), C←O(p3)` |

#### 3-Hop Causal Patterns (Cross-Company, Same Year)
These patterns are enabled with `--3-hop-method causal` and are built from a causal link
`x1 → x2` plus anchors from each company:
```
Pattern A (both_orgs_originator): ORG1 ──> x1 <── ORG2, and x1 ──> x2
Pattern B (both_orgs_recipient):  ORG1 ──> x2 <── ORG2, and x1 ──> x2
Pattern C (both_orgs_separate):   ORG1 ──> x1 ──> x2 <── ORG2
```
All causal patterns enforce cross-company evidence: the two ORG anchoring
relationships must come from different source files (different companies).
Note: causal 3-hop discovery does **not** use IDF filtering or scoring.

---

## Project Structure

```
dataset_generation_new/
├── config.yaml              # Central configuration file
├── prompts.yaml             # LLM prompts with context engineering
├── pipeline.py              # Main orchestrator (entry point)
├── pair_generator.py        # Company pair generation
├── connector_discovery.py   # Neo4j connector & path discovery
├── qa_generator.py          # QA generation with LLM
├── evidence_retrieval.py    # Chunk text retrieval
├── llm.py                   # OpenAI-compatible LLM client
├── gics_loader.py           # GICS sub-industry data
└── outputs/                 # Generated QA files
```

---

## Prerequisites

1. **Neo4j Database** running with 10-K data loaded
2. **Python 3.10+**
3. **LLM endpoint** (OpenAI-compatible API)

---

## Installation

```bash
# From project root
cd src/dataset_generation_new

# Install dependencies (if not already installed)
pip install neo4j openai pyyaml
```

---

## Configuration

All settings are in `config.yaml`:

### 1. Neo4j Connection
```yaml
neo4j:
  uri: "<YOUR_NEO4J_URI>"
  username: "neo4j"
  password: "your_password"
  database: "neo4j"
```

### 2. Hop Settings
```yaml
hop_settings:
  enabled_hops:
    - 2
    - 3

  two_hop:
    enabled: true
    patterns:
      - name: "cross_company"
        enabled: true
      - name: "cross_year"
        enabled: true
      - name: "intra_doc"
        enabled: true

  three_hop:
    enabled: true
    patterns:
      # Type 1: Connector Chain (O─C1─C2─O)
      - name: "cross_company_type1"
        enabled: true
      - name: "cross_year_type1"
        enabled: true
      - name: "intra_doc_type1"
        enabled: true
      # Type 2: Multi-Branch (3 anchors → 1 connector)
      - name: "cross_company_type2"
        enabled: true
      - name: "cross_year_type2"
        enabled: true
      - name: "intra_doc_type2"
        enabled: true
```

### 3. LLM Settings
```yaml
llm:
  default_model: "qwen3-235b"

  models:
    qwen3-235b:
      base_url: "http://your-llm-endpoint/v1"
      model: "Qwen/Qwen3-235B-A22B"
      api_key: "EMPTY"
      max_tokens: 40000
      temperature: 0.3
```

### 4. IDF Thresholds (Difficulty)
```yaml
idf_thresholds:
  hard: 5.0      # IDF > 5.0 = rare/specific connector
  medium: 4.0    # IDF 4.0-5.0 = moderately specific
  easy: 3.5      # IDF 3.5-4.0 = common connector
```

### 5. Neo4j Credentials Precedence
By default, connectors and evidence retrieval read Neo4j settings from `config.yaml`.
You can override per-run with environment variables:

```bash
export NEO4J_URI=<YOUR_NEO4J_URI>
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
export NEO4J_DATABASE=neo4j
# or
export NEO4J_AUTH="neo4j/your_password"
```

---

## Quick Start

### Option 1: Command Line

```bash
# Quick test (5 questions, NVDA-AMD, 2024, 2-hop)
python pipeline.py --test

# Full production run from config
python pipeline.py --full
```

### Option 2: Python API

```python
from pipeline import Pipeline

p = Pipeline()
results = p.test()  # Quick test mode
```

---

## Caching (Performance Optimization)

The pipeline can cache validated pairs to avoid redundant Neo4j queries on subsequent runs.

### First Run: Build Cache
```bash
# Generate pairs, validate, and save to cache
python pipeline.py --build-cache --orgs NVDA,AMD --year 2024 --hops 2 --num 10
```

### Subsequent Runs: Use Cache
```bash
# Skip Neo4j validation, load from cache (much faster)
python pipeline.py --use-cache --orgs NVDA,AMD --year 2024 --hops 2 --num 10
```

### How It Works
```
Without cache:  Neo4j queries (1600+) → Filter → Generate QA
With cache:     Load JSON (~instant) → Filter → Generate QA
```

Cache file: `outputs/validated_pairs_cache.json`

---

## Usage Examples

### Testing Specific Companies

```bash
# Generate 10 questions for NVDA and AMD only
python pipeline.py --orgs NVDA,AMD --year 2024 --hops 2 --num 10
```

### Connector Mode (Quantitative vs Qualitative)

Use `--quantitative` for numerical/financial connectors or `--qualitative` for strategic/descriptive connectors:

```bash
# Quantitative mode: FIN_METRIC, SEGMENT (financial metrics, segments)
python pipeline.py --quantitative --orgs NVDA,AMD --year 2024 --hops 2 --num 10

# Qualitative mode: RISK_FACTOR, COMP, MACRO_CONDITION (risks, competitors, strategies)
python pipeline.py --qualitative --orgs NVDA,AMD --year 2024 --hops 2 --num 10

# 3-hop Type 2 with quantitative connectors
python pipeline.py --quantitative --hops 3 --pattern cross_company_type2 --orgs NVDA,AMD,INTC --year 2024 --num 5
```

### Testing Specific Hop Patterns

```bash
# 3-hop Type 1: Connector Chain patterns
python pipeline.py --hops 3 --pattern cross_company_type1 --num 5
python pipeline.py --hops 3 --pattern cross_year_type1 --num 5
python pipeline.py --hops 3 --pattern intra_doc_type1 --num 5

# 3-hop Type 2: Multi-Branch patterns
python pipeline.py --hops 3 --pattern cross_company_type2 --num 5
python pipeline.py --hops 3 --pattern cross_year_type2 --num 5
python pipeline.py --hops 3 --pattern intra_doc_type2 --num 5

# Both 2-hop and 3-hop
python pipeline.py --hops 2,3 --num 50
```

### 3-Hop Method (Default vs Causal)
Use the default 3-hop patterns (Type 1 + Type 2), or switch to causal patterns (A/B/C):

```bash
# Default 3-hop logic (Type 1 connector chain + Type 2 multi-branch)
python pipeline.py --hops 3 --num 10

# Causal 3-hop logic (shared driver/outcome/cross anchor)
python pipeline.py --hops 3 --3-hop-method causal --num 10
```

### Causal 3-Hop Patterns (A/B/C)
When `--3-hop-method causal` is enabled, `--pattern` can target one of:

- **Pattern A (Shared driver)**: `both_orgs_originator`
  - Structure: ORG1 → x1 ← ORG2 and x1 → x2
- **Pattern B (Shared outcome)**: `both_orgs_recipient`
  - Structure: ORG1 → x2 ← ORG2 and x1 → x2
- **Pattern C (Cross anchor)**: `both_orgs_separate`
  - Structure: ORG1 → x1 → x2 ← ORG2

If `--pattern` is not provided, all three patterns are used in round‑robin order (A → B → C → repeat) until `--num` is satisfied.

```bash
# All three causal patterns (A/B/C) in round-robin
python pipeline.py --hops 3 --3-hop-method causal --num 9

# Only Pattern A (Shared driver)
python pipeline.py --hops 3 --3-hop-method causal --pattern both_orgs_originator --num 5

# Only Pattern B (Shared outcome)
python pipeline.py --hops 3 --3-hop-method causal --pattern both_orgs_recipient --num 5

# Only Pattern C (Cross anchor)
python pipeline.py --hops 3 --3-hop-method causal --pattern both_orgs_separate --num 5
```

### 2‑Hop + 1‑Branch (Causal “3‑Hop”)
The causal 3‑hop method is effectively a **2‑hop causal link (x1 → x2)** plus a **1‑hop anchor** from each company:

- ORG1 anchors to x1 or x2
- ORG2 anchors to x1 or x2

This yields 3 pieces of evidence (ORG1 anchor, causal link, ORG2 anchor) and produces 3‑hop QA pairs even though the core causal link is 2‑hop.

### When to Use Each Mode
- **Default 3-hop**: broad pattern coverage; good for general multi-hop QA.
- **Causal 3-hop**: emphasizes “driver → outcome” reasoning with cross-company anchoring.
- **2-hop only**: fast validation, connector quality checks, debugging.

### Override Connector Types (2-Hop)
You can override connector entity types used for 2-hop discovery:

```bash
python pipeline.py --orgs NVDA,AMD --year 2024 --hops 2 --num 10 \
  --connector-types COMP,RISK_FACTOR
```

### Testing Specific Years

```bash
# Only 2024, no temporal (single year)
python pipeline.py --year 2024 --skip-temporal --num 20

# Specific years
python pipeline.py --years 2022,2024 --num 30
```

### Testing by Sector

```bash
# Semiconductors sector only
python pipeline.py --sector "Semiconductors" --hops 2,3 --num 100
```

### Python API with Custom Parameters

```python
from pipeline import Pipeline

p = Pipeline()

# Custom run with all overrides
results = p.run(
    orgs=['NVDA', 'AMD', 'INTC'],      # Specific companies
    year=2024,                          # Single year
    hops=[2, 3],                        # Both hop types
    num_questions=50,                   # Total limit
    questions_per_pair=3,               # Per pair limit
    skip_temporal=True,                 # Skip cross_year pairs
    output_prefix="semiconductor_test"  # Output file prefix
)

# Access results
print(f"Total: {results['metadata']['total_questions']}")
for pattern, count in results['metadata']['by_pattern'].items():
    print(f"  {pattern}: {count}")
```

---

## CLI Reference

```
python pipeline.py [OPTIONS]

Mode (mutually exclusive):
  --test              Quick test mode (NVDA-AMD, 2024, 2-hop, 5 questions)
  --full              Full production run from config.yaml

Filters:
  --orgs ORGS         Companies to include (comma-separated)
                      Example: --orgs NVDA,AMD,INTC
  --sector SECTOR     Filter by sector
                      Example: --sector "Semiconductors"
  --year YEAR         Single year to filter
                      Example: --year 2024
  --years YEARS       Multiple years (comma-separated)
                      Example: --years 2022,2024

Hop Configuration:
  --hops HOPS         Hop counts (comma-separated)
                      Example: --hops 2,3
  --pattern PATTERN   Single pattern to run
                      2-hop: cross_company, cross_year, intra_doc
                      3-hop Type 1: cross_company_type1, cross_year_type1, intra_doc_type1
                      3-hop Type 2: cross_company_type2, cross_year_type2, intra_doc_type2
                      3-hop Causal: both_orgs_originator, both_orgs_recipient, both_orgs_separate
                      Note: causal patterns require --3-hop-method causal
  --skip-temporal     Skip cross_year (temporal) pairs
  --skip-cross-company Skip cross_company pairs

Limits:
  --num NUM           Total questions to generate
  --per-pair NUM      Questions per company pair (default: 3)
  --max-pairs NUM     Maximum pairs to process

Connector Mode:
  --quantitative      Use quantitative connector types (FIN_METRIC, SEGMENT, etc.)
  --qualitative       Use qualitative connector types (RISK_FACTOR, COMP, etc.)
                      Note: --connector-types overrides these modes

Other:
  --min-idf FLOAT     Minimum IDF threshold
  --connector-types   Override connector types (comma-separated)
  --llm MODEL         LLM model name (from config.yaml)
  --3-hop-method      3-hop method: default or causal
  --output-dir DIR    Output directory
  --no-save           Don't save results to files

Cache:
  --build-cache       Build pairs cache (saves ~1600 Neo4j queries)
  --use-cache         Load pairs from cache (much faster)
```

---

## How It Works

### Step 1: Pair Generation

The `pair_generator.py` creates company pairs:

- **cross_company Pairs**: Different companies in the same GICS sub-industry, same year
  - Example: (NVDA, AMD, 2024) - both in Semiconductors

- **cross_year Pairs**: Same company across different years
  - Example: (NVDA, NVDA, 2022, 2024) - temporal comparison

- **intra_doc Pairs**: Same company, same year, different pages/chunks
  - Example: (NVDA, NVDA, 2024) - contextual comparison within same document

### Step 2: Connector Discovery

The `connector_discovery.py` finds high-IDF connectors in Neo4j:

```
IDF Score = log(Total_ORGs / Connector_ORG_Degree)
```

- **High IDF (>5.0)**: Rare connector, links few companies = Hard question
- **Low IDF (<3.5)**: Common connector (e.g., "SEC") = Too generic, filtered out

#### 2-Hop Discovery
```cypher
MATCH (o1:ORG {name: 'NVDA'})-[r1]-(connector)-[r2]-(o2:ORG {name: 'AMD'})
WHERE connector:COMP OR connector:RISK_FACTOR ...
RETURN connector, IDF_score
```

#### 3-Hop Discovery
```cypher
// Connector Chain: ORG1 → C1 → C2 ← ORG2
MATCH (o1:ORG)-[r1]-(c1)-[r2]-(c2)-[r3]-(o2:ORG)
WHERE c1 <> c2 AND IDF(c1) > 4.0 AND IDF(c2) > 4.0
RETURN path
```

### Step 3: Evidence Retrieval

The `evidence_retrieval.py` fetches chunk text for each hop:

```python
# Get evidence for the connector from both companies' documents
evidence1 = retriever.get_evidence_for_connector(
    org='NVDA', year=2024,
    connector_name='TSMC', connector_type='COMP'
)
evidence2 = retriever.get_evidence_for_connector(
    org='AMD', year=2024,
    connector_name='TSMC', connector_type='COMP'
)
```

### Step 4: QA Generation

The `qa_generator.py` uses the LLM to generate questions:

1. Formats prompt with evidence from all hops
2. Calls LLM with context-engineered prompts
3. Parses JSON response into QAItem dataclass
4. Validates multi-hop requirement

### Step 5: Output

Results saved to `outputs/` directory, organized by connector mode, hop count, and pattern category:

```
outputs/
├── quantitative/
│   ├── 2hop/
│   │   ├── cross-company/
│   │   │   └── qa_2hop_cross_company_20240112_143022.json
│   │   ├── cross-year/
│   │   │   └── qa_2hop_cross_year_20240112_143022.json
│   │   └── intra-doc/
│   │       └── qa_2hop_intra_doc_20240112_143022.json
│   └── 3hop/
│       ├── cross-company/
│       │   ├── qa_3hop_cross_company_type1_20240112_143022.json
│       │   └── qa_3hop_cross_company_type2_20240112_143022.json
│       ├── cross-year/
│       │   ├── qa_3hop_cross_year_type1_20240112_143022.json
│       │   └── qa_3hop_cross_year_type2_20240112_143022.json
│       └── intra-doc/
│           ├── qa_3hop_intra_doc_type1_20240112_143022.json
│           └── qa_3hop_intra_doc_type2_20240112_143022.json
└── qualitative/
    └── ... (same structure)
```

---

## Output Format

Each generated QA item contains:

```json
{
  "question": "How do NVIDIA and AMD differ in their TSMC dependency?",
  "answer": "NVIDIA relies on TSMC for substantially all GPU production...",
  "reasoning_steps": [
    "Step 1: Extract NVIDIA's TSMC relationship from evidence",
    "Step 2: Extract AMD's TSMC relationship from evidence",
    "Step 3: Compare and synthesize findings"
  ],
  "difficulty": "hard",
  "hop_count": 2,
  "hop_pattern": "cross_company",
  "evidence_ids": ["chunk_nvda_2024_p42", "chunk_amd_2024_p38"],
  "connectors": {
    "connector1": {
      "name": "Taiwan Semiconductor Manufacturing",
      "type": "COMP",
      "idf_score": 5.23
    }
  },
  "metadata": {
    "org1": "NVDA",
    "org2": "AMD",
    "org1_year": 2024,
    "org2_year": 2024,
    "sector": "Semiconductors"
  }
}
```

---

## QA Validation

After generating QA pairs, you can validate and score them using the `qa_validate_output.py` script. The system supports **two validation prompt variants**:

1. **Default (Scoring)**: Uses a 0-50 scoring system with 5 criteria
2. **V3 Binary (Pass/Fail)**: Uses binary PASS/FAIL with explicit failure tags

Both variants can be used during generation (via `pipeline.py`) or post-hoc (via `qa_validate_output.py`).

### Quick Start

```bash
# Validate a single JSON file
python qa_validate_output.py --input outputs/qa_all_20260117_232433.json

# Validate all JSON files in a directory (recursive)
python qa_validate_output.py --input outputs/quantitative/
```

### Validation Variants

#### Default Variant (Scoring 0-50)

Uses a detailed scoring system with 5 criteria, each worth 0-10 points (total 50).

**Scoring Criteria:**
1. **MULTI_CHUNK_DEPENDENCY (0-10)**: Does the question require ALL chunks?
2. **FINANCIAL_CONTEXT (0-10)**: Is it specific to SEC 10-K filings?
3. **ANSWER_GROUNDING (0-10)**: Is every fact traceable to chunks?
4. **SYNTHESIS_AND_INSIGHT (0-10)**: Does it require genuine synthesis?
5. **NATURAL_ANALYST_QUALITY (0-10)**: Does it sound like a real analyst question?

**Thresholds:**
- **EXCELLENT (45-50)**: Exemplary - ACCEPT
- **GOOD (35-44)**: Solid with minor improvements - ACCEPT
- **NEEDS_WORK (25-34)**: Requires revision - REJECT
- **REJECT (0-24)**: Fundamental issues - REJECT

**Output Format:**
```json
{
  "question": "...",
  "answer": "...",
  "validation": {
    "score": 42,
    "decision": "accept",
    "breakdown": {
      "multi_chunk_dependency": 8,
      "financial_context": 9,
      "answer_grounding": 9,
      "synthesis_and_insight": 8,
      "natural_analyst_quality": 8
    },
    "explanation": "This QA pair demonstrates strong multi-hop reasoning...",
    "issues": [],
    "suggestions": [],
    "validated_at": "2026-01-24T18:47:38.628066+00:00"
  }
}
```

#### V3 Binary Variant (Pass/Fail)

Uses binary PASS/FAIL evaluation with explicit failure tags. No scoring - either passes all criteria or fails with specific failure reasons.

**Acceptance Criteria (ALL must be true):**
- A1) Multi-chunk necessity: Requires ALL evidence chunks
- A2) Full answerability: All question parts answerable from evidence
- A3) Grounded answer: All numbers/facts traceable to evidence
- A4) No document structure references: No page/section/chunk mentions
- A5) Natural analyst question: Sounds like real analyst question
- A6) Non-trivial reasoning: Beyond simple fact extraction

**Failure Tags:**
- `FAIL_DOC_REFERENCE`: Mentions pages/sections/chunks
- `FAIL_SINGLE_CHUNK_ANSWERABLE`: Answerable from one chunk
- `FAIL_UNUSED_CHUNK`: Chunk not meaningfully needed
- `FAIL_UNANSWERABLE_SUBPART`: Multi-part question, evidence doesn't support all parts
- `FAIL_MATH_ERROR`: Arithmetic incorrect
- `FAIL_OUT_OF_EVIDENCE_NUMBER`: Number not in evidence
- `FAIL_UNGROUNDED_CLAIM`: Factual claim not supported
- `FAIL_EXTERNAL_KNOWLEDGE_AS_FACT`: External knowledge presented as fact
- `FAIL_AMBIGUOUS_OR_ILL_POSED`: Unclear time framing or entities
- `FAIL_NOT_ANALYST_NATURAL`: Sounds artificial/database-like
- `FAIL_TOO_TRIVIAL`: Simple fact extraction
- `FAIL_FORCED_LINKAGE`: Artificial connection between chunks

**Output Format:**
```json
{
  "question": "...",
  "answer": "...",
  "validation": {
    "score": 50,
    "decision": "pass",
    "breakdown": {},
    "explanation": "This QA pair meets all acceptance criteria...",
    "issues": [],
    "suggestions": [],
    "failure_tags": [],
    "must_fix_issues": [],
    "rewrite_suggestions": [],
    "validated_at": "2026-01-24T18:47:38.628066+00:00"
  }
}
```

For failed questions:
```json
{
  "validation": {
    "score": 0,
    "decision": "fail",
    "failure_tags": ["FAIL_SINGLE_CHUNK_ANSWERABLE", "FAIL_NOT_ANALYST_NATURAL"],
    "must_fix_issues": ["Question can be answered from single chunk", "Sounds like database query"],
    "rewrite_suggestions": ["Require information from both companies' evidence", "Frame as analyst question"],
    "notes": "Question fails multi-chunk requirement..."
  }
}
```

**Note:** For binary variant, `score` is always `0` (FAIL) or `50` (PASS) for compatibility with filtering logic.

### Output Files

For each input file, the script generates two output files in the same directory:

- **`<filename>_validated.json`**: All questions with validation results and feedback
- **`<filename>_filtered.json`**: Only questions that passed validation (score >= threshold for default, decision="pass" for binary)

### Usage Examples

#### Using Default (Scoring) Variant

```bash
# Basic validation (uses default min_score=35 from config.yaml)
python qa_validate_output.py --input outputs/qa_2hop_cross_company_20260122.json

# Explicitly use default variant
python qa_validate_output.py --input outputs/qa_2hop_cross_company_20260122.json --prompt-variant default

# Custom minimum score (only applies to default variant)
python qa_validate_output.py --input outputs/qa_2hop_cross_company_20260122.json --min-score 40

# Limit to first 50 questions (for testing)
python qa_validate_output.py --input outputs/qa_2hop_cross_company_20260122.json --limit 50
```

#### Using V3 Binary Variant

```bash
# Use binary pass/fail validation (min-score ignored)
python qa_validate_output.py --input outputs/qa_2hop_cross_company_20260122.json --prompt-variant v3_binary

# Binary variant with custom settings
python qa_validate_output.py \
  --input outputs/qa_2hop_cross_company_20260122.json \
  --prompt-variant v3_binary \
  --model qwen3-235b
```

#### Directory Processing

```bash
# Process all JSON files recursively (uses variant from config.yaml)
python qa_validate_output.py --input outputs/quantitative/

# Process with default variant and custom threshold
python qa_validate_output.py \
  --input outputs/quantitative/ \
  --prompt-variant default \
  --min-score 40

# Process with binary variant (no threshold needed)
python qa_validate_output.py \
  --input outputs/quantitative/ \
  --prompt-variant v3_binary

# Process only top-level files (no recursion)
python qa_validate_output.py --input outputs/quantitative/ --no-recursive
```

#### Using During Generation

You can also specify the validation variant when generating questions:

```bash
# Generate with default validation (scoring)
python pipeline.py --quantitative --orgs NVDA,AMD --year 2024 --hops 2 --pattern cross_company --num 10

# Generate with binary validation (pass/fail)
python pipeline.py --quantitative --orgs NVDA,AMD --year 2024 --hops 2 --pattern cross_company --num 10 --prompt-variant v3_binary
```

Or set it in `config.yaml`:
```yaml
qa_validation:
  prompt_variant: "v3_binary"  # or "default"
  quality_threshold: 45  # Only used for "default" variant
```

### Command-Line Options

```
python qa_validate_output.py [OPTIONS]

Required:
  --input PATH          Path to QA output JSON file or directory containing JSON files

Optional:
  --output PATH         Path to write scored JSON (default: <input>_validated.json)
                        For directories, outputs go to same subdirectory as input
  --filtered-output PATH Path to write accepted-only JSON (default: <input>_filtered.json)
                        For directories, outputs go to same subdirectory as input
  --prompt-variant {default,v3_binary}
                        Validation prompt variant (default: from config.yaml)
                        - "default": Scoring system (0-50), uses --min-score
                        - "v3_binary": Binary pass/fail, ignores --min-score
  --min-score INT       Minimum score out of 50 to accept (default: 35 from config.yaml)
                        Only used when --prompt-variant is "default"
  --model NAME          Override LLM model name from config.yaml (e.g. qwen3-235b)
  --limit INT           Only validate first N questions per file
  --max-chars-per-chunk INT  Truncate each chunk_text to this many chars (default: 3500)
  --temperature FLOAT   Override LLM temperature for validation calls
  --no-recursive        If input is a directory, only process top-level files
```

### How It Works

1. **Selects prompt variant**: Uses `--prompt-variant` or reads from `config.yaml`
   - `"default"`: Uses `validation` prompt (scoring 0-50)
   - `"v3_binary"`: Uses `validation_v3_binary` prompt (pass/fail)
2. **Loads prompts**: Reads the selected validation prompt from `prompts.yaml`
3. **Extracts evidence**: Pulls chunk text from either:
   - `item["evidence"]` (old format: `org1_evidence`, `org2_evidence`)
   - `item["path_data"]["hop_1_rel"]`, `hop_2_rel`, etc. (new format)
4. **Formats validation context**: Builds pattern, entity chain, evidence, and source files
5. **Calls LLM**: Sends validation prompt and parses JSON response
6. **Processes results**:
   - **Default variant**: Parses score (0-50) and breakdown, applies threshold
   - **Binary variant**: Parses decision (pass/fail) and failure tags, converts to score (0 or 50)
7. **Filters results**: Creates separate file with only accepted questions
   - Default: Questions with `score >= min_score`
   - Binary: Questions with `decision == "pass"`

### Batch Processing Summary

When processing a directory, the script:
- Finds all `*.json` files (excluding `*_validated.json` and `*_filtered.json`)
- Processes each file independently
- Writes outputs to the same subdirectory as the input
- Prints progress for each file and a final summary

Example output:
```
Found 12 JSON file(s) in outputs/quantitative/
Processing with min_score=35, limit=None

[1/12] Processing: 2hop/cross-company/qa_2hop_cross_company_20260122.json
  [qa_2hop_cross_company_20260122.json] Validated 10/10 (accepted so far: 8)
  [qa_2hop_cross_company_20260122.json] Wrote: qa_2hop_cross_company_20260122_validated.json, qa_2hop_cross_company_20260122_filtered.json (accepted 8/10)

...

============================================================
Summary: Processed 12 file(s)
  Total questions: 120
  Accepted: 95 (score >= 35)
  Rejected: 25
```

### Integration with Pipeline

You can chain validation after generation:

```bash
# Generate QA pairs (uses validation variant from config.yaml)
python pipeline.py --orgs NVDA,AMD --year 2024 --hops 2 --num 50

# Validate the output (uses same variant from config.yaml)
python qa_validate_output.py --input outputs/quantitative/2hop/cross-company/
```

Or use it to re-validate existing outputs with different settings:

```bash
# Re-validate with stricter threshold (default variant)
python qa_validate_output.py \
  --input outputs/quantitative/ \
  --prompt-variant default \
  --min-score 40

# Re-validate with binary variant
python qa_validate_output.py \
  --input outputs/quantitative/ \
  --prompt-variant v3_binary
```

### When to Use Each Variant

**Use Default (Scoring) Variant when:**
- You want granular quality scores (0-50)
- You need to filter by score thresholds (e.g., "only excellent questions")
- You want detailed breakdown by criteria
- You're doing quality analysis or ranking

**Use V3 Binary Variant when:**
- You want strict pass/fail evaluation
- You need explicit failure reasons (failure tags)
- You want actionable rewrite suggestions
- You prefer Hamel-style explicit failure modes over Likert scores
- You're doing binary classification or filtering

---

## Module Reference

| Module | Description |
|--------|-------------|
| `pipeline.py` | Main orchestrator, CLI entry point |
| `pair_generator.py` | Generates cross_company/cross_year/intra_doc company pairs |
| `connector_discovery.py` | Finds connectors in Neo4j, calculates IDF |
| `qa_generator.py` | Generates QA using LLM |
| `evidence_retrieval.py` | Retrieves chunk text from Neo4j |
| `llm.py` | OpenAI-compatible LLM client |
| `gics_loader.py` | GICS sub-industry classification |
| `prompts.yaml` | Context-engineered prompts |
| `config.yaml` | Central configuration |

---

## Connector Types

Connectors are bridge entities in the knowledge graph that link companies. The system uses **sector-specific connector types** for relevance.

### All Available Connector Types

| Type | Node Count | Description | Example Entities |
|------|------------|-------------|------------------|
| `FIN_METRIC` | 112,308 | Financial metrics (quantitative) | Revenue, EBITDA, Net Income |
| `FIN_INST` | 21,806 | Financial instruments | Derivatives, Credit Facilities |
| `PRODUCT` | 18,180 | Products & offerings | GPUs, Cloud Services |
| `ACCOUNTING_POLICY` | 17,940 | Accounting policies | Revenue Recognition, Depreciation |
| `RISK_FACTOR` | 15,889 | Risk factors | Supply Chain Disruptions, Cybersecurity |
| `REGULATORY_REQUIREMENT` | 14,727 | Regulatory requirements | Basel III, Dodd-Frank, SOX |
| `COMP` | 8,261 | Competitors & Partners | TSMC, Intel, Samsung |
| `SEGMENT` | 7,579 | Business segments (quantitative) | Data Center, Gaming, Automotive |
| `COMMENTARY` | 7,007 | Management commentary | Outlook, Guidance |
| `CONCEPT` | 6,257 | Business concepts | AI, Cloud Computing |
| `EVENT` | 5,285 | Business events | Acquisitions, Restructuring |
| `MACRO_CONDITION` | 4,945 | Macroeconomic conditions | Inflation, Interest Rates |
| `LITIGATION` | 4,566 | Legal proceedings | Patent disputes, Class actions |
| `PERSON` | 3,747 | Key personnel | CEO, CFO, Directors |
| `LOGISTICS` | 3,286 | Supply chain & logistics | Distribution, Warehousing |
| `GPE` | 2,426 | Geographic entities | United States, China, Taiwan |
| `ESG_TOPIC` | 1,953 | ESG topics | Climate, Diversity, Governance |
| `ECON_IND` | 1,502 | Economic indicators | GDP, Unemployment |
| `SECTOR` | 1,342 | Industry sectors | Technology, Healthcare |
| `RAW_MATERIAL` | 1,191 | Raw materials & suppliers | Silicon Wafers, Rare Earth Elements |
| `ORG_REG` | 1,079 | Regulatory organizations | SEC, PCAOB, Federal Reserve |
| `FIN_MARKET` | 914 | Financial markets | NYSE, Bond Markets |
| `ORG_GOV` | 356 | Government organizations | Federal Reserve, Treasury |

**Note:** Node counts are from the knowledge graph. Higher counts = more QA opportunities.

**Types in sector defaults:** `FIN_METRIC`, `FIN_INST`, `RISK_FACTOR`, `REGULATORY_REQUIREMENT`, `COMP`, `SEGMENT`, `MACRO_CONDITION`, `RAW_MATERIAL`, `ORG_REG`

**Types NOT in defaults (use `--connector-types` to include):** `PRODUCT`, `ACCOUNTING_POLICY`, `COMMENTARY`, `CONCEPT`, `EVENT`, `LITIGATION`, `PERSON`, `LOGISTICS`, `GPE`, `ESG_TOPIC`, `ECON_IND`, `SECTOR`, `FIN_MARKET`, `ORG_GOV`

### Sector-Specific Defaults

When no `--connector-types` is specified, the system uses GICS sector-based defaults:

| Sector | Default Connector Types |
|--------|------------------------|
| Information Technology | `COMP`, `RISK_FACTOR`, `MACRO_CONDITION`, `RAW_MATERIAL` |
| Financials | `FIN_INST`, `REGULATORY_REQUIREMENT`, `RISK_FACTOR`, `ORG_REG`, `MACRO_CONDITION` |
| Health Care | `REGULATORY_REQUIREMENT`, `RISK_FACTOR`, `COMP`, `FIN_INST` |
| Industrials | `COMP`, `RISK_FACTOR`, `MACRO_CONDITION`, `RAW_MATERIAL`, `REGULATORY_REQUIREMENT` |
| Communication Services | `COMP`, `RISK_FACTOR`, `MACRO_CONDITION`, `REGULATORY_REQUIREMENT` |
| Consumer Staples | `COMP`, `RISK_FACTOR`, `MACRO_CONDITION`, `RAW_MATERIAL` |
| Consumer Discretionary | `COMP`, `RISK_FACTOR`, `MACRO_CONDITION` |
| Energy | `FIN_INST`, `REGULATORY_REQUIREMENT`, `RISK_FACTOR`, `RAW_MATERIAL`, `MACRO_CONDITION` |
| Utilities | `REGULATORY_REQUIREMENT`, `RISK_FACTOR`, `MACRO_CONDITION`, `FIN_INST` |
| Real Estate | `FIN_INST`, `REGULATORY_REQUIREMENT`, `RISK_FACTOR`, `MACRO_CONDITION` |
| Materials | `RAW_MATERIAL`, `RISK_FACTOR`, `MACRO_CONDITION`, `COMP` |

### Overriding Connector Types

Use `--connector-types` to override sector defaults:

```bash
# Use only quantitative connectors
python pipeline.py --pattern cross_company --connector-types FIN_METRIC,SEGMENT --num 5

# Use specific qualitative connectors
python pipeline.py --pattern cross_year --connector-types RISK_FACTOR,MACRO_CONDITION --num 5

# Mix of types
python pipeline.py --pattern intra_doc --connector-types COMP,FIN_METRIC,SEGMENT --num 5
```

### Why Sector-Specific Types?

Different sectors have different "vocabulary" of connectors in their 10-K filings:
- **Tech companies** discuss competitors, supply chain, raw materials
- **Banks** discuss financial instruments, regulations, credit risks
- **Energy companies** discuss commodities, regulations, environmental factors

See `docs/logic.md` Section 5 for empirical analysis and rationale.

---

## Troubleshooting

### Neo4j Connection Failed
```
Check config.yaml:
- neo4j.uri is correct (default: <YOUR_NEO4J_URI>)
- neo4j.username and password are correct
- Neo4j server is running
```

### No Connectors Found
```
Possible causes:
- IDF threshold too high (try --min-idf 3.5)
- Companies have no shared connectors
- Year filter too restrictive
```

### LLM Errors
```
Check config.yaml:
- llm.models.<model>.base_url is accessible
- llm.models.<model>.model name is correct
- API key if required
```

---

## Examples of Generated Questions

### 2-Hop Cross-Company (Hard)
> **Q**: How do NVIDIA and AMD differ in their disclosed dependency on Taiwan Semiconductor Manufacturing Company (TSMC) for chip fabrication, and what specific risks does each company identify related to this supplier relationship?

### 2-Hop Cross-Year (Medium)
> **Q**: How has NVIDIA's data center business segment evolved from fiscal 2022 to fiscal 2024 in terms of revenue contribution and strategic positioning?

### 2-Hop Intra-Document
> **Q**: How does NVIDIA characterize TSMC differently in its supply chain risks section compared to its revenue concentration discussion within the same 10-K filing?

### 3-Hop Type 1: Connector Chain (cross_company_type1)
> **Q**: How do NVIDIA's production arrangements with TSMC and AMD's disclosed silicon wafer supply concerns relate through their shared dependency on semiconductor manufacturing materials?

### 3-Hop Type 1: Cross-Year Chain (cross_year_type1)
> **Q**: How has NVIDIA's relationship with AI chip demand evolved from 2022 to 2024, considering the company's data center strategy and market positioning across these years?

### 3-Hop Type 2: Multi-Company (cross_company_type2)
> **Q**: How do NVIDIA, AMD, and Intel each characterize their dependency on advanced semiconductor manufacturing, and what commonalities emerge from their risk disclosures?

### 3-Hop Type 2: Multi-Year (cross_year_type2)
> **Q**: How has NVIDIA's characterization of supply chain concentration risk evolved across 2022, 2023, and 2024, and what trends can be identified?

---

## License

Internal use only.
