#!/bin/bash
# Generate QA pairs for all patterns with SECTOR-WISE BALANCE (excluding causal)
# Usage: ./generate_all.sh [mode] [workers]
# mode: quantitative (default) or qualitative
# workers: parallel workers (default 50)

MODE=${1:-quantitative}
WORKERS=${2:-50}

# All 11 GICS sectors
SECTORS=(
    "Information Technology"
    "Financials"
    "Health Care"
    "Industrials"
    "Consumer Discretionary"
    "Communication Services"
    "Consumer Staples"
    "Energy"
    "Utilities"
    "Materials"
    "Real Estate"
)

echo "=============================================="
echo "QA Generation - Mode: $MODE, Workers: $WORKERS"
echo "With SECTOR-WISE BALANCE (11 sectors)"
echo "=============================================="

# Set mode flag
if [ "$MODE" == "qualitative" ]; then
    MODE_FLAG="--qualitative"
else
    MODE_FLAG="--quantitative"
fi

cd "$(dirname "$0")"

# Per-sector counts (total / 11 sectors, rounded up)
# 2-hop: 1000 total = ~91 per sector
# 3-hop: 500 total = ~46 per sector
COUNT_2HOP=91
COUNT_3HOP=46

echo ""
echo "===== 2-HOP PATTERNS (1000 each = 91 per sector) ====="
echo "---------------------------------------------"

for SECTOR in "${SECTORS[@]}"; do
    echo ""
    echo ">>> Sector: $SECTOR"

    echo "  [2-hop] cross_company..."
    uv run python pipeline.py $MODE_FLAG --sector "$SECTOR" --hops 2 --pattern cross_company --num $COUNT_2HOP --workers $WORKERS

    echo "  [2-hop] cross_year..."
    uv run python pipeline.py $MODE_FLAG --sector "$SECTOR" --hops 2 --pattern cross_year --num $COUNT_2HOP --workers $WORKERS

    echo "  [2-hop] intra_doc..."
    uv run python pipeline.py $MODE_FLAG --sector "$SECTOR" --hops 2 --pattern intra_doc --num $COUNT_2HOP --workers $WORKERS
done

echo ""
echo "===== 3-HOP TYPE 1 PATTERNS (500 each = 46 per sector) ====="
echo "---------------------------------------------"

for SECTOR in "${SECTORS[@]}"; do
    echo ""
    echo ">>> Sector: $SECTOR"

    echo "  [3-hop] cross_company_type1..."
    uv run python pipeline.py $MODE_FLAG --sector "$SECTOR" --hops 3 --pattern cross_company_type1 --num $COUNT_3HOP --workers $WORKERS

    echo "  [3-hop] cross_year_type1..."
    uv run python pipeline.py $MODE_FLAG --sector "$SECTOR" --hops 3 --pattern cross_year_type1 --num $COUNT_3HOP --workers $WORKERS

    echo "  [3-hop] intra_doc_type1..."
    uv run python pipeline.py $MODE_FLAG --sector "$SECTOR" --hops 3 --pattern intra_doc_type1 --num $COUNT_3HOP --workers $WORKERS
done

echo ""
echo "===== 3-HOP TYPE 2 PATTERNS (500 each = 46 per sector) ====="
echo "---------------------------------------------"

for SECTOR in "${SECTORS[@]}"; do
    echo ""
    echo ">>> Sector: $SECTOR"

    echo "  [3-hop] cross_company_type2..."
    uv run python pipeline.py $MODE_FLAG --sector "$SECTOR" --hops 3 --pattern cross_company_type2 --num $COUNT_3HOP --workers $WORKERS

    echo "  [3-hop] cross_year_type2..."
    uv run python pipeline.py $MODE_FLAG --sector "$SECTOR" --hops 3 --pattern cross_year_type2 --num $COUNT_3HOP --workers $WORKERS

    echo "  [3-hop] intra_doc_type2..."
    uv run python pipeline.py $MODE_FLAG --sector "$SECTOR" --hops 3 --pattern intra_doc_type2 --num $COUNT_3HOP --workers $WORKERS
done

echo ""
echo "=============================================="
echo "Generation complete!"
echo "Mode: $MODE"
echo "Output files in: outputs/$MODE/"
echo ""
echo "Summary (per mode):"
echo "  2-hop: ~3003 questions (91 x 11 sectors x 3 patterns)"
echo "  3-hop type1: ~1518 questions (46 x 11 sectors x 3 patterns)"
echo "  3-hop type2: ~1518 questions (46 x 11 sectors x 3 patterns)"
echo "  Total: ~6039 questions"
echo "=============================================="
