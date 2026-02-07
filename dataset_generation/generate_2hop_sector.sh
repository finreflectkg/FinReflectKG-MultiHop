#!/bin/bash
# Generate 2-hop QA pairs with SECTOR-WISE distribution
#
# Total per pattern: ~2750 questions (250 per sector × 11 sectors)
# Sectors: 11 GICS sectors
# Per sector: 250 questions
#
# HOW TO USE:
# 1. Uncomment the command(s) you want to run
# 2. Run: ./generate_2hop_sector.sh
# 3. Comment out completed commands, uncomment next ones

cd "$(dirname "$0")"

# Per-sector count
COUNT=250
WORKERS=40
# ============================================================
# BUILD CACHE FIRST (only needs to run once)
# ============================================================
# echo "Building pairs cache (skips Neo4j validation on future runs)..."
# uv run python pipeline.py --build-cache --hops 2 --num 1 --workers 1

# ============================================================
# QUANTITATIVE 2-HOP - cross_company
# DONE: IT, Financials, Health Care, Industrials, Consumer Discretionary
# ============================================================

# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Information Technology" --num $COUNT --workers $WORKERS  # DONE
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Financials" --num $COUNT --workers $WORKERS  # DONE
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Health Care" --num $COUNT --workers $WORKERS  # DONE
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Industrials" --num $COUNT --workers $WORKERS  # DONE
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS  # DONE
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_company --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUANTITATIVE 2-HOP - cross_year
# ============================================================

# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Information Technology" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Financials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern cross_year --sector "Real Estate" --num $COUNT --workers $WORKERS

# # ============================================================
# # QUANTITATIVE 2-HOP - intra_doc
# # ============================================================

# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Information Technology" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Financials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 2 --pattern intra_doc --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUALITATIVE 2-HOP - cross_company
# ============================================================

# uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Information Technology" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Financials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Communication Services" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Consumer Staples" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Energy" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Utilities" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Materials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_company --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUALITATIVE 2-HOP - cross_year
# ============================================================

uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Information Technology" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Financials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Health Care" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Industrials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Communication Services" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Consumer Staples" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Energy" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Utilities" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Materials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern cross_year --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUALITATIVE 2-HOP - intra_doc
# ============================================================

uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Information Technology" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Financials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Health Care" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Industrials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Communication Services" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Consumer Staples" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Energy" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Utilities" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Materials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 2 --pattern intra_doc --sector "Real Estate" --num $COUNT --workers $WORKERS

echo "Done! Uncomment commands to run them."
