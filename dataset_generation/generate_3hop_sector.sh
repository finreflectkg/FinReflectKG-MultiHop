#!/bin/bash
# Generate 3-hop QA pairs with SECTOR-WISE distribution
#
# Total per pattern: 1500 questions
# Sectors: 11 GICS sectors
# Per sector: ~137 questions (1500/11)
#
# HOW TO USE:
# 1. Uncomment the command(s) you want to run
# 2. Run: ./generate_3hop_sector.sh
# 3. Comment out completed commands, uncomment next ones

cd "$(dirname "$0")"

# Per-sector count: 1500 / 11 = ~137
COUNT=250
WORKERS=50

# ============================================================
# QUANTITATIVE 3-HOP TYPE 1 - cross_company_type1
# DONE: Information Technology
# ============================================================

# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Information Technology" --num $COUNT --workers $WORKERS  # DONE
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Financials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type1 --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUANTITATIVE 3-HOP TYPE 1 - cross_year_type1
# ============================================================

# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Information Technology" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Financials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type1 --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUANTITATIVE 3-HOP TYPE 1 - intra_doc_type1
# ============================================================

# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Information Technology" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Financials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type1 --sector "Real Estate" --num $COUNT --workers $WORKERS

# # ============================================================
# # QUANTITATIVE 3-HOP TYPE 2 - cross_company_type2
# # DONE: IT, Financials, Health Care, Industrials, Consumer Discretionary, Communication Services
# # ============================================================

# # uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Information Technology" --num $COUNT --workers $WORKERS  # DONE
# # uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Financials" --num $COUNT --workers $WORKERS  # DONE
# # uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Health Care" --num $COUNT --workers $WORKERS  # DONE
# # uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Industrials" --num $COUNT --workers $WORKERS  # DONE
# # uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS  # DONE
# # uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Communication Services" --num $COUNT --workers $WORKERS  # DONE
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_company_type2 --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUANTITATIVE 3-HOP TYPE 2 - cross_year_type2
# DONE: IT, Financials
# ============================================================

# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Information Technology" --num $COUNT --workers $WORKERS  # DONE
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Financials" --num $COUNT --workers $WORKERS  # DONE
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern cross_year_type2 --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUANTITATIVE 3-HOP TYPE 2 - intra_doc_type2
# ============================================================

# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Information Technology" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Financials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --quantitative --hops 3 --pattern intra_doc_type2 --sector "Real Estate" --num $COUNT --workers $WORKERS

# # ============================================================
# # QUALITATIVE 3-HOP TYPE 1 - cross_company_type1
# # ============================================================

# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Information Technology" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Financials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Energy" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Utilities" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Materials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type1 --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUALITATIVE 3-HOP TYPE 1 - cross_year_type1
# ============================================================

# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Information Technology" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Financials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Health Care" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Industrials" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Communication Services" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
# uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Energy" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Utilities" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Materials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type1 --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUALITATIVE 3-HOP TYPE 1 - intra_doc_type1
# ============================================================

uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Information Technology" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Financials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Health Care" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Industrials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Communication Services" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Energy" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Utilities" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Materials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type1 --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUALITATIVE 3-HOP TYPE 2 - cross_company_type2
# ============================================================

uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Information Technology" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Financials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Health Care" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Industrials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Communication Services" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Energy" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Utilities" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Materials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_company_type2 --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUALITATIVE 3-HOP TYPE 2 - cross_year_type2
# ============================================================

uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Information Technology" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Financials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Health Care" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Industrials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Communication Services" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Energy" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Utilities" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Materials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern cross_year_type2 --sector "Real Estate" --num $COUNT --workers $WORKERS

# ============================================================
# QUALITATIVE 3-HOP TYPE 2 - intra_doc_type2
# ============================================================

uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Information Technology" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Financials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Health Care" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Industrials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Consumer Discretionary" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Communication Services" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Consumer Staples" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Energy" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Utilities" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Materials" --num $COUNT --workers $WORKERS
uv run python pipeline.py --use-cache --qualitative --hops 3 --pattern intra_doc_type2 --sector "Real Estate" --num $COUNT --workers $WORKERS

echo "Done! Uncomment commands to run them."
