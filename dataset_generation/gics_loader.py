"""
GICS Data Loader for SP100 Companies

This module fetches GICS (Global Industry Classification Standard) data
for S&P 100 companies from Wikipedia and provides sector-specific
connector type configurations.

Connector types are split into two categories:
- Quantitative: FIN_METRIC, SEGMENT, ECON_IND (numerical/financial data)
- Qualitative: FIN_INST, PRODUCT, RISK_FACTOR, etc. (descriptive/text data)

Usage:
    from gics_loader import (
        load_sp100_gics,
        get_sector_quantitative_types,
        get_sector_qualitative_types,
        get_sector_connector_types,
        get_companies_by_sector
    )

    gics_data = load_sp100_gics()
    # {'NVDA': {'sector': 'Information Technology', 'sub_industry': 'Semiconductors'}, ...}

    # Get quantitative types (for numerical comparison questions)
    quant_types = get_sector_quantitative_types('Information Technology')
    # ['FIN_METRIC', 'SEGMENT', 'ECON_IND']

    # Get qualitative types (for descriptive/strategy questions)
    qual_types = get_sector_qualitative_types('Information Technology')
    # ['PRODUCT', 'FIN_INST', 'RISK_FACTOR', 'ACCOUNTING_POLICY', ...]

    # Get combined types (backwards compatible)
    all_types = get_sector_connector_types('Information Technology')

See docs/sector-connector-analysis.md for detailed analysis of connector type distribution.
"""

import pandas as pd
import requests
from io import StringIO
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# SECTOR-SPECIFIC CONNECTOR TYPES
# Based on empirical analysis of Neo4j knowledge graph (Jan 2025)
# See docs/sector-connector-analysis.md for detailed breakdown
###############################################################################

# Quantitative connector types: Numerical/financial data for comparison questions
# These types contain metrics, figures, and quantitative data
SECTOR_QUANTITATIVE_TYPES: Dict[str, List[str]] = {
    'Information Technology': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
    'Financials': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
    'Health Care': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
    'Industrials': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
    'Consumer Discretionary': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
    'Communication Services': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
    'Consumer Staples': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
    'Energy': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
    'Utilities': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
    'Materials': ['FIN_METRIC', 'SEGMENT'],  # Limited ECON_IND data
    'Real Estate': ['FIN_METRIC', 'SEGMENT', 'ECON_IND'],
}

# Qualitative connector types: Descriptive/text data for strategy/risk questions
# These types contain narrative information about risks, products, policies, etc.
SECTOR_QUALITATIVE_TYPES: Dict[str, List[str]] = {
    'Information Technology': ['PRODUCT', 'FIN_INST', 'RISK_FACTOR', 'ACCOUNTING_POLICY', 'REGULATORY_REQUIREMENT', 'COMP', 'CONCEPT', 'GPE'],
    'Financials': ['FIN_INST', 'ACCOUNTING_POLICY', 'RISK_FACTOR', 'REGULATORY_REQUIREMENT', 'PRODUCT', 'GPE', 'MACRO_CONDITION', 'COMP'],
    'Health Care': ['FIN_INST', 'PRODUCT', 'REGULATORY_REQUIREMENT', 'ACCOUNTING_POLICY', 'RISK_FACTOR', 'COMP', 'GPE', 'LITIGATION'],
    'Industrials': ['FIN_INST', 'ACCOUNTING_POLICY', 'RISK_FACTOR', 'REGULATORY_REQUIREMENT', 'PRODUCT', 'GPE', 'MACRO_CONDITION', 'COMP'],
    'Consumer Discretionary': ['FIN_INST', 'ACCOUNTING_POLICY', 'RISK_FACTOR', 'REGULATORY_REQUIREMENT', 'PRODUCT', 'GPE', 'COMP', 'CONCEPT'],
    'Communication Services': ['FIN_INST', 'PRODUCT', 'ACCOUNTING_POLICY', 'REGULATORY_REQUIREMENT', 'RISK_FACTOR', 'COMP', 'GPE', 'LITIGATION'],
    'Consumer Staples': ['FIN_INST', 'ACCOUNTING_POLICY', 'REGULATORY_REQUIREMENT', 'GPE', 'RISK_FACTOR', 'PRODUCT', 'COMP', 'MACRO_CONDITION'],
    'Energy': ['GPE', 'FIN_INST', 'ACCOUNTING_POLICY', 'REGULATORY_REQUIREMENT', 'RISK_FACTOR', 'COMP', 'RAW_MATERIAL', 'MACRO_CONDITION'],
    'Utilities': ['FIN_INST', 'ACCOUNTING_POLICY', 'REGULATORY_REQUIREMENT', 'RISK_FACTOR', 'GPE', 'COMP', 'PRODUCT', 'MACRO_CONDITION'],
    'Materials': ['FIN_INST', 'GPE', 'ACCOUNTING_POLICY', 'RAW_MATERIAL', 'RISK_FACTOR', 'REGULATORY_REQUIREMENT', 'PRODUCT', 'MACRO_CONDITION'],
    'Real Estate': ['FIN_INST', 'GPE', 'COMP', 'ACCOUNTING_POLICY', 'REGULATORY_REQUIREMENT', 'RISK_FACTOR', 'PRODUCT', 'MACRO_CONDITION'],
}

# Default types for unknown sectors
DEFAULT_QUANTITATIVE_TYPES = ['FIN_METRIC', 'SEGMENT', 'ECON_IND']
DEFAULT_QUALITATIVE_TYPES = ['FIN_INST', 'PRODUCT', 'RISK_FACTOR', 'ACCOUNTING_POLICY', 'REGULATORY_REQUIREMENT', 'COMP', 'GPE', 'MACRO_CONDITION']

# Legacy: Combined connector types (for backwards compatibility)
# This merges quantitative + qualitative for each sector
SECTOR_CONNECTOR_TYPES: Dict[str, List[str]] = {
    sector: SECTOR_QUANTITATIVE_TYPES.get(sector, DEFAULT_QUANTITATIVE_TYPES) +
            SECTOR_QUALITATIVE_TYPES.get(sector, DEFAULT_QUALITATIVE_TYPES)
    for sector in set(list(SECTOR_QUANTITATIVE_TYPES.keys()) + list(SECTOR_QUALITATIVE_TYPES.keys()))
}

# Default connector types for unknown sectors (combined)
DEFAULT_CONNECTOR_TYPES = DEFAULT_QUANTITATIVE_TYPES + DEFAULT_QUALITATIVE_TYPES


def fetch_sp500_data() -> pd.DataFrame:
    """
    Fetch S&P 500 company data from Wikipedia.

    Returns:
        DataFrame with columns: Symbol, Security, GICS Sector, GICS Sub-Industry, etc.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    logger.info("Fetching S&P 500 data from Wikipedia...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    df_sp500 = pd.DataFrame(tables[0])
    logger.info(f"Fetched {len(df_sp500)} S&P 500 companies")

    return df_sp500


def fetch_sp100_symbols() -> List[str]:
    """
    Fetch S&P 100 company symbols from Wikipedia.

    Returns:
        List of ticker symbols in S&P 100
    """
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    logger.info("Fetching S&P 100 symbols from Wikipedia...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    df_sp100 = pd.DataFrame(tables[2])
    symbols = df_sp100['Symbol'].unique().tolist()
    logger.info(f"Fetched {len(symbols)} S&P 100 symbols")

    return symbols


def load_sp100_gics() -> Dict[str, Dict[str, str]]:
    """
    Load GICS classification for all S&P 100 companies.

    Returns:
        Dictionary mapping ticker to GICS info:
        {
            'NVDA': {
                'sector': 'Information Technology',
                'sub_industry': 'Semiconductors',
                'security_name': 'NVIDIA Corporation'
            },
            ...
        }
    """
    df_sp500 = fetch_sp500_data()
    sp100_symbols = fetch_sp100_symbols()

    # Filter to S&P 100
    df_sp100 = df_sp500[df_sp500['Symbol'].isin(sp100_symbols)]
    logger.info(f"Filtered to {len(df_sp100)} S&P 100 companies")

    gics_data = {}
    for _, row in df_sp100.iterrows():
        symbol = row['Symbol']
        gics_data[symbol] = {
            'sector': row['GICS Sector'],
            'sub_industry': row['GICS Sub-Industry'],
            'security_name': row['Security']
        }

    return gics_data


def get_sector_quantitative_types(sector: str) -> List[str]:
    """
    Get the quantitative connector types for a given GICS sector.

    Quantitative types contain numerical/financial data suitable for
    comparison questions about metrics, performance, and trends.

    Args:
        sector: GICS sector name (e.g., 'Information Technology')

    Returns:
        List of quantitative connector type labels (e.g., ['FIN_METRIC', 'SEGMENT', 'ECON_IND'])
    """
    return SECTOR_QUANTITATIVE_TYPES.get(sector, DEFAULT_QUANTITATIVE_TYPES)


def get_sector_qualitative_types(sector: str) -> List[str]:
    """
    Get the qualitative connector types for a given GICS sector.

    Qualitative types contain descriptive/text data suitable for
    comparison questions about strategies, risks, products, and policies.

    Args:
        sector: GICS sector name (e.g., 'Information Technology')

    Returns:
        List of qualitative connector type labels (e.g., ['PRODUCT', 'FIN_INST', 'RISK_FACTOR', ...])
    """
    return SECTOR_QUALITATIVE_TYPES.get(sector, DEFAULT_QUALITATIVE_TYPES)


def get_sector_connector_types(sector: str, mode: str = 'all') -> List[str]:
    """
    Get connector types for a given GICS sector.

    Args:
        sector: GICS sector name (e.g., 'Information Technology')
        mode: One of 'all', 'quantitative', 'qualitative'
            - 'all': Returns combined quantitative + qualitative types
            - 'quantitative': Returns only quantitative types (FIN_METRIC, SEGMENT, ECON_IND)
            - 'qualitative': Returns only qualitative types (PRODUCT, FIN_INST, etc.)

    Returns:
        List of connector type labels to use for this sector
    """
    if mode == 'quantitative':
        return get_sector_quantitative_types(sector)
    elif mode == 'qualitative':
        return get_sector_qualitative_types(sector)
    else:  # 'all' or any other value
        return SECTOR_CONNECTOR_TYPES.get(sector, DEFAULT_CONNECTOR_TYPES)


def get_companies_by_sector(gics_data: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Group companies by their GICS sector.

    Args:
        gics_data: Output from load_sp100_gics()

    Returns:
        Dictionary mapping sector to list of tickers:
        {
            'Information Technology': ['AAPL', 'MSFT', 'NVDA', ...],
            'Financials': ['JPM', 'BAC', 'GS', ...],
            ...
        }
    """
    sectors = {}
    for ticker, info in gics_data.items():
        sector = info['sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(ticker)

    # Sort tickers within each sector
    for sector in sectors:
        sectors[sector].sort()

    return sectors


def get_companies_by_sub_industry(gics_data: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Group companies by their GICS sub-industry.

    Args:
        gics_data: Output from load_sp100_gics()

    Returns:
        Dictionary mapping sub-industry to list of tickers:
        {
            'Semiconductors': ['NVDA', 'AMD', 'INTC', ...],
            'Diversified Banks': ['JPM', 'BAC', 'WFC', ...],
            ...
        }
    """
    sub_industries = {}
    for ticker, info in gics_data.items():
        sub_ind = info['sub_industry']
        if sub_ind not in sub_industries:
            sub_industries[sub_ind] = []
        sub_industries[sub_ind].append(ticker)

    # Sort tickers within each sub-industry
    for sub_ind in sub_industries:
        sub_industries[sub_ind].sort()

    return sub_industries


def print_sector_summary(gics_data: Dict[str, Dict[str, str]]) -> None:
    """Print a summary of companies by sector with connector type breakdown."""
    sectors = get_companies_by_sector(gics_data)

    print("=" * 80)
    print("S&P 100 Companies by GICS Sector")
    print("=" * 80)

    for sector, tickers in sorted(sectors.items(), key=lambda x: -len(x[1])):
        print(f"\n{sector} ({len(tickers)} companies):")
        print(f"  Tickers: {', '.join(tickers)}")
        print(f"  Quantitative Types: {get_sector_quantitative_types(sector)}")
        print(f"  Qualitative Types: {get_sector_qualitative_types(sector)}")


if __name__ == "__main__":
    # Test the loader
    gics_data = load_sp100_gics()
    print_sector_summary(gics_data)

    # Show sub-industry groupings
    print("\n" + "=" * 80)
    print("Sub-Industry Groupings (for within-industry QA pairs)")
    print("=" * 80)

    sub_industries = get_companies_by_sub_industry(gics_data)
    for sub_ind, tickers in sorted(sub_industries.items(), key=lambda x: -len(x[1])):
        if len(tickers) >= 2:  # Only show sub-industries with 2+ companies
            print(f"\n{sub_ind} ({len(tickers)} companies):")
            print(f"  {', '.join(tickers)}")
