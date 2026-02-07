"""
Company Pair Generator Module

This module generates company pairs for multi-hop QA generation.
It uses GICS classification to group companies by sector/sub-industry,
then validates that pairs have sufficient connectors in the graph.

Strategy:
1. Within sub-industry pairs (highest quality): Companies in same sub-industry
   e.g., NVDA-AMD (both Semiconductors)
2. Within sector pairs (good quality): Companies in same sector but different sub-industry
   e.g., NVDA-AAPL (both Information Technology)
3. Cross-sector pairs (use sparingly): Companies from different sectors
   e.g., NVDA-JPM (Tech vs Finance)

Configuration:
    All parameters are loaded from config.yaml. See config.yaml for:
    - cross_company_years: Which years to use for cross-company pairs
    - cross_year_combinations: Which year combinations for cross-year pairs
    - intra_doc_years: Which years to use for intra-doc pairs
    - validation thresholds
    - output settings

Usage:
    from pair_generator import PairGenerator, load_config

    config = load_config()
    generator = PairGenerator(config=config)
    pairs = generator.generate_all_pairs_from_config()
"""

import itertools
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from gics_loader import (
    load_sp100_gics,
    get_sector_connector_types,
    get_companies_by_sector,
    get_companies_by_sub_industry
)
from connector_discovery import ConnectorDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.yaml

    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


@dataclass
class CompanyPair:
    """A pair of companies for QA generation."""
    org1: str
    org2: str
    sector: str
    sub_industry1: str
    sub_industry2: str
    pair_type: str  # 'within_subindustry', 'within_sector', 'cross_sector'
    connector_types: List[str]

    # Temporal fields
    org1_year: int = 2024  # Default to most recent year
    org2_year: int = 2024
    pair_category: str = 'cross_company'  # 'cross_company', 'cross_year' (same-company temporal), 'intra_doc' (same company, same year, different pages)

    # Stats (filled after validation)
    hard_connectors: int = 0
    medium_connectors: int = 0
    easy_connectors: int = 0
    total_connectors: int = 0
    relationship_changes: int = 0  # For Type 2 pairs

    @property
    def sub_industry(self) -> str:
        """Return sub-industry (same if within_subindustry)."""
        return self.sub_industry1

    @property
    def is_same_year(self) -> bool:
        """Check if both orgs are from the same year."""
        return self.org1_year == self.org2_year

    @property
    def is_same_company(self) -> bool:
        """Check if this is a temporal pair (same company, different years)."""
        return self.org1 == self.org2

    @property
    def has_sufficient_connectors(self) -> bool:
        """Check if pair has enough connectors for QA generation."""
        return self.hard_connectors >= 1 or self.medium_connectors >= 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'org1': self.org1,
            'org1_year': self.org1_year,
            'org2': self.org2,
            'org2_year': self.org2_year,
            'sector': self.sector,
            'sub_industry1': self.sub_industry1,
            'sub_industry2': self.sub_industry2,
            'pair_type': self.pair_type,
            'pair_category': self.pair_category,
            'connector_types': self.connector_types,
            'stats': {
                'hard_connectors': self.hard_connectors,
                'medium_connectors': self.medium_connectors,
                'easy_connectors': self.easy_connectors,
                'total_connectors': self.total_connectors,
                'relationship_changes': self.relationship_changes
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompanyPair':
        """Create CompanyPair from dictionary (for loading from cache)."""
        stats = data.get('stats', {})
        return cls(
            org1=data['org1'],
            org2=data['org2'],
            sector=data['sector'],
            sub_industry1=data['sub_industry1'],
            sub_industry2=data['sub_industry2'],
            pair_type=data['pair_type'],
            connector_types=data['connector_types'],
            org1_year=data.get('org1_year', 2024),
            org2_year=data.get('org2_year', 2024),
            pair_category=data.get('pair_category', 'cross_company'),
            hard_connectors=stats.get('hard_connectors', 0),
            medium_connectors=stats.get('medium_connectors', 0),
            easy_connectors=stats.get('easy_connectors', 0),
            total_connectors=stats.get('total_connectors', 0),
            relationship_changes=stats.get('relationship_changes', 0)
        )


class PairGenerator:
    """
    Generates company pairs for multi-hop QA generation.

    Example:
        config = load_config()
        generator = PairGenerator(config=config)
        pairs = generator.generate_all_pairs_from_config()
        for p in pairs:
            if p.has_sufficient_connectors:
                print(f"{p.org1}-{p.org2}: {p.hard_connectors} hard connectors")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, validate_connectors: bool = True):
        """
        Initialize the pair generator.

        Args:
            config: Configuration dictionary (from load_config()). If None, loads default.
            validate_connectors: If True, check Neo4j for connector counts
        """
        # Load config if not provided
        if config is None:
            config = load_config()
        self.config = config

        self.validate_connectors = validate_connectors
        self.gics_data = None
        self.discovery = None

        # Load GICS data
        logger.info("Loading GICS data...")
        self.gics_data = load_sp100_gics()
        logger.info(f"Loaded GICS data for {len(self.gics_data)} companies")

        # Initialize connector discovery if validation enabled
        if validate_connectors:
            self.discovery = ConnectorDiscovery()

    def close(self):
        """Close resources."""
        if self.discovery:
            self.discovery.close()

    # =========================================================================
    # CACHE METHODS - Avoid redundant Neo4j queries
    # =========================================================================

    def save_pairs_to_cache(self, pairs: List[CompanyPair], cache_path: Optional[str] = None) -> str:
        """
        Save validated pairs to JSON cache file.

        Args:
            pairs: List of CompanyPair objects
            cache_path: Optional path. Default: outputs/validated_pairs_cache.json

        Returns:
            Path to saved cache file
        """
        if cache_path is None:
            cache_dir = Path(__file__).parent / "outputs"
            cache_dir.mkdir(exist_ok=True)
            cache_path = cache_dir / "validated_pairs_cache.json"
        else:
            cache_path = Path(cache_path)

        data = {
            'version': '1.0',
            'count': len(pairs),
            'pairs': [p.to_dict() for p in pairs]
        }

        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(pairs)} pairs to cache: {cache_path}")
        return str(cache_path)

    @staticmethod
    def load_pairs_from_cache(cache_path: Optional[str] = None) -> List[CompanyPair]:
        """
        Load validated pairs from JSON cache file.

        Args:
            cache_path: Optional path. Default: outputs/validated_pairs_cache.json

        Returns:
            List of CompanyPair objects
        """
        if cache_path is None:
            cache_path = Path(__file__).parent / "outputs" / "validated_pairs_cache.json"
        else:
            cache_path = Path(cache_path)

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        with open(cache_path, 'r') as f:
            data = json.load(f)

        pairs = [CompanyPair.from_dict(p) for p in data['pairs']]
        logger.info(f"Loaded {len(pairs)} pairs from cache: {cache_path}")
        return pairs

    @staticmethod
    def cache_exists(cache_path: Optional[str] = None) -> bool:
        """Check if cache file exists."""
        if cache_path is None:
            cache_path = Path(__file__).parent / "outputs" / "validated_pairs_cache.json"
        return Path(cache_path).exists()

    def _get_company_info(self, ticker: str) -> Dict[str, str]:
        """Get GICS info for a company."""
        return self.gics_data.get(ticker, {
            'sector': 'Unknown',
            'sub_industry': 'Unknown'
        })

    def _validate_pair(self, pair: CompanyPair) -> CompanyPair:
        """Validate a pair by checking connector counts in Neo4j."""
        if not self.discovery:
            return pair

        if pair.pair_category == 'intra_doc':
            # intra_doc: Same company, same year, different pages/chunks
            stats = self.discovery.get_intra_doc_stats(pair.org1, pair.org1_year)
            pair.hard_connectors = stats.get('hard', 0)
            pair.medium_connectors = stats.get('medium', 0)
            pair.easy_connectors = 0  # Not tracked for intra-doc
            pair.total_connectors = stats.get('total', 0)
            pair.relationship_changes = stats.get('medium_plus_changed', 0)
        elif pair.is_same_company:
            # cross_year: Same company, different years - use temporal stats
            stats = self.discovery.get_temporal_stats(pair.org1, pair.org1_year, pair.org2_year)
            pair.hard_connectors = stats.get('hard', 0)
            pair.medium_connectors = stats.get('medium', 0)
            pair.easy_connectors = 0  # Not tracked for temporal
            pair.total_connectors = stats.get('total', 0)
            pair.relationship_changes = stats.get('medium_plus_changed', 0)
        else:
            # Type 1: Cross-company pairs
            stats = self.discovery.get_connector_stats(pair.org1, pair.org2)
            pair.hard_connectors = stats.get('hard', 0)
            pair.medium_connectors = stats.get('medium', 0)
            pair.easy_connectors = stats.get('easy', 0)
            pair.total_connectors = stats.get('total', 0)

        return pair

    def generate_within_subindustry_pairs(
        self,
        min_companies: int = 2,
        year: int = 2024,
        tickers_filter: Optional[List[str]] = None
    ) -> List[CompanyPair]:
        """
        Generate pairs of companies within the same sub-industry (Type 1).
        These are the highest quality pairs for comparative QA.

        Args:
            min_companies: Minimum companies in sub-industry to consider
            year: Year for both companies' evidence (same year comparison)

        Returns:
            List of CompanyPair objects
        """
        sub_industries = get_companies_by_sub_industry(self.gics_data)
        pairs = []
        tickers_filter_set = set(tickers_filter) if tickers_filter else None

        for sub_ind, tickers in sub_industries.items():
            if tickers_filter_set is not None:
                tickers = [t for t in tickers if t in tickers_filter_set]
            if len(tickers) < min_companies:
                continue

            # Get sector for this sub-industry
            sector = self._get_company_info(tickers[0])['sector']
            connector_types = get_sector_connector_types(sector)

            # Generate all pairs within this sub-industry
            for org1, org2 in itertools.combinations(sorted(tickers), 2):
                pair = CompanyPair(
                    org1=org1,
                    org2=org2,
                    sector=sector,
                    sub_industry1=sub_ind,
                    sub_industry2=sub_ind,
                    pair_type='within_subindustry',
                    connector_types=connector_types,
                    org1_year=year,
                    org2_year=year,
                    pair_category='cross_company'
                )

                if self.validate_connectors:
                    pair = self._validate_pair(pair)

                pairs.append(pair)

        logger.info(f"Generated {len(pairs)} within-subindustry pairs for year {year}")
        return pairs

    def generate_temporal_pairs(
        self,
        year1: int = 2022,
        year2: int = 2024,
        tickers: Optional[List[str]] = None
    ) -> List[CompanyPair]:
        """
        Generate Type 2 pairs: same company, different years.
        These are valuable for temporal trend analysis.

        Args:
            year1: Earlier year
            year2: Later year
            tickers: Optional list of tickers to consider (default: all SP100)

        Returns:
            List of CompanyPair objects for temporal comparison
        """
        if tickers is None:
            tickers = list(self.gics_data.keys())

        pairs = []

        for ticker in sorted(tickers):
            info = self._get_company_info(ticker)
            sector = info['sector']
            sub_ind = info['sub_industry']
            connector_types = get_sector_connector_types(sector)

            pair = CompanyPair(
                org1=ticker,
                org2=ticker,  # Same company
                sector=sector,
                sub_industry1=sub_ind,
                sub_industry2=sub_ind,
                pair_type='cross_year',
                connector_types=connector_types,
                org1_year=year1,
                org2_year=year2,
                pair_category='cross_year'
            )

            if self.validate_connectors:
                pair = self._validate_pair(pair)

            pairs.append(pair)

        logger.info(f"Generated {len(pairs)} temporal pairs ({year1} vs {year2})")
        return pairs

    def generate_intra_doc_pairs(
        self,
        years: Optional[List[int]] = None,
        tickers: Optional[List[str]] = None
    ) -> List[CompanyPair]:
        """
        Generate Type 3 pairs: same company, same year, different pages/chunks (intra-document).
        These are valuable for analyzing how companies discuss the same topic in different contexts.

        Args:
            years: List of years to generate pairs for (default: [2022, 2023, 2024])
            tickers: Optional list of tickers to consider (default: all SP100)

        Returns:
            List of CompanyPair objects for intra-document comparison
        """
        if years is None:
            years = [2022, 2023, 2024]
        if tickers is None:
            tickers = list(self.gics_data.keys())

        pairs = []

        for ticker in sorted(tickers):
            info = self._get_company_info(ticker)
            sector = info['sector']
            sub_ind = info['sub_industry']
            connector_types = get_sector_connector_types(sector)

            for year in years:
                pair = CompanyPair(
                    org1=ticker,
                    org2=ticker,  # Same company
                    sector=sector,
                    sub_industry1=sub_ind,
                    sub_industry2=sub_ind,
                    pair_type='intra_doc',
                    connector_types=connector_types,
                    org1_year=year,
                    org2_year=year,  # Same year
                    pair_category='intra_doc'
                )

                if self.validate_connectors:
                    pair = self._validate_pair(pair)

                pairs.append(pair)

        logger.info(f"Generated {len(pairs)} intra-doc pairs for years {years}")
        return pairs

    def generate_cross_company_pairs_all_years(
        self,
        years: Optional[List[int]] = None
    ) -> List[CompanyPair]:
        """
        Generate Type 1 pairs for all specified years.

        Args:
            years: List of years to generate pairs for (default: [2022, 2023, 2024])

        Returns:
            List of CompanyPair objects across all years
        """
        if years is None:
            years = [2022, 2023, 2024]

        all_pairs = []
        for year in years:
            # Within sub-industry for this year
            pairs = self.generate_within_subindustry_pairs(year=year)
            all_pairs.extend(pairs)

            # Within sector (different sub-industry) for this year
            sector_pairs = self.generate_within_sector_pairs_for_year(year=year)
            all_pairs.extend(sector_pairs)

        logger.info(f"Generated {len(all_pairs)} Type 1 pairs across years {years}")
        return all_pairs

    def generate_within_sector_pairs_for_year(
        self,
        year: int = 2024,
        exclude_same_subindustry: bool = True,
        tickers_filter: Optional[List[str]] = None
    ) -> List[CompanyPair]:
        """
        Generate pairs within the same sector but different sub-industries for a specific year.

        Args:
            year: Year for both companies
            exclude_same_subindustry: If True, exclude pairs already in within_subindustry

        Returns:
            List of CompanyPair objects
        """
        sectors = get_companies_by_sector(self.gics_data)
        pairs = []
        tickers_filter_set = set(tickers_filter) if tickers_filter else None

        for sector, tickers in sectors.items():
            if tickers_filter_set is not None:
                tickers = [t for t in tickers if t in tickers_filter_set]
            if len(tickers) < 2:
                continue

            connector_types = get_sector_connector_types(sector)

            for org1, org2 in itertools.combinations(sorted(tickers), 2):
                info1 = self._get_company_info(org1)
                info2 = self._get_company_info(org2)

                # Skip if same sub-industry (already covered)
                if exclude_same_subindustry and info1['sub_industry'] == info2['sub_industry']:
                    continue

                pair = CompanyPair(
                    org1=org1,
                    org2=org2,
                    sector=sector,
                    sub_industry1=info1['sub_industry'],
                    sub_industry2=info2['sub_industry'],
                    pair_type='within_sector',
                    connector_types=connector_types,
                    org1_year=year,
                    org2_year=year,
                    pair_category='cross_company'
                )

                if self.validate_connectors:
                    pair = self._validate_pair(pair)

                pairs.append(pair)

        logger.info(f"Generated {len(pairs)} within-sector pairs for year {year}")
        return pairs

    def generate_within_sector_pairs(
        self,
        exclude_same_subindustry: bool = True
    ) -> List[CompanyPair]:
        """
        Generate pairs of companies within the same sector but different sub-industries.

        Args:
            exclude_same_subindustry: If True, exclude pairs already in within_subindustry

        Returns:
            List of CompanyPair objects
        """
        sectors = get_companies_by_sector(self.gics_data)
        pairs = []

        for sector, tickers in sectors.items():
            if len(tickers) < 2:
                continue

            connector_types = get_sector_connector_types(sector)

            # Generate all pairs within this sector
            for org1, org2 in itertools.combinations(sorted(tickers), 2):
                info1 = self._get_company_info(org1)
                info2 = self._get_company_info(org2)

                # Skip if same sub-industry (already covered)
                if exclude_same_subindustry and info1['sub_industry'] == info2['sub_industry']:
                    continue

                pair = CompanyPair(
                    org1=org1,
                    org2=org2,
                    sector=sector,
                    sub_industry1=info1['sub_industry'],
                    sub_industry2=info2['sub_industry'],
                    pair_type='within_sector',
                    connector_types=connector_types
                )

                if self.validate_connectors:
                    pair = self._validate_pair(pair)

                pairs.append(pair)

        logger.info(f"Generated {len(pairs)} within-sector pairs")
        return pairs

    def generate_cross_sector_pairs(
        self,
        sectors: Optional[List[str]] = None,
        limit_per_sector_pair: int = 5
    ) -> List[CompanyPair]:
        """
        Generate pairs of companies from different sectors.
        Use sparingly - these are lower quality for domain-specific QA.

        Args:
            sectors: List of sectors to consider (default: all)
            limit_per_sector_pair: Max pairs per sector combination

        Returns:
            List of CompanyPair objects
        """
        all_sectors = get_companies_by_sector(self.gics_data)
        if sectors:
            all_sectors = {k: v for k, v in all_sectors.items() if k in sectors}

        sector_list = list(all_sectors.keys())
        pairs = []

        for sector1, sector2 in itertools.combinations(sorted(sector_list), 2):
            tickers1 = all_sectors[sector1]
            tickers2 = all_sectors[sector2]

            # Use generic connector types for cross-sector
            connector_types = ['RISK_FACTOR', 'MACRO_CONDITION', 'REGULATORY_REQUIREMENT']

            # Generate limited pairs
            count = 0
            for org1 in sorted(tickers1):
                if count >= limit_per_sector_pair:
                    break
                for org2 in sorted(tickers2):
                    if count >= limit_per_sector_pair:
                        break

                    info1 = self._get_company_info(org1)
                    info2 = self._get_company_info(org2)

                    pair = CompanyPair(
                        org1=org1,
                        org2=org2,
                        sector=f"{sector1} x {sector2}",
                        sub_industry1=info1['sub_industry'],
                        sub_industry2=info2['sub_industry'],
                        pair_type='cross_sector',
                        connector_types=connector_types
                    )

                    if self.validate_connectors:
                        pair = self._validate_pair(pair)
                        if pair.has_sufficient_connectors:
                            pairs.append(pair)
                            count += 1
                    else:
                        pairs.append(pair)
                        count += 1

        logger.info(f"Generated {len(pairs)} cross-sector pairs")
        return pairs

    def generate_all_pairs(
        self,
        include_cross_sector: bool = False,
        include_temporal: bool = True,
        year: int = 2024
    ) -> List[CompanyPair]:
        """
        Generate all pairs (within-subindustry + within-sector + temporal).

        Args:
            include_cross_sector: If True, also include cross-sector pairs
            include_temporal: If True, include Type 2 temporal pairs
            year: Year for Type 1 pairs (default: 2024)

        Returns:
            List of all CompanyPair objects
        """
        all_pairs = []

        # Type 1: Within sub-industry (highest quality)
        all_pairs.extend(self.generate_within_subindustry_pairs(year=year))

        # Type 1: Within sector (good quality)
        all_pairs.extend(self.generate_within_sector_pairs_for_year(year=year))

        # Type 2: Temporal pairs (same company, different years)
        if include_temporal:
            all_pairs.extend(self.generate_temporal_pairs(year1=2022, year2=2024))

        # Cross-sector (optional, lower quality)
        if include_cross_sector:
            all_pairs.extend(self.generate_cross_sector_pairs())

        logger.info(f"Generated {len(all_pairs)} total pairs")
        return all_pairs

    def get_validated_pairs(
        self,
        min_hard_connectors: int = 1,
        min_total_connectors: int = 5,
        include_temporal: bool = True,
        year: int = 2024
    ) -> List[CompanyPair]:
        """
        Get pairs that pass validation criteria.

        Args:
            min_hard_connectors: Minimum hard (IDF > 5.0) connectors required
            min_total_connectors: Minimum total connectors required
            include_temporal: If True, include Type 2 temporal pairs
            year: Year for Type 1 pairs

        Returns:
            List of validated CompanyPair objects
        """
        all_pairs = self.generate_all_pairs(include_temporal=include_temporal, year=year)

        validated = [
            p for p in all_pairs
            if p.hard_connectors >= min_hard_connectors
            and p.total_connectors >= min_total_connectors
        ]

        logger.info(f"Validated {len(validated)} / {len(all_pairs)} pairs")
        return validated

    def save_pairs_to_json(
        self,
        pairs: List[CompanyPair],
        output_path: str
    ) -> None:
        """
        Save pairs to a JSON file.

        Args:
            pairs: List of CompanyPair objects to save
            output_path: Path to output JSON file
        """
        import json
        from pathlib import Path

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = [p.to_dict() for p in pairs]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(pairs)} pairs to {output_path}")

    def generate_all_pairs_from_config(
        self,
        cross_company_years_override: Optional[List[int]] = None,
        cross_year_combinations_override: Optional[List[List[int]]] = None,
        intra_doc_years_override: Optional[List[int]] = None,
        include_cross_sector_override: Optional[bool] = None,
        min_companies_override: Optional[int] = None,
        tickers_override: Optional[List[str]] = None
    ) -> List[CompanyPair]:
        """
        Generate all pairs based on config.yaml settings.

        Uses:
            - config['cross_company_years']: List of years for cross-company pairs
            - config['cross_year_combinations']: List of [year1, year2] for cross-year pairs
            - config['intra_doc_years']: List of years for intra-doc pairs
            - config['include_cross_sector']: Whether to include cross-sector pairs

        Returns:
            List of all generated CompanyPair objects (before validation)
        """
        all_pairs = []

        # Get config values
        cross_company_years = cross_company_years_override if cross_company_years_override is not None else self.config.get('cross_company_years', [2024])
        cross_year_combinations = cross_year_combinations_override if cross_year_combinations_override is not None else self.config.get('cross_year_combinations', [[2022, 2024]])
        intra_doc_years = intra_doc_years_override if intra_doc_years_override is not None else self.config.get('intra_doc_years', [])
        include_cross_sector = include_cross_sector_override if include_cross_sector_override is not None else self.config.get('include_cross_sector', False)
        min_companies = min_companies_override if min_companies_override is not None else self.config.get('min_companies_per_subindustry', 2)

        logger.info(f"Generating pairs from config:")
        logger.info(f"  cross_company years: {cross_company_years}")
        logger.info(f"  cross_year combinations: {cross_year_combinations}")
        logger.info(f"  intra_doc years: {intra_doc_years}")
        logger.info(f"  Include cross-sector: {include_cross_sector}")

        # Type 1: Same year, cross-company pairs
        for year in cross_company_years:
            # Within sub-industry
            subind_pairs = self.generate_within_subindustry_pairs(
                min_companies=min_companies,
                year=year,
                tickers_filter=tickers_override
            )
            all_pairs.extend(subind_pairs)

            # Within sector (different sub-industry)
            sector_pairs = self.generate_within_sector_pairs_for_year(
                year=year,
                tickers_filter=tickers_override
            )
            all_pairs.extend(sector_pairs)

        # Type 2: Same company, different years (temporal pairs)
        for year1, year2 in cross_year_combinations:
            temporal_pairs = self.generate_temporal_pairs(year1=year1, year2=year2, tickers=tickers_override)
            all_pairs.extend(temporal_pairs)

        # Type 3: Same company, same year, different pages (intra-doc pairs)
        if intra_doc_years:
            intra_doc_pairs = self.generate_intra_doc_pairs(years=intra_doc_years, tickers=tickers_override)
            all_pairs.extend(intra_doc_pairs)

        # Cross-sector (optional)
        if include_cross_sector and not tickers_override:
            cross_pairs = self.generate_cross_sector_pairs()
            all_pairs.extend(cross_pairs)

        logger.info(f"Generated {len(all_pairs)} total pairs from config")
        return all_pairs

    def get_validated_pairs_from_config(self) -> List[CompanyPair]:
        """
        Generate and validate pairs based on config.yaml settings.

        Uses:
            - config['validation']['min_hard_connectors']
            - config['validation']['min_total_connectors']

        Returns:
            List of validated CompanyPair objects
        """
        # Get validation thresholds from config
        validation_config = self.config.get('validation', {})
        min_hard = validation_config.get('min_hard_connectors', 1)
        min_total = validation_config.get('min_total_connectors', 5)

        # Generate all pairs
        all_pairs = self.generate_all_pairs_from_config()

        # Filter to validated pairs
        validated = [
            p for p in all_pairs
            if p.hard_connectors >= min_hard
            and p.total_connectors >= min_total
        ]

        logger.info(f"Validated {len(validated)} / {len(all_pairs)} pairs")
        logger.info(f"  (min_hard={min_hard}, min_total={min_total})")
        return validated


def main():
    """Generate pairs using config.yaml settings."""
    import json
    from pathlib import Path
    from collections import defaultdict

    # Load config
    config = load_config()
    print("=" * 80)
    print("Company Pair Generation (Config-Based)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  cross_company years: {config.get('cross_company_years', [])}")
    print(f"  cross_year combinations: {config.get('cross_year_combinations', [])}")
    print(f"  intra_doc years: {config.get('intra_doc_years', [])}")
    print(f"  Include cross-sector: {config.get('include_cross_sector', False)}")
    print(f"  Validation: min_hard={config.get('validation', {}).get('min_hard_connectors', 1)}, "
          f"min_total={config.get('validation', {}).get('min_total_connectors', 5)}")

    # Create output directory from config
    output_config = config.get('output', {})
    output_dir = Path(__file__).parent / output_config.get('directory', 'outputs')
    output_dir.mkdir(exist_ok=True)

    generator = PairGenerator(config=config, validate_connectors=True)

    try:
        # ===== Generate and Validate All Pairs =====
        print("\n" + "=" * 60)
        print("Generating Pairs from Config")
        print("=" * 60)

        all_validated = generator.get_validated_pairs_from_config()

        # ===== Summary by Type =====
        print("\n" + "=" * 60)
        print("Summary by Temporal Type")
        print("=" * 60)

        cross_company_pairs = [p for p in all_validated if p.pair_category == 'cross_company']
        cross_year_pairs = [p for p in all_validated if p.pair_category == 'cross_year']

        print(f"\nTotal validated pairs: {len(all_validated)}")
        print(f"  Type 1 (same year, cross-company): {len(cross_company_pairs)}")
        print(f"  Type 2 (same company, temporal): {len(cross_year_pairs)}")

        # ===== Type 1 Breakdown by Year =====
        print("\n" + "-" * 40)
        print("Type 1 Breakdown by Year:")
        print("-" * 40)
        by_year = defaultdict(list)
        for p in cross_company_pairs:
            by_year[p.org1_year].append(p)

        for year, pairs in sorted(by_year.items()):
            print(f"  Year {year}: {len(pairs)} pairs")

        # ===== Type 2 Breakdown by Year Combination =====
        print("\n" + "-" * 40)
        print("Type 2 Breakdown by Year Combination:")
        print("-" * 40)
        by_combo = defaultdict(list)
        for p in cross_year_pairs:
            combo = f"{p.org1_year} vs {p.org2_year}"
            by_combo[combo].append(p)

        for combo, pairs in sorted(by_combo.items()):
            with_changes = sum(1 for p in pairs if p.relationship_changes > 0)
            print(f"  {combo}: {len(pairs)} pairs ({with_changes} with relationship changes)")

        # ===== Top Pairs =====
        print("\n" + "-" * 40)
        print("Top 10 Pairs by Hard Connector Count:")
        print("-" * 40)
        top_pairs = sorted(all_validated, key=lambda p: -p.hard_connectors)[:10]
        for p in top_pairs:
            if p.is_same_company:
                desc = f"{p.org1} ({p.org1_year} vs {p.org2_year})"
            else:
                desc = f"{p.org1}({p.org1_year}) vs {p.org2}({p.org2_year})"
            print(f"  {desc}: {p.hard_connectors} hard, {p.total_connectors} total ({p.pair_category})")

        # ===== Save Outputs =====
        print("\n" + "=" * 60)
        print("Saving Outputs")
        print("=" * 60)

        # Get output file names from config
        output_files = output_config.get('files', {})

        # Save all validated pairs
        all_pairs_file = output_dir / output_files.get('all_pairs', 'validated_pairs.json')
        generator.save_pairs_to_json(all_validated, str(all_pairs_file))

        # Save Type 1 pairs
        cross_company_file = output_dir / output_files.get('cross_company_pairs', 'cross_company_pairs.json')
        generator.save_pairs_to_json(cross_company_pairs, str(cross_company_file))

        # Save Type 2 pairs
        cross_year_file = output_dir / output_files.get('cross_year_pairs', 'cross_year_pairs.json')
        generator.save_pairs_to_json(cross_year_pairs, str(cross_year_file))

        print(f"\nOutputs saved to: {output_dir}")
        print(f"  {all_pairs_file.name}: {len(all_validated)} pairs")
        print(f"  {cross_company_file.name}: {len(cross_company_pairs)} pairs")
        print(f"  {cross_year_file.name}: {len(cross_year_pairs)} pairs")

    finally:
        generator.close()


if __name__ == "__main__":
    main()
