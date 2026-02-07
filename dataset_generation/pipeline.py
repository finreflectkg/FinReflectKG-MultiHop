"""
Multi-hop QA Generation Pipeline

Flexible orchestrator with easy parameter overrides for testing and production.

Usage Examples:
    # Quick test: 10 questions between NVDA and AMD, 2024 only, 2-hop
    python pipeline.py --test --orgs NVDA,AMD --year 2024 --hops 2 --num 10

    # Test 3-hop cross_company_type1 only
    python pipeline.py --test --hops 3 --pattern cross_company_type1 --num 5

    # Full production run from config
    python pipeline.py --full

    # Custom run: specific sector, both hops
    python pipeline.py --sector "Semiconductors" --hops 2,3 --num 50

    # Python API:
    from pipeline import Pipeline
    p = Pipeline()
    results = p.run(orgs=['NVDA', 'AMD'], year=2024, hops=[2], num_questions=10)
"""

import argparse
import itertools
import json
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from pair_generator import PairGenerator, CompanyPair, load_config
from qa_generator import QAGenerator, QAItem
from connector_discovery import ConnectorDiscovery
from gics_loader import get_sector_connector_types

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Pipeline configuration with easy overrides.

    Any parameter set here overrides the config.yaml value.
    Set to None to use config.yaml default.
    """
    # Company filters
    orgs: Optional[List[str]] = None  # e.g., ['NVDA', 'AMD'] - if set, only these orgs
    org_pairs: Optional[List[tuple]] = None  # e.g., [('NVDA', 'AMD')] - specific pairs
    sector: Optional[str] = None  # e.g., 'Semiconductors' - filter by sector

    # Year filters
    years: Optional[List[int]] = None  # e.g., [2024] - only these years
    year_combinations: Optional[List[tuple]] = None  # e.g., [(2022, 2024)] for temporal
    skip_temporal: bool = False  # Skip cross_year (temporal) pairs entirely
    skip_cross_company: bool = False  # Skip cross_company pairs entirely

    # Hop configuration
    hops: Optional[List[int]] = None  # e.g., [2, 3] or [2] or [3]
    patterns_2hop: Optional[List[str]] = None  # e.g., ['cross_company', 'cross_year']
    patterns_3hop: Optional[List[str]] = None  # e.g., ['cross_company_type1', 'entity_extension']

    # Generation limits
    num_questions: Optional[int] = None  # Total questions to generate
    questions_per_pair: Optional[int] = None  # Questions per company pair (None = no limit)
    max_pairs: Optional[int] = None  # Limit number of pairs to process

    # IDF thresholds (override config)
    min_idf: Optional[float] = None  # Minimum IDF for connectors

    # Connector type configuration
    connector_types: Optional[List[str]] = None  # Manual override (e.g., ["COMP"])
    connector_mode: str = "all"  # "quantitative", "qualitative", or "all"

    # Output
    output_dir: Optional[str] = None  # Output directory
    output_prefix: str = "qa"  # Output file prefix

    # LLM
    llm_model: Optional[str] = None  # Override LLM model
    # 3-hop method
    three_hop_method: str = "default"  # "default" or "causal"

    # Parallelism
    max_workers: int = 20  # Max parallel workers for LLM requests

    # Cache options
    use_cache: bool = False  # Load pairs from cache (skip Neo4j validation queries)
    build_cache: bool = False  # Build cache after generating pairs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Pipeline:
    """
    Multi-hop QA Generation Pipeline.

    Example:
        # Quick test
        p = Pipeline()
        results = p.run(
            orgs=['NVDA', 'AMD'],
            year=2024,
            hops=[2],
            num_questions=10
        )

        # Full run from config
        p = Pipeline()
        results = p.run_from_config()
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with config."""
        self.base_config = load_config(config_path) if config_path else load_config()
        self.pair_generator = None
        self.qa_generator = None
        self.discovery = None

    def _init_components(self, pipeline_config: PipelineConfig):
        """Initialize pipeline components."""
        # Close existing if any
        self._cleanup()

        logger.info("Initializing pipeline components...")

        # Check if we'll need to generate pairs (if not using cache or cache doesn't exist)
        need_validation = not (pipeline_config.use_cache and PairGenerator.cache_exists())
        self.pair_generator = PairGenerator(config=self.base_config, validate_connectors=need_validation)

        self.qa_generator = QAGenerator(config=self.base_config, llm_model=pipeline_config.llm_model)
        self.discovery = ConnectorDiscovery()
        logger.info("Pipeline components initialized")

    def _cleanup(self):
        """Clean up resources."""
        if self.pair_generator:
            self.pair_generator.close()
            self.pair_generator = None
        if self.qa_generator:
            self.qa_generator.close()
            self.qa_generator = None
        if self.discovery:
            self.discovery.close()
            self.discovery = None

    def _filter_pairs(
        self,
        pairs: List[CompanyPair],
        config: PipelineConfig
    ) -> List[CompanyPair]:
        """Filter pairs based on pipeline config."""
        filtered = pairs

        # Filter by specific orgs (BOTH orgs must be in the set)
        if config.orgs:
            org_set = set(config.orgs)
            filtered = [p for p in filtered if p.org1 in org_set and p.org2 in org_set]
            logger.info(f"Filtered to orgs {config.orgs}: {len(filtered)} pairs")

        # Filter by specific org pairs
        if config.org_pairs:
            pair_set = {tuple(sorted(pair)) for pair in config.org_pairs}
            filtered = [p for p in filtered if tuple(sorted([p.org1, p.org2])) in pair_set]
            logger.info(f"Filtered to specific pairs: {len(filtered)} pairs")

        # Filter by sector
        if config.sector:
            filtered = [p for p in filtered if p.sector == config.sector]
            logger.info(f"Filtered to sector {config.sector}: {len(filtered)} pairs")

        # Filter by years
        if config.years:
            year_set = set(config.years)
            filtered = [p for p in filtered if p.org1_year in year_set and p.org2_year in year_set]
            logger.info(f"Filtered to years {config.years}: {len(filtered)} pairs")

        # Skip temporal pairs
        if config.skip_temporal:
            filtered = [p for p in filtered if p.pair_category != 'cross_year']
            logger.info(f"Skipped temporal pairs: {len(filtered)} pairs remaining")

        # Skip cross-company pairs
        if config.skip_cross_company:
            filtered = [p for p in filtered if p.pair_category != 'cross_company']
            logger.info(f"Skipped cross-company pairs: {len(filtered)} pairs remaining")

        # Limit pairs
        if config.max_pairs and len(filtered) > config.max_pairs:
            filtered = filtered[:config.max_pairs]
            logger.info(f"Limited to {config.max_pairs} pairs")

        return filtered

    def _get_enabled_hops_and_patterns(
        self,
        config: PipelineConfig
    ) -> Dict[int, List[str]]:
        """Get enabled hops and patterns from config or overrides."""
        result = {}

        # Determine hops to use
        if config.hops:
            hops = config.hops
        else:
            hop_settings = self.base_config.get('hop_settings', {})
            hops = hop_settings.get('enabled_hops', [2])

        # Get patterns for each hop
        for hop in hops:
            if hop == 2:
                if config.patterns_2hop:
                    result[2] = config.patterns_2hop
                else:
                    hop_settings = self.base_config.get('hop_settings', {})
                    two_hop = hop_settings.get('two_hop', {})
                    patterns = [p['name'] for p in two_hop.get('patterns', []) if p.get('enabled', True)]
                    result[2] = patterns if patterns else ['cross_company', 'cross_year', 'intra_doc']

            elif hop == 3:
                if config.patterns_3hop:
                    result[3] = config.patterns_3hop
                elif config.three_hop_method == "causal":
                    result[3] = ["both_orgs_originator", "both_orgs_recipient", "both_orgs_separate"]
                else:
                    hop_settings = self.base_config.get('hop_settings', {})
                    three_hop = hop_settings.get('three_hop', {})
                    patterns = [p['name'] for p in three_hop.get('patterns', []) if p.get('enabled', True)]
                    result[3] = patterns if patterns else [
                        'cross_company_type1', 'cross_year_type1', 'intra_doc_type1',
                        'cross_company_type2', 'cross_year_type2', 'intra_doc_type2'
                    ]

        return result

    def run(
        self,
        # Company filters
        orgs: Optional[List[str]] = None,
        org_pairs: Optional[List[tuple]] = None,
        sector: Optional[str] = None,
        # Year filters
        year: Optional[int] = None,  # Shortcut for years=[year]
        years: Optional[List[int]] = None,
        skip_temporal: bool = False,
        skip_cross_company: bool = False,
        # Hop configuration
        hops: Optional[Union[int, List[int]]] = None,  # e.g., 2 or [2, 3]
        pattern: Optional[str] = None,  # Shortcut for single pattern
        patterns_2hop: Optional[List[str]] = None,
        patterns_3hop: Optional[List[str]] = None,
        # Limits
        num_questions: Optional[int] = None,
        questions_per_pair: Optional[int] = None,  # None = no per-pair limit
        max_pairs: Optional[int] = None,
        # Connector configuration
        min_idf: Optional[float] = None,
        connector_types: Optional[List[str]] = None,
        connector_mode: str = "all",  # "quantitative", "qualitative", or "all"
        # Other
        llm_model: Optional[str] = None,
        three_hop_method: str = "default",
        max_workers: int = 20,
        output_dir: Optional[str] = None,
        output_prefix: str = "qa",
        save_results: bool = True,
        # Cache options
        use_cache: bool = False,
        build_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Run the pipeline with flexible overrides.

        Args:
            orgs: Filter to specific companies, e.g., ['NVDA', 'AMD']
            org_pairs: Specific pairs to process, e.g., [('NVDA', 'AMD')]
            sector: Filter by sector, e.g., 'Semiconductors'
            year: Single year to filter (shortcut for years=[year])
            years: List of years to filter
            skip_temporal: Skip cross_year (temporal) pairs
            skip_cross_company: Skip cross_company pairs
            hops: Which hops to generate, e.g., 2 or [2, 3]
            pattern: Single pattern (for quick testing)
            patterns_2hop: 2-hop patterns to use
            patterns_3hop: 3-hop patterns to use
            num_questions: Total question limit
            questions_per_pair: Questions per company pair
            max_pairs: Limit number of pairs
            min_idf: Minimum IDF threshold
            connector_types: Manual override of connector types (e.g., ['COMP', 'RISK_FACTOR'])
            connector_mode: Connector type mode - 'quantitative' (FIN_METRIC, SEGMENT, ECON_IND),
                           'qualitative' (PRODUCT, FIN_INST, etc.), or 'all' (both)
            llm_model: Override LLM model
            max_workers: Max parallel workers for LLM requests (default: 20)
            output_dir: Output directory
            output_prefix: Output file prefix
            save_results: Whether to save results to files
            use_cache: Load pairs from cache (skips Neo4j validation queries)
            build_cache: Save pairs to cache after generation

        Returns:
            Dict with results by pattern and metadata
        """
        # Build config from parameters
        config = PipelineConfig(
            orgs=orgs,
            org_pairs=org_pairs,
            sector=sector,
            years=years or ([year] if year else None),
            skip_temporal=skip_temporal,
            skip_cross_company=skip_cross_company,
            hops=[hops] if isinstance(hops, int) else hops,
            patterns_2hop=patterns_2hop or ([pattern] if pattern and pattern in ['cross_company', 'cross_year', 'intra_doc'] else None),
            patterns_3hop=patterns_3hop or ([pattern] if pattern and pattern in [
                # Type 1 patterns (Connector Chain)
                'cross_company_type1', 'cross_year_type1', 'intra_doc_type1',
                # Type 2 patterns (Multi-Branch)
                'cross_company_type2', 'cross_year_type2', 'intra_doc_type2',
                # Causal patterns
                'both_orgs_originator', 'both_orgs_recipient', 'both_orgs_separate'
            ] else None),
            num_questions=num_questions,
            questions_per_pair=questions_per_pair,
            max_pairs=max_pairs,
            min_idf=min_idf,
            connector_types=connector_types,
            connector_mode=connector_mode,
            llm_model=llm_model,
            three_hop_method=three_hop_method,
            max_workers=max_workers,
            output_dir=output_dir,
            output_prefix=output_prefix,
            use_cache=use_cache,
            build_cache=build_cache
        )

        return self._run_with_config(config, save_results)

    def _run_with_config(
        self,
        config: PipelineConfig,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run pipeline with PipelineConfig."""
        logger.info("=" * 80)
        logger.info("Starting Multi-hop QA Generation Pipeline")
        logger.info("=" * 80)
        logger.info(f"Config overrides: {config.to_dict()}")
        logger.info(f"DEBUG: config.sector = {repr(config.sector)}")

        try:
            # Initialize components
            self._init_components(config)

            # Get all pairs - use cache if available and requested
            if config.use_cache and PairGenerator.cache_exists():
                logger.info("Loading pairs from cache (skipping ~1600 Neo4j validation queries)...")
                all_pairs = PairGenerator.load_pairs_from_cache()
                logger.info(f"Loaded {len(all_pairs)} pairs from cache")
            else:
                if config.use_cache:
                    logger.warning("--use-cache specified but no cache found. Generating pairs...")

                logger.info("Generating company pairs (querying Neo4j for validation)...")
                cross_company_years_override = config.years if config.years else None
                if config.year_combinations is not None:
                    cross_year_combinations_override = config.year_combinations
                elif config.years and len(config.years) > 1:
                    cross_year_combinations_override = [list(c) for c in itertools.combinations(sorted(config.years), 2)]
                elif config.years and len(config.years) == 1:
                    cross_year_combinations_override = []
                else:
                    cross_year_combinations_override = None

                all_pairs = list(
                    self.pair_generator.generate_all_pairs_from_config(
                        cross_company_years_override=cross_company_years_override,
                        cross_year_combinations_override=cross_year_combinations_override,
                        tickers_override=config.orgs,
                    )
                )
                logger.info(f"Generated {len(all_pairs)} total pairs")

                # Save to cache if requested
                if config.build_cache or config.use_cache:
                    self.pair_generator.save_pairs_to_cache(all_pairs)
                    logger.info("Pairs cached for future runs (use --use-cache next time)")

            # Filter pairs
            pairs = self._filter_pairs(all_pairs, config)

            if not pairs:
                logger.warning("No pairs match the filter criteria!")
                return {'results': {}, 'metadata': {'total_questions': 0}}

            # Set connector types based on mode or manual override
            if config.connector_types:
                # Manual override takes precedence
                for pair in pairs:
                    pair.connector_types = config.connector_types
                logger.info(f"Manual connector types override: {config.connector_types}")
            elif config.connector_mode in ('quantitative', 'qualitative'):
                # Use sector-specific types based on mode
                for pair in pairs:
                    sector = getattr(pair, 'sector', None)
                    if sector:
                        pair.connector_types = get_sector_connector_types(sector, mode=config.connector_mode)
                    else:
                        # Fallback to default types for the mode
                        from gics_loader import DEFAULT_QUANTITATIVE_TYPES, DEFAULT_QUALITATIVE_TYPES
                        pair.connector_types = DEFAULT_QUANTITATIVE_TYPES if config.connector_mode == 'quantitative' else DEFAULT_QUALITATIVE_TYPES
                logger.info(f"Using {config.connector_mode} connector types (sector-specific)")

            # Get enabled hops and patterns
            hops_patterns = self._get_enabled_hops_and_patterns(config)
            logger.info(f"Enabled hops and patterns: {hops_patterns}")

            # Separate pairs by type
            cross_company_pairs = [p for p in pairs if p.pair_category == 'cross_company']
            cross_year_pairs = [p for p in pairs if p.pair_category == 'cross_year']
            intra_doc_pairs = [p for p in pairs if p.pair_category == 'intra_doc']

            # Shuffle all pair lists for diversity (avoid alphabetical bias)
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(cross_company_pairs)
            random.shuffle(cross_year_pairs)
            random.shuffle(intra_doc_pairs)

            logger.info(f"cross_company pairs: {len(cross_company_pairs)} (shuffled)")
            logger.info(f"cross_year pairs: {len(cross_year_pairs)} (shuffled)")
            logger.info(f"intra_doc pairs: {len(intra_doc_pairs)} (shuffled)")

            # Determine effective per-pair limit:
            # - If --per-pair specified, use it
            # - Else default to 5 (reasonable limit per pair/triplet to avoid overshoot with parallel workers)
            # Note: Previously used --num as per-pair, but this caused massive overshoot
            #       because each worker could generate up to --num items before stop flag takes effect
            if config.questions_per_pair:
                effective_per_pair = config.questions_per_pair
            else:
                effective_per_pair = 5  # Default: 5 questions per pair/triplet

            # Generate QA
            results = {}
            total_generated = 0
            questions_limit = config.num_questions

            causal_3hop_done = False
            causal_patterns = {'both_orgs_originator', 'both_orgs_recipient', 'both_orgs_separate'}
            cross_company_patterns = {'cross_company', 'cross_company_type1', 'cross_company_type2'} | causal_patterns
            cross_year_patterns = {'cross_year', 'cross_year_type1', 'cross_year_type2'}
            intra_doc_patterns = {'intra_doc', 'intra_doc_type1', 'intra_doc_type2'}
            for hop_count, patterns in hops_patterns.items():
                for pattern_name in patterns:
                    # Check if we've hit the limit
                    if questions_limit and total_generated >= questions_limit:
                        logger.info(f"Reached question limit ({questions_limit}), stopping")
                        break

                    # Determine which pairs to use
                    if pattern_name in intra_doc_patterns:
                        target_pairs = intra_doc_pairs
                    elif pattern_name in cross_year_patterns:
                        target_pairs = cross_year_pairs
                    elif pattern_name in cross_company_patterns:
                        target_pairs = cross_company_pairs
                    else:
                        # Unknown pattern - default to cross_company pairs
                        logger.warning(f"Unknown pattern {pattern_name}, defaulting to cross_company pairs")
                        target_pairs = cross_company_pairs

                    if not target_pairs:
                        logger.info(f"No pairs for pattern {pattern_name}, skipping")
                        continue

                    # Calculate questions for this pattern
                    remaining = questions_limit - total_generated if questions_limit else None

                    logger.info(f"\n{'='*60}")
                    logger.info(f"Generating {hop_count}-hop {pattern_name} questions...")
                    logger.info(f"{'='*60}")

                    # Generate with early stopping if limit set
                    if hop_count == 3 and config.three_hop_method == "causal":
                        if causal_3hop_done:
                            logger.info("Causal 3-hop already generated, skipping remaining 3-hop patterns")
                            continue
                        causal_names = [p for p in patterns if p in causal_patterns] if patterns else list(causal_patterns)
                        if not causal_names:
                            logger.info("No causal 3-hop patterns enabled; skipping")
                            continue
                        if pattern_name not in causal_patterns and patterns is not None:
                            logger.info(f"Skipping non-causal 3-hop pattern in causal mode: {pattern_name}")
                            continue
                        qa_items = self.qa_generator.generate_3hop_causal_batch(
                            pairs=target_pairs,
                            max_questions_per_pair=effective_per_pair,
                            max_total_questions=remaining,
                            pattern_names=causal_names,
                            max_workers=config.max_workers
                        )
                        result_key = "3hop_causal"
                        causal_3hop_done = True

                    # Type 1 patterns: Need pairs + connector chain
                    # CONNECTOR-FIRST APPROACH: Find chains first, then generate pairs
                    elif hop_count == 3 and pattern_name == 'cross_company_type1':
                        # Extract unique ORGs for filtering
                        unique_orgs = list(set(
                            org for pair in target_pairs
                            for org in [pair.org1, pair.org2]
                        ))

                        if len(unique_orgs) < 2:
                            logger.warning(f"cross_company_type1 requires at least 2 ORGs, got {len(unique_orgs)}: {unique_orgs}")
                            qa_items = []
                        else:
                            qa_items = []
                            # Get settings from target pairs
                            unique_years = list(set(pair.org1_year for pair in target_pairs)) if target_pairs else [2024]
                            unique_years.sort()
                            sector = getattr(target_pairs[0], 'sector', None) if target_pairs else None
                            sub_industry = getattr(target_pairs[0], 'sub_industry', None) if target_pairs else None
                            connector_types = getattr(target_pairs[0], 'connector_types', None) if target_pairs else None

                            logger.info(f"Using CONNECTOR-FIRST approach for cross_company_type1")
                            logger.info(f"  ORGs: {len(unique_orgs)}, Years: {unique_years}, Connector types: {connector_types}")

                            # STEP 1: Find connector chains with ORG connections (ONE query per year)
                            all_valid_pairs = []
                            for year in unique_years:
                                logger.info(f"Finding connector chains for year {year}...")
                                chains = self.discovery.find_connector_chains_with_orgs(
                                    year=year,
                                    connector1_types=connector_types,
                                    connector2_types=connector_types,
                                    org_filter=unique_orgs,
                                    limit=500  # Top 500 chains per year
                                )
                                logger.info(f"  Found {len(chains)} connector chains for {year}")

                                # STEP 2: Generate pairs from each chain's ORG lists
                                pairs_for_year = self.discovery.generate_type1_pairs_from_chains(
                                    chains=chains,
                                    org_filter=unique_orgs,
                                    max_pairs_per_chain=20,  # Limit per chain for diversity
                                    shuffle=True,
                                    seed=42
                                )

                                # Add year to each pair
                                for org1, org2, chain in pairs_for_year:
                                    all_valid_pairs.append((org1, org2, year, chain))

                            # Shuffle all pairs across years for diversity
                            random.seed(42)
                            random.shuffle(all_valid_pairs)

                            total_valid = len(all_valid_pairs)
                            logger.info(f"Total valid pairs (guaranteed to have connector chain): {total_valid}")
                            logger.info(f"  vs brute-force: C({len(unique_orgs)},2) × {len(unique_years)} = {len(list(itertools.combinations(unique_orgs, 2))) * len(unique_years)} combinations")

                            if total_valid == 0:
                                logger.warning("No valid pairs found with connector-first approach")
                            else:
                                # STEP 3: Process valid pairs (100% hit rate - no wasted queries!)
                                lock = threading.Lock()
                                stop_flag = threading.Event()
                                processed_count = [0]

                                def process_valid_pair(pair_with_index):
                                    """Process a pre-validated pair (guaranteed to have connector chain)."""
                                    idx, (org1, org2, year, chain) = pair_with_index
                                    if stop_flag.is_set():
                                        return []
                                    try:
                                        logger.info(f"Processing pair {idx+1}/{total_valid}: {org1}, {org2} ({year}) → {chain['c1_name']}→{chain['c2_name']}")

                                        # Build the Type1Result directly from pre-fetched chain data
                                        type1_result = self.discovery.build_type1_result_from_chain(
                                            chain=chain,
                                            org1=org1,
                                            org2=org2
                                        )

                                        if type1_result is None:
                                            logger.warning(f"Could not build result for {org1}-{org2}")
                                            return []

                                        # Generate QA using the pre-built result
                                        items = self.qa_generator.generate_3hop_cross_company_type1_from_result(
                                            type1_result=type1_result,
                                            year=year,
                                            sector=sector,
                                            sub_industry=sub_industry,
                                            max_questions=effective_per_pair
                                        )
                                        logger.info(f"Generated {len(items)} QA items for {org1}-{org2} ({year})")
                                        return items
                                    except Exception as e:
                                        logger.error(f"Failed to process pair {org1}-{org2} ({year}): {e}")
                                        return []

                                logger.info(f"Starting parallel processing with {config.max_workers} workers")

                                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                                    # Only submit what we need (not all pairs at once)
                                    batch_size = min(total_valid, (remaining or total_valid) * 3)  # 3x buffer for failures
                                    pairs_to_process = all_valid_pairs[:batch_size]

                                    futures = {
                                        executor.submit(process_valid_pair, (i, pair_data)): pair_data
                                        for i, pair_data in enumerate(pairs_to_process)
                                    }

                                    for future in as_completed(futures):
                                        pair_data = futures[future]
                                        try:
                                            items = future.result()
                                            with lock:
                                                qa_items.extend(items)
                                                processed_count[0] += 1
                                                if processed_count[0] % 10 == 0 or len(qa_items) >= (remaining or float('inf')):
                                                    logger.info(f"Progress: {processed_count[0]}/{len(pairs_to_process)} pairs, {len(qa_items)} QA items")

                                                if remaining and len(qa_items) >= remaining:
                                                    logger.info(f"Reached target ({remaining}), stopping...")
                                                    stop_flag.set()
                                        except Exception as e:
                                            logger.error(f"Pair future failed: {e}")

                                        if stop_flag.is_set():
                                            logger.info("Stop flag set, cancelling remaining futures...")
                                            for f in futures:
                                                f.cancel()
                                            break

                                logger.info(f"Connector-first approach complete: {len(qa_items)} QA items from {processed_count[0]} pairs")

                        result_key = f"{hop_count}hop_{pattern_name}"

                    # cross_year_type1: Same ORG, different years
                    # CONNECTOR-FIRST APPROACH
                    elif hop_count == 3 and pattern_name == 'cross_year_type1':
                        # Get unique ORGs and year combinations from cross_year pairs
                        unique_orgs = list(set(pair.org1 for pair in target_pairs))
                        year_combos = list(set((pair.org1_year, pair.org2_year) for pair in target_pairs))

                        if not year_combos:
                            logger.warning(f"No year combinations found for cross_year_type1")
                            qa_items = []
                        else:
                            qa_items = []
                            sector = getattr(target_pairs[0], 'sector', None) if target_pairs else None
                            sub_industry = getattr(target_pairs[0], 'sub_industry', None) if target_pairs else None
                            connector_types = getattr(target_pairs[0], 'connector_types', None) if target_pairs else None

                            logger.info(f"Using CONNECTOR-FIRST approach for cross_year_type1")
                            logger.info(f"  ORGs: {len(unique_orgs)}, Year combinations: {year_combos}")

                            # STEP 1: Find connector chains for each year combination
                            all_valid_orgs = []
                            for year1, year2 in year_combos:
                                logger.info(f"Finding connector chains for years {year1} vs {year2}...")
                                chains = self.discovery.find_connector_chains_cross_year(
                                    year1=year1,
                                    year2=year2,
                                    connector1_types=connector_types,
                                    connector2_types=connector_types,
                                    org_filter=unique_orgs,
                                    limit=500
                                )
                                logger.info(f"  Found {len(chains)} chains for {year1} vs {year2}")

                                # STEP 2: Generate valid ORGs from chains
                                orgs_for_combo = self.discovery.generate_type1_cross_year_from_chains(
                                    chains=chains,
                                    org_filter=unique_orgs,
                                    max_orgs_per_chain=20,
                                    shuffle=True,
                                    seed=42
                                )

                                for org, chain in orgs_for_combo:
                                    all_valid_orgs.append((org, year1, year2, chain))

                            # Shuffle all ORGs across year combos
                            random.seed(42)
                            random.shuffle(all_valid_orgs)

                            total_valid = len(all_valid_orgs)
                            logger.info(f"Total valid ORGs (guaranteed to have connector chain): {total_valid}")

                            if total_valid == 0:
                                logger.warning("No valid ORGs found with connector-first approach")
                            else:
                                lock = threading.Lock()
                                stop_flag = threading.Event()
                                processed_count = [0]

                                def process_valid_org_cross_year(org_with_index):
                                    idx, (org, year1, year2, chain) = org_with_index
                                    if stop_flag.is_set():
                                        return []
                                    try:
                                        logger.info(f"Processing {idx+1}/{total_valid}: {org} ({year1} vs {year2}) → {chain['c1_name']}→{chain['c2_name']}")

                                        type1_result = self.discovery.build_type1_cross_year_result(
                                            chain=chain,
                                            org=org,
                                            year1=year1,
                                            year2=year2
                                        )

                                        if type1_result is None:
                                            return []

                                        items = self.qa_generator.generate_3hop_cross_year_type1_from_result(
                                            type1_result=type1_result,
                                            org=org,
                                            year1=year1,
                                            year2=year2,
                                            sector=sector,
                                            sub_industry=sub_industry,
                                            max_questions=effective_per_pair
                                        )
                                        logger.info(f"Generated {len(items)} QA items for {org} ({year1} vs {year2})")
                                        return items
                                    except Exception as e:
                                        logger.error(f"Failed to process {org} ({year1} vs {year2}): {e}")
                                        return []

                                logger.info(f"Starting parallel processing with {config.max_workers} workers")

                                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                                    batch_size = min(total_valid, (remaining or total_valid) * 3)
                                    orgs_to_process = all_valid_orgs[:batch_size]

                                    futures = {
                                        executor.submit(process_valid_org_cross_year, (i, org_data)): org_data
                                        for i, org_data in enumerate(orgs_to_process)
                                    }

                                    for future in as_completed(futures):
                                        try:
                                            items = future.result()
                                            with lock:
                                                qa_items.extend(items)
                                                processed_count[0] += 1
                                                if processed_count[0] % 10 == 0 or len(qa_items) >= (remaining or float('inf')):
                                                    logger.info(f"Progress: {processed_count[0]}/{len(orgs_to_process)}, {len(qa_items)} QA items")
                                                if remaining and len(qa_items) >= remaining:
                                                    stop_flag.set()
                                        except Exception as e:
                                            logger.error(f"Future failed: {e}")

                                        if stop_flag.is_set():
                                            for f in futures:
                                                f.cancel()
                                            break

                                logger.info(f"Connector-first complete: {len(qa_items)} QA items from {processed_count[0]} ORGs")

                        result_key = f"{hop_count}hop_{pattern_name}"

                    # intra_doc_type1: Same ORG, same year, same document, different pages
                    # CONNECTOR-FIRST APPROACH
                    elif hop_count == 3 and pattern_name == 'intra_doc_type1':
                        unique_orgs = list(set(pair.org1 for pair in target_pairs))
                        unique_years = list(set(pair.org1_year for pair in target_pairs))

                        if not unique_orgs or not unique_years:
                            logger.warning(f"No ORGs or years found for intra_doc_type1")
                            qa_items = []
                        else:
                            qa_items = []
                            sector = getattr(target_pairs[0], 'sector', None) if target_pairs else None
                            sub_industry = getattr(target_pairs[0], 'sub_industry', None) if target_pairs else None
                            connector_types = getattr(target_pairs[0], 'connector_types', None) if target_pairs else None

                            logger.info(f"Using CONNECTOR-FIRST approach for intra_doc_type1")
                            logger.info(f"  ORGs: {len(unique_orgs)}, Years: {unique_years}")

                            # STEP 1: Find connector chains for each year
                            all_valid_orgs = []
                            for year in unique_years:
                                logger.info(f"Finding intra_doc connector chains for year {year}...")
                                chains = self.discovery.find_connector_chains_intra_doc(
                                    year=year,
                                    connector1_types=connector_types,
                                    connector2_types=connector_types,
                                    org_filter=unique_orgs,
                                    limit=500
                                )
                                logger.info(f"  Found {len(chains)} chains for {year}")

                                # STEP 2: Generate valid ORGs from chains
                                orgs_for_year = self.discovery.generate_type1_intra_doc_from_chains(
                                    chains=chains,
                                    org_filter=unique_orgs,
                                    max_orgs_per_chain=20,
                                    shuffle=True,
                                    seed=42
                                )

                                for org, chain in orgs_for_year:
                                    all_valid_orgs.append((org, year, chain))

                            # Shuffle all ORGs across years
                            random.seed(42)
                            random.shuffle(all_valid_orgs)

                            total_valid = len(all_valid_orgs)
                            logger.info(f"Total valid ORGs (guaranteed to have intra-doc chain): {total_valid}")

                            if total_valid == 0:
                                logger.warning("No valid ORGs found with connector-first approach")
                            else:
                                lock = threading.Lock()
                                stop_flag = threading.Event()
                                processed_count = [0]

                                def process_valid_org_intra_doc(org_with_index):
                                    idx, (org, year, chain) = org_with_index
                                    if stop_flag.is_set():
                                        return []
                                    try:
                                        logger.info(f"Processing {idx+1}/{total_valid}: {org} ({year}) → {chain['c1_name']}→{chain['c2_name']}")

                                        type1_result = self.discovery.build_type1_intra_doc_result(
                                            chain=chain,
                                            org=org
                                        )

                                        if type1_result is None:
                                            return []

                                        items = self.qa_generator.generate_3hop_intra_doc_type1_from_result(
                                            type1_result=type1_result,
                                            org=org,
                                            year=year,
                                            sector=sector,
                                            sub_industry=sub_industry,
                                            max_questions=effective_per_pair
                                        )
                                        logger.info(f"Generated {len(items)} QA items for {org} ({year})")
                                        return items
                                    except Exception as e:
                                        logger.error(f"Failed to process {org} ({year}): {e}")
                                        return []

                                logger.info(f"Starting parallel processing with {config.max_workers} workers")

                                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                                    batch_size = min(total_valid, (remaining or total_valid) * 3)
                                    orgs_to_process = all_valid_orgs[:batch_size]

                                    futures = {
                                        executor.submit(process_valid_org_intra_doc, (i, org_data)): org_data
                                        for i, org_data in enumerate(orgs_to_process)
                                    }

                                    for future in as_completed(futures):
                                        try:
                                            items = future.result()
                                            with lock:
                                                qa_items.extend(items)
                                                processed_count[0] += 1
                                                if processed_count[0] % 10 == 0 or len(qa_items) >= (remaining or float('inf')):
                                                    logger.info(f"Progress: {processed_count[0]}/{len(orgs_to_process)}, {len(qa_items)} QA items")
                                                if remaining and len(qa_items) >= remaining:
                                                    stop_flag.set()
                                        except Exception as e:
                                            logger.error(f"Future failed: {e}")

                                        if stop_flag.is_set():
                                            for f in futures:
                                                f.cancel()
                                            break

                                logger.info(f"Connector-first complete: {len(qa_items)} QA items from {processed_count[0]} ORGs")

                        result_key = f"{hop_count}hop_{pattern_name}"

                    # Type 2 patterns: Need 3 ORGs (triplets) instead of pairs
                    # CONNECTOR-FIRST APPROACH: Much more efficient than brute-force
                    elif hop_count == 3 and pattern_name == 'cross_company_type2':
                        # Extract unique ORGs for filtering
                        unique_orgs = list(set(
                            org for pair in target_pairs
                            for org in [pair.org1, pair.org2]
                        ))

                        if len(unique_orgs) < 3:
                            logger.warning(f"cross_company_type2 requires at least 3 ORGs, got {len(unique_orgs)}: {unique_orgs}")
                            qa_items = []
                        else:
                            qa_items = []
                            # Get settings from target pairs
                            unique_years = list(set(pair.org1_year for pair in target_pairs)) if target_pairs else [2024]
                            unique_years.sort()
                            sector = getattr(target_pairs[0], 'sector', None) if target_pairs else None
                            connector_types = getattr(target_pairs[0], 'connector_types', None) if target_pairs else None

                            logger.info(f"Using CONNECTOR-FIRST approach for cross_company_type2")
                            logger.info(f"  ORGs: {len(unique_orgs)}, Years: {unique_years}, Connector types: {connector_types}")

                            # STEP 1: Find connectors with 3+ ORG connections (ONE query per year)
                            all_valid_triplets = []
                            for year in unique_years:
                                logger.info(f"Finding connectors with 3+ ORG connections for year {year}...")
                                connectors = self.discovery.find_connectors_with_multiple_orgs(
                                    year=year,
                                    connector_types=connector_types,
                                    min_org_count=3,
                                    org_filter=unique_orgs,
                                    limit=500  # Top 500 connectors per year
                                )
                                logger.info(f"  Found {len(connectors)} connectors with 3+ ORGs for {year}")

                                # STEP 2: Generate triplets from each connector's ORG list
                                triplets_for_year = self.discovery.generate_type2_triplets_from_connectors(
                                    connectors=connectors,
                                    org_filter=unique_orgs,
                                    max_triplets_per_connector=20,  # Limit per connector for diversity
                                    shuffle=True,
                                    seed=42
                                )

                                # Add year to each triplet
                                for org1, org2, org3, conn in triplets_for_year:
                                    all_valid_triplets.append((org1, org2, org3, year, conn))

                            # Shuffle all triplets across years for diversity
                            random.seed(42)
                            random.shuffle(all_valid_triplets)

                            total_valid = len(all_valid_triplets)
                            logger.info(f"Total valid triplets (guaranteed to share connector): {total_valid}")
                            logger.info(f"  vs brute-force: C({len(unique_orgs)},3) × {len(unique_years)} = {len(list(itertools.combinations(unique_orgs, 3))) * len(unique_years)} combinations")

                            if total_valid == 0:
                                logger.warning("No valid triplets found with connector-first approach")
                            else:
                                # STEP 3: Process valid triplets (100% hit rate - no wasted queries!)
                                lock = threading.Lock()
                                stop_flag = threading.Event()
                                processed_count = [0]

                                def process_valid_triplet(triplet_with_index):
                                    """Process a pre-validated triplet (guaranteed to have shared connector)."""
                                    idx, (org1, org2, org3, year, conn) = triplet_with_index
                                    if stop_flag.is_set():
                                        return []
                                    try:
                                        logger.info(f"Processing triplet {idx+1}/{total_valid}: {org1}, {org2}, {org3} ({year}) → {conn['connector_name']}")

                                        # Build the Type2Result directly from pre-fetched connector data
                                        type2_result = self.discovery.build_type2_result_from_connector(
                                            connector=conn,
                                            org1=org1,
                                            org2=org2,
                                            org3=org3
                                        )

                                        if type2_result is None:
                                            logger.warning(f"Could not build result for {org1}-{org2}-{org3}")
                                            return []

                                        # Generate QA using the pre-built result
                                        items = self.qa_generator.generate_3hop_cross_company_type2_from_result(
                                            type2_result=type2_result,
                                            year=year,
                                            sector=sector,
                                            max_questions=effective_per_pair
                                        )
                                        logger.info(f"Generated {len(items)} QA items for {org1}-{org2}-{org3} ({year})")
                                        return items
                                    except Exception as e:
                                        logger.error(f"Failed to process triplet {org1}-{org2}-{org3} ({year}): {e}")
                                        return []

                                logger.info(f"Starting parallel processing with {config.max_workers} workers")

                                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                                    # Only submit what we need (not all triplets at once)
                                    batch_size = min(total_valid, (remaining or total_valid) * 3)  # 3x buffer for failures
                                    triplets_to_process = all_valid_triplets[:batch_size]

                                    futures = {
                                        executor.submit(process_valid_triplet, (i, triplet)): triplet
                                        for i, triplet in enumerate(triplets_to_process)
                                    }

                                    for future in as_completed(futures):
                                        triplet = futures[future]
                                        try:
                                            items = future.result()
                                            with lock:
                                                qa_items.extend(items)
                                                processed_count[0] += 1
                                                if processed_count[0] % 10 == 0 or len(qa_items) >= (remaining or float('inf')):
                                                    logger.info(f"Progress: {processed_count[0]}/{len(triplets_to_process)} triplets, {len(qa_items)} QA items")

                                                if remaining and len(qa_items) >= remaining:
                                                    logger.info(f"Reached target ({remaining}), stopping...")
                                                    stop_flag.set()
                                        except Exception as e:
                                            logger.error(f"Triplet future failed: {e}")

                                        if stop_flag.is_set():
                                            logger.info("Stop flag set, cancelling remaining futures...")
                                            for f in futures:
                                                f.cancel()
                                            break

                                logger.info(f"Connector-first approach complete: {len(qa_items)} QA items from {processed_count[0]} triplets")

                        result_key = f"{hop_count}hop_{pattern_name}"

                    else:
                        qa_items = self.qa_generator.generate_qa_batch(
                            pairs=target_pairs,
                            max_questions_per_pair=effective_per_pair,
                            hop_count=hop_count,
                            pattern=pattern_name if hop_count == 3 else None,
                            max_total_questions=remaining,  # Stop early when limit reached
                            max_workers=config.max_workers
                        )
                        result_key = f"{hop_count}hop_{pattern_name}"

                    results[result_key] = qa_items
                    total_generated += len(qa_items)

                    logger.info(f"Generated {len(qa_items)} questions for {result_key}")

            # Compile metadata
            metadata = {
                'total_questions': total_generated,
                'by_pattern': {k: len(v) for k, v in results.items()},
                'pairs_processed': len(pairs),
                'cross_company_pairs': len(cross_company_pairs),
                'cross_year_pairs': len(cross_year_pairs),
                'config_overrides': config.to_dict(),
                'timestamp': datetime.now().isoformat()
            }

            # Save results
            if save_results:
                self._save_results(results, metadata, config)

            logger.info("\n" + "=" * 80)
            logger.info("Pipeline Complete!")
            logger.info(f"Total questions generated: {total_generated}")
            for key, items in results.items():
                logger.info(f"  {key}: {len(items)}")
            logger.info("=" * 80)

            return {
                'results': results,
                'metadata': metadata
            }

        finally:
            self._cleanup()

    def _save_results(
        self,
        results: Dict[str, List[QAItem]],
        metadata: Dict[str, Any],
        config: PipelineConfig
    ):
        """Save results to files organized by connector mode, hop count, and category.

        Folder structure:
            outputs/
            ├── quantitative/
            │   ├── 2hop/
            │   │   ├── cross-company/
            │   │   ├── cross-year/
            │   │   └── intra-doc/
            │   └── 3hop/
            │       ├── cross-company/
            │       └── cross-year/
            └── qualitative/
                ├── 2hop/
                │   ├── cross-company/
                │   ├── cross-year/
                │   └── intra-doc/
                └── 3hop/
                    ├── cross-company/
                    └── cross-year/
        """
        base_output_dir = Path(config.output_dir) if config.output_dir else Path(__file__).parent / "outputs"

        # Add connector mode subfolder if using quantitative or qualitative mode
        if config.connector_mode in ('quantitative', 'qualitative'):
            mode_dir = base_output_dir / config.connector_mode
        else:
            mode_dir = base_output_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = config.output_prefix

        # Pattern mapping: pattern -> (hop_count, category)
        PATTERN_MAP = {
            # 2-hop patterns
            '2hop_cross_company': ('2hop', 'cross-company'),
            '2hop_cross_year': ('2hop', 'cross-year'),
            '2hop_intra_doc': ('2hop', 'intra-doc'),

            # 3-hop Type 1 patterns (Connector Chain: O─C1─C2─O)
            '3hop_cross_company_type1': ('3hop', 'cross-company'),
            '3hop_cross_year_type1': ('3hop', 'cross-year'),
            '3hop_intra_doc_type1': ('3hop', 'intra-doc'),

            # 3-hop Type 2 patterns (Multi-Branch: 3 anchors → 1 connector)
            '3hop_cross_company_type2': ('3hop', 'cross-company'),
            '3hop_cross_year_type2': ('3hop', 'cross-year'),
            '3hop_intra_doc_type2': ('3hop', 'intra-doc'),

            # 3-hop causal patterns
            '3hop_causal': ('3hop', 'cross-company'),
        }

        # Save each pattern separately to categorized folders
        for key, items in results.items():
            if items:
                # Determine hop and category folder
                hop_count, category = PATTERN_MAP.get(key, ('other', 'other'))
                category_dir = mode_dir / hop_count / category
                category_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{prefix}_{key}_{timestamp}.json"
                filepath = category_dir / filename

                data = {
                    'metadata': {
                        'pattern': key,
                        'hop_count': hop_count,
                        'category': category,
                        'connector_mode': config.connector_mode,
                        'count': len(items),
                        'timestamp': timestamp
                    },
                    'questions': [item.to_dict(question_id=i+1) for i, item in enumerate(items)]
                }

                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Saved {len(items)} items to {filepath}")

    def run_from_config(self, save_results: bool = True) -> Dict[str, Any]:
        """Run pipeline using config.yaml settings (no overrides)."""
        return self._run_with_config(PipelineConfig(), save_results)

    def test(
        self,
        orgs: List[str] = ['NVDA', 'AMD'],
        year: int = 2024,
        hops: List[int] = [2],
        num_questions: int = 5
    ) -> Dict[str, Any]:
        """
        Quick test mode with sensible defaults.

        Example:
            p = Pipeline()
            results = p.test()  # 5 questions, NVDA-AMD, 2024, 2-hop
            results = p.test(hops=[2, 3], num_questions=10)
        """
        logger.info("Running in TEST mode")
        return self.run(
            orgs=orgs,
            year=year,
            hops=hops,
            num_questions=num_questions,
            skip_temporal=True,  # Faster testing
            output_prefix="test"
        )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-hop QA Generation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test: 10 questions, NVDA-AMD, 2024, 2-hop
  python pipeline.py --test

  # Custom test
  python pipeline.py --orgs NVDA,AMD,INTC --year 2024 --hops 2 --num 20

  # Quantitative QA (FIN_METRIC, SEGMENT, ECON_IND) - saves to outputs/quantitative/
  python pipeline.py --quantitative --orgs NVDA,AMD --year 2024 --num 10

  # Qualitative QA (PRODUCT, FIN_INST, RISK_FACTOR, etc.) - saves to outputs/qualitative/
  python pipeline.py --qualitative --orgs NVDA,AMD --year 2024 --num 10

  # 3-hop only
  python pipeline.py --hops 3 --pattern cross_company_type1 --num 10

  # Full production run
  python pipeline.py --full

  # Specific sector with qualitative connectors
  python pipeline.py --sector "Semiconductors" --qualitative --hops 2,3 --num 100

  # Manual connector type override (takes precedence over --quantitative/--qualitative)
  python pipeline.py --connector-types COMP,RISK_FACTOR --orgs NVDA,AMD --num 10
        """
    )

    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--test', action='store_true', help='Quick test mode (NVDA-AMD, 2024, 2-hop)')
    mode.add_argument('--full', action='store_true', help='Full production run from config')

    # Filters
    parser.add_argument('--orgs', type=str, help='Companies to include (comma-separated), e.g., NVDA,AMD')
    parser.add_argument('--sector', type=str, help='Filter by sector, e.g., "Semiconductors"')
    parser.add_argument('--year', type=int, help='Single year to filter')
    parser.add_argument('--years', type=str, help='Years to include (comma-separated), e.g., 2022,2024')

    # Hop configuration
    parser.add_argument('--hops', type=str, default='2', help='Hop counts (comma-separated), e.g., 2,3')
    parser.add_argument('--pattern', type=str, help='Single pattern to run')
    parser.add_argument('--skip-temporal', action='store_true', help='Skip cross_year (temporal) pairs')
    parser.add_argument('--skip-cross-company', action='store_true', help='Skip cross_company pairs')

    # Limits
    parser.add_argument('--num', type=int, help='Total questions to generate')
    parser.add_argument('--per-pair', type=int, help='Max questions per pair (optional limit)')
    parser.add_argument('--max-pairs', type=int, help='Max pairs to process')

    # Connector configuration
    parser.add_argument('--min-idf', type=float, help='Minimum IDF threshold')
    parser.add_argument('--connector-types', type=str,
                        help='Override connector types (comma-separated), e.g., COMP,RISK_FACTOR')
    connector_mode = parser.add_mutually_exclusive_group()
    connector_mode.add_argument('--quantitative', action='store_true',
                                help='Use quantitative connector types (FIN_METRIC, SEGMENT, ECON_IND)')
    connector_mode.add_argument('--qualitative', action='store_true',
                                help='Use qualitative connector types (PRODUCT, FIN_INST, RISK_FACTOR, etc.)')

    # Other
    parser.add_argument('--llm', type=str, help='LLM model name')
    parser.add_argument('--3-hop-method', dest='three_hop_method', choices=['default', 'causal'],
                        default='default', help='3-hop method: default or causal')
    parser.add_argument('--prompt-variant', type=str, choices=['default', 'v3_binary'],
                        help='QA validation prompt variant: "default" (scoring 0-50) or "v3_binary" (pass/fail). Overrides config.yaml')
    parser.add_argument('--workers', type=int, default=20,
                        help='Max parallel workers for LLM requests (default: 20)')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to files')

    # Cache options
    parser.add_argument('--use-cache', action='store_true', help='Load pairs from cache (skip Neo4j validation)')
    parser.add_argument('--build-cache', action='store_true', help='Build pairs cache for future runs')

    args = parser.parse_args()

    pipeline = Pipeline()
    
    # Override prompt_variant in config if provided via CLI
    if args.prompt_variant:
        if 'qa_validation' not in pipeline.base_config:
            pipeline.base_config['qa_validation'] = {}
        pipeline.base_config['qa_validation']['prompt_variant'] = args.prompt_variant
        logger.info(f"Overriding prompt_variant to: {args.prompt_variant}")

    try:
        if args.test:
            results = pipeline.test()
        elif args.full:
            results = pipeline.run_from_config(save_results=not args.no_save)
        else:
            # Determine connector mode
            if args.quantitative:
                conn_mode = 'quantitative'
            elif args.qualitative:
                conn_mode = 'qualitative'
            else:
                conn_mode = 'all'

            # Custom run with CLI args
            results = pipeline.run(
                orgs=args.orgs.split(',') if args.orgs else None,
                sector=args.sector,
                year=args.year,
                years=[int(y) for y in args.years.split(',')] if args.years else None,
                hops=[int(h) for h in args.hops.split(',')],
                pattern=args.pattern,
                skip_temporal=args.skip_temporal,
                skip_cross_company=args.skip_cross_company,
                num_questions=args.num,
                questions_per_pair=args.per_pair,
                max_pairs=args.max_pairs,
                min_idf=args.min_idf,
                connector_types=[t.strip() for t in args.connector_types.split(',')] if args.connector_types else None,
                connector_mode=conn_mode,
                llm_model=args.llm,
                three_hop_method=args.three_hop_method,
                max_workers=args.workers,
                output_dir=args.output_dir,
                save_results=not args.no_save,
                use_cache=args.use_cache,
                build_cache=args.build_cache
            )

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total questions: {results['metadata']['total_questions']}")
        for pattern, count in results['metadata']['by_pattern'].items():
            print(f"  {pattern}: {count}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
