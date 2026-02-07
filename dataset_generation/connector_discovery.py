"""
Connector Discovery Module

This module finds high-IDF connectors between company pairs in the Neo4j graph.
Connectors are entities (COMP, RISK_FACTOR, etc.) that link two companies,
enabling multi-hop QA generation.

Key concepts:
- IDF (Inverse Document Frequency): log(N / org_degree) where N=total ORGs
- Higher IDF = more specific/informative connector
- See docs/logic.md for full rationale

Usage:
    from connector_discovery import ConnectorDiscovery

    discovery = ConnectorDiscovery()
    connectors = discovery.find_connectors('NVDA', 'AMD', min_idf=4.0)
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

import yaml
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Causal 2-hop motifs (x1)-[p2]->(x2) used for 3-hop causal patterns
CAUSAL_2HOP_MOTIFS = [
    ("RISK_FACTOR", "Negatively_Impacts", "FIN_METRIC"),
    ("ESG_TOPIC", "Positively_Impacts", "FIN_METRIC"),
    ("ESG_TOPIC", "Negatively_Impacts", "FIN_METRIC"),
    ("MACRO_CONDITION", "Impacts", "FIN_METRIC"),
    ("MACRO_CONDITION", "Negatively_Impacts", "FIN_METRIC"),
    ("MACRO_CONDITION", "Impacts", "RISK_FACTOR"),
    ("MACRO_CONDITION", "Negatively_Impacts", "RISK_FACTOR"),
    ("EVENT", "Impacted_By", "MACRO_CONDITION"),
    ("EVENT", "Causes_Shortage_Of", "RAW_MATERIAL"),
    ("RAW_MATERIAL", "Impacts", "FIN_METRIC"),
    ("LOGISTICS", "Impacts", "RISK_FACTOR"),
    ("LOGISTICS", "Impacts", "FIN_METRIC"),
    ("REGULATORY_REQUIREMENT", "Impacts", "FIN_METRIC"),
    ("ACCOUNTING_POLICY", "Impacts", "FIN_METRIC"),
    ("LITIGATION", "Negatively_Impacts", "FIN_METRIC"),
    ("CONCEPT", "Impacts", "FIN_METRIC"),
    ("PRODUCT", "Impacts", "FIN_METRIC"),
]


@dataclass
class ConnectorResult:
    """Result from 2-hop connector discovery."""
    # Required fields (no defaults) - must come first
    connector_name: str
    connector_type: str
    org_degree: int
    idf_score: float

    # Company 1 (org1) evidence - Hop 1
    org1_relationship: str
    org1_source_file: str
    org1_page_id: str
    org1_chunk_id: Optional[str]

    # Company 2 (org2) evidence - Hop 2
    org2_relationship: str
    org2_source_file: str
    org2_page_id: str
    org2_chunk_id: Optional[str]

    # Fields with defaults - must come after required fields
    hop_count: int = 2
    org1_context: Optional[str] = None
    org2_context: Optional[str] = None

    @property
    def has_asymmetric_relationships(self) -> bool:
        """Check if the two companies have different relationship types."""
        return self.org1_relationship != self.org2_relationship

    @property
    def org1_year(self) -> Optional[int]:
        """Extract year from org1 source file."""
        return self._extract_year(self.org1_source_file)

    @property
    def org2_year(self) -> Optional[int]:
        """Extract year from org2 source file."""
        return self._extract_year(self.org2_source_file)

    @staticmethod
    def _extract_year(source_file: str) -> Optional[int]:
        """Extract year from source file name like 'NVDA_10k_2024.pdf'."""
        if not source_file:
            return None
        for year in [2024, 2023, 2022]:
            if str(year) in source_file:
                return year
        return None


@dataclass
class ThreeHopResult:
    """
    Result from 3-hop connector discovery.

    Patterns supported:
    - cross_company_type1: ORG1 → Connector1 → Connector2 ← ORG2
    - entity_extension: ORG1 → Connector ← ORG2 → Entity
    - temporal_extension: ORG(year1) → Connector ← ORG(year2) → Entity
    """
    # Required fields (no defaults) - must come first
    pattern_type: str  # 'cross_company_type1', 'entity_extension', 'temporal_extension'

    # Connector 1 (bridge entity)
    connector1_name: str
    connector1_type: str
    connector1_idf: float
    connector1_org_degree: int

    # Hop 1: ORG1 → Connector1
    hop1_org: str
    hop1_relationship: str
    hop1_source_file: str
    hop1_page_id: str
    hop1_chunk_id: Optional[str]

    # Hop 2: Connector1 → Connector2 OR Connector1 ← ORG2
    hop2_entity_name: str
    hop2_entity_type: str
    hop2_relationship: str
    hop2_source_file: str
    hop2_page_id: str
    hop2_chunk_id: Optional[str]

    # Hop 3: Connector2 ← ORG2 OR ORG2 → Entity
    hop3_entity_name: str  # For entity_extension, this is the extended entity
    hop3_entity_type: str
    hop3_org: str  # ORG2
    hop3_relationship: str
    hop3_source_file: str
    hop3_page_id: str
    hop3_chunk_id: Optional[str]

    # Fields with defaults - must come after required fields
    hop_count: int = 3
    combined_idf_score: float = 0.0
    hop3_entity_idf: Optional[float] = None

    @property
    def hop1_year(self) -> Optional[int]:
        return self._extract_year(self.hop1_source_file)

    @property
    def hop2_year(self) -> Optional[int]:
        return self._extract_year(self.hop2_source_file)

    @property
    def hop3_year(self) -> Optional[int]:
        return self._extract_year(self.hop3_source_file)

    @staticmethod
    def _extract_year(source_file: str) -> Optional[int]:
        if not source_file:
            return None
        for year in [2024, 2023, 2022]:
            if str(year) in source_file:
                return year
        return None


@dataclass
class ThreeHopType2Result:
    """
    Result from 3-hop Type 2 (Multi-Branch) connector discovery.

    Pattern: 3 anchors → 1 connector
             ORG1
              │
            [r1]
              │
      ORG2 ─[r2]─ Connector ─[r3]─ ORG3

    Patterns supported:
    - cross_company_type2: 3 different ORGs, same year
    - cross_year_type2: Same ORG, 3 different years
    - intra_doc_type2: Same ORG, same year, 3 different pages
    """
    # Required fields (no defaults) - must come first
    pattern_type: str  # 'cross_company_type2', 'cross_year_type2', 'intra_doc_type2'

    # Single connector (bridge entity)
    connector_name: str
    connector_type: str
    connector_idf: float
    connector_org_degree: int

    # Hop 1: ORG1 → Connector
    hop1_org: str
    hop1_relationship: str
    hop1_source_file: str
    hop1_page_id: str
    hop1_chunk_id: Optional[str]

    # Hop 2: ORG2 → Connector
    hop2_org: str
    hop2_relationship: str
    hop2_source_file: str
    hop2_page_id: str
    hop2_chunk_id: Optional[str]

    # Hop 3: ORG3 → Connector
    hop3_org: str
    hop3_relationship: str
    hop3_source_file: str
    hop3_page_id: str
    hop3_chunk_id: Optional[str]

    # Optional context fields
    hop_count: int = 3
    hop1_context: Optional[str] = None
    hop2_context: Optional[str] = None
    hop3_context: Optional[str] = None

    @property
    def hop1_year(self) -> Optional[int]:
        return self._extract_year(self.hop1_source_file)

    @property
    def hop2_year(self) -> Optional[int]:
        return self._extract_year(self.hop2_source_file)

    @property
    def hop3_year(self) -> Optional[int]:
        return self._extract_year(self.hop3_source_file)

    @staticmethod
    def _extract_year(source_file: str) -> Optional[int]:
        if not source_file:
            return None
        for year in [2024, 2023, 2022]:
            if str(year) in source_file:
                return year
        return None


@dataclass
class ConnectorDiscoveryConfig:
    """Configuration for connector discovery."""
    # Neo4j connection
    neo4j_uri: str = "<YOUR_NEO4J_URI>"
    neo4j_username: str = "<YOUR_NEO4J_USERNAME>"
    neo4j_password: str = "<YOUR_NEO4J_PASSWORD>"
    neo4j_database: str = "neo4j"

    # IDF parameters
    total_orgs: int = 1432  # Total ORGs in graph (for IDF calculation)
    default_min_idf: float = 4.0  # Default IDF threshold

    # Difficulty tiers
    hard_idf_threshold: float = 5.0
    medium_idf_threshold: float = 4.0
    easy_idf_threshold: float = 3.5

    def __post_init__(self):
        """Load from config.yaml and environment variables if available."""
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            neo_cfg = config.get("neo4j", {})
            self.neo4j_uri = neo_cfg.get("uri", self.neo4j_uri)
            self.neo4j_username = neo_cfg.get("username", self.neo4j_username)
            self.neo4j_password = neo_cfg.get("password", self.neo4j_password)
            self.neo4j_database = neo_cfg.get("database", self.neo4j_database)
            self.total_orgs = config.get("total_orgs", self.total_orgs)

        neo4j_auth = os.getenv("NEO4J_AUTH", "")
        if neo4j_auth:
            parts = neo4j_auth.split("/")
            if len(parts) == 2:
                self.neo4j_username = parts[0]
                self.neo4j_password = parts[1]

        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_username = os.getenv("NEO4J_USER", self.neo4j_username)
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        self.neo4j_database = os.getenv("NEO4J_DATABASE", self.neo4j_database)


class ConnectorDiscovery:
    """
    Discovers high-IDF connectors between company pairs in Neo4j.

    Example:
        discovery = ConnectorDiscovery()
        connectors = discovery.find_connectors('NVDA', 'AMD', min_idf=4.0)
        for c in connectors:
            print(f"{c.connector_name} (IDF: {c.idf_score:.2f})")
    """

    def __init__(self, config: Optional[ConnectorDiscoveryConfig] = None):
        self.config = config or ConnectorDiscoveryConfig()
        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j."""
        logger.info(f"Connecting to Neo4j at {self.config.neo4j_uri}...")
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_username, self.config.neo4j_password),
            max_connection_pool_size=100,  # Handle parallel workers
            connection_acquisition_timeout=60,  # Wait up to 60s for connection from pool
            connection_timeout=30,  # Connection establishment timeout
        )
        # Test connection
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run("RETURN 1 as test")
            result.single()
        logger.info("Connected to Neo4j successfully")

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def get_total_orgs(self) -> int:
        """Get the total number of ORG nodes (for IDF calculation)."""
        query = """
        MATCH (o:ORG)
        WHERE o.name =~ '[A-Z]{1,5}'
        RETURN count(o) as total
        """
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query)
            record = result.single()
            return record["total"] if record else self.config.total_orgs

    def find_connectors(
        self,
        org1: str,
        org2: str,
        connector_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ConnectorResult]:
        """
        Find connectors between two companies.

        Args:
            org1: First company ticker (e.g., 'NVDA')
            org2: Second company ticker (e.g., 'AMD')
            connector_types: List of connector types to consider
                            (e.g., ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION'])
            min_idf: Minimum IDF threshold (default: 4.0)
            limit: Maximum number of connectors to return

        Returns:
            List of ConnectorResult objects sorted by IDF score descending
        """
        if connector_types is None:
            connector_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION',
                            'RAW_MATERIAL', 'FIN_INST', 'REGULATORY_REQUIREMENT', 'ORG_REG']

        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs

        query = """
        MATCH (o1:ORG {name: $org1})-[r1]-(x)-[r2]-(o2:ORG {name: $org2})
        WHERE any(label IN labels(x) WHERE label IN $valid_types)

        WITH x,
             type(r1) as rel1, type(r2) as rel2,
             r1.source_file as src1, r1.page_id as page1, r1.chunk_id as chunk1,
             r2.source_file as src2, r2.page_id as page2, r2.chunk_id as chunk2,
             r1.context as ctx1, r2.context as ctx2

        MATCH (x)--(o:ORG)
        WITH x, rel1, rel2, src1, page1, chunk1, src2, page2, chunk2, ctx1, ctx2,
             count(DISTINCT o) as org_degree

        WITH x, rel1, rel2, src1, page1, chunk1, src2, page2, chunk2, ctx1, ctx2,
             org_degree, log(toFloat($total_orgs) / org_degree) as idf_score
        WHERE idf_score > $min_idf

        RETURN
            x.name as connector_name,
            [l IN labels(x) WHERE l <> 'Entity'][0] as connector_type,
            org_degree,
            idf_score,
            rel1, src1, page1, chunk1, ctx1,
            rel2, src2, page2, chunk2, ctx2
        ORDER BY idf_score DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org1=org1,
                org2=org2,
                valid_types=connector_types,
                total_orgs=total_orgs,
                min_idf=min_idf,
                limit=limit
            )

            # Track seen connectors + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_connectors = set()

            for record in records:
                # Deduplicate by (connector, page1, page2)
                key = (record["connector_name"], record["page1"], record["page2"])
                if key in seen_connectors:
                    continue
                seen_connectors.add(key)

                result = ConnectorResult(
                    connector_name=record["connector_name"],
                    connector_type=record["connector_type"],
                    org_degree=record["org_degree"],
                    idf_score=record["idf_score"],
                    org1_relationship=record["rel1"],
                    org1_source_file=record["src1"],
                    org1_page_id=record["page1"],
                    org1_chunk_id=record["chunk1"],
                    org2_relationship=record["rel2"],
                    org2_source_file=record["src2"],
                    org2_page_id=record["page2"],
                    org2_chunk_id=record["chunk2"],
                    org1_context=record["ctx1"],
                    org2_context=record["ctx2"]
                )
                results.append(result)

        logger.info(f"Found {len(results)} unique connectors between {org1} and {org2} with IDF > {min_idf}")
        return results

    def find_connectors_by_difficulty(
        self,
        org1: str,
        org2: str,
        difficulty: str,
        connector_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[ConnectorResult]:
        """
        Find connectors filtered by difficulty tier.

        Args:
            org1: First company ticker
            org2: Second company ticker
            difficulty: 'hard', 'medium', or 'easy'
            connector_types: List of connector types to consider
            limit: Maximum number of connectors to return

        Returns:
            List of ConnectorResult objects
        """
        thresholds = {
            'hard': self.config.hard_idf_threshold,
            'medium': self.config.medium_idf_threshold,
            'easy': self.config.easy_idf_threshold
        }

        if difficulty not in thresholds:
            raise ValueError(f"Invalid difficulty: {difficulty}. Use 'hard', 'medium', or 'easy'")

        min_idf = thresholds[difficulty]
        return self.find_connectors(org1, org2, connector_types, min_idf, limit)

    def get_connector_stats(self, org1: str, org2: str) -> Dict[str, Any]:
        """
        Get statistics about connectors between two companies.

        Returns:
            Dictionary with connector counts at different IDF thresholds
        """
        query = """
        MATCH (o1:ORG {name: $org1})-[r1]-(x)-[r2]-(o2:ORG {name: $org2})
        WHERE x:COMP OR x:RISK_FACTOR OR x:MACRO_CONDITION OR x:RAW_MATERIAL
           OR x:FIN_INST OR x:REGULATORY_REQUIREMENT OR x:ORG_REG

        WITH DISTINCT x
        MATCH (x)--(o:ORG)
        WITH x, count(DISTINCT o) as org_degree

        WITH x, org_degree, log(toFloat($total_orgs) / org_degree) as idf

        RETURN
            count(CASE WHEN idf > 5.0 THEN 1 END) as hard_count,
            count(CASE WHEN idf > 4.0 AND idf <= 5.0 THEN 1 END) as medium_count,
            count(CASE WHEN idf > 3.5 AND idf <= 4.0 THEN 1 END) as easy_count,
            count(CASE WHEN idf <= 3.5 THEN 1 END) as generic_count,
            count(*) as total_count
        """

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, org1=org1, org2=org2, total_orgs=self.config.total_orgs)
            record = result.single()

            if record:
                return {
                    'hard': record['hard_count'],
                    'medium': record['medium_count'],
                    'easy': record['easy_count'],
                    'generic': record['generic_count'],
                    'total': record['total_count']
                }
            return {'hard': 0, 'medium': 0, 'easy': 0, 'generic': 0, 'total': 0}

    def find_connectors_by_year(
        self,
        org1: str,
        org2: str,
        year1: int,
        year2: int,
        connector_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ConnectorResult]:
        """
        Find connectors between two companies filtered by year (Type 1 pairs).

        Args:
            org1: First company ticker
            org2: Second company ticker
            year1: Year for org1's evidence (2022, 2023, or 2024)
            year2: Year for org2's evidence (2022, 2023, or 2024)
            connector_types: List of connector types to consider
            min_idf: Minimum IDF threshold
            limit: Maximum number of connectors to return

        Returns:
            List of ConnectorResult objects with year-filtered evidence
        """
        if connector_types is None:
            connector_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION',
                            'RAW_MATERIAL', 'FIN_INST', 'REGULATORY_REQUIREMENT', 'ORG_REG']

        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs

        # Build year filter patterns
        year1_pattern = f".*{year1}.*"
        year2_pattern = f".*{year2}.*"

        query = """
        MATCH (o1:ORG {name: $org1})-[r1]-(x)-[r2]-(o2:ORG {name: $org2})
        WHERE any(label IN labels(x) WHERE label IN $valid_types)
          AND r1.source_file =~ $year1_pattern
          AND r2.source_file =~ $year2_pattern

        WITH x,
             type(r1) as rel1, type(r2) as rel2,
             r1.source_file as src1, r1.page_id as page1, r1.chunk_id as chunk1,
             r2.source_file as src2, r2.page_id as page2, r2.chunk_id as chunk2,
             r1.context as ctx1, r2.context as ctx2

        MATCH (x)--(o:ORG)
        WITH x, rel1, rel2, src1, page1, chunk1, src2, page2, chunk2, ctx1, ctx2,
             count(DISTINCT o) as org_degree

        WITH x, rel1, rel2, src1, page1, chunk1, src2, page2, chunk2, ctx1, ctx2,
             org_degree, log(toFloat($total_orgs) / org_degree) as idf_score
        WHERE idf_score > $min_idf

        RETURN
            x.name as connector_name,
            [l IN labels(x) WHERE l <> 'Entity'][0] as connector_type,
            org_degree,
            idf_score,
            rel1, src1, page1, chunk1, ctx1,
            rel2, src2, page2, chunk2, ctx2
        ORDER BY idf_score DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org1=org1,
                org2=org2,
                valid_types=connector_types,
                total_orgs=total_orgs,
                min_idf=min_idf,
                year1_pattern=year1_pattern,
                year2_pattern=year2_pattern,
                limit=limit
            )

            # Track seen connectors + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_connectors = set()

            for record in records:
                # Deduplicate by (connector, page1, page2)
                key = (record["connector_name"], record["page1"], record["page2"])
                if key in seen_connectors:
                    continue
                seen_connectors.add(key)

                result = ConnectorResult(
                    connector_name=record["connector_name"],
                    connector_type=record["connector_type"],
                    org_degree=record["org_degree"],
                    idf_score=record["idf_score"],
                    org1_relationship=record["rel1"],
                    org1_source_file=record["src1"],
                    org1_page_id=record["page1"],
                    org1_chunk_id=record["chunk1"],
                    org2_relationship=record["rel2"],
                    org2_source_file=record["src2"],
                    org2_page_id=record["page2"],
                    org2_chunk_id=record["chunk2"],
                    org1_context=record["ctx1"],
                    org2_context=record["ctx2"]
                )
                results.append(result)

        logger.info(f"Found {len(results)} unique connectors between {org1}({year1}) and {org2}({year2}) with IDF > {min_idf}")
        return results

    def find_temporal_connectors(
        self,
        org: str,
        year1: int,
        year2: int,
        connector_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ConnectorResult]:
        """
        Find connectors for same company across different years (Type 2 pairs).
        Returns connectors where the company's relationship may have changed over time.

        Args:
            org: Company ticker
            year1: First year (e.g., 2022)
            year2: Second year (e.g., 2024)
            connector_types: List of connector types to consider
            min_idf: Minimum IDF threshold
            limit: Maximum number of connectors to return

        Returns:
            List of ConnectorResult objects where org1 and org2 are the same company
            but from different years
        """
        if connector_types is None:
            connector_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION',
                            'RAW_MATERIAL', 'FIN_INST', 'REGULATORY_REQUIREMENT', 'ORG_REG']

        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs

        year1_pattern = f".*{year1}.*"
        year2_pattern = f".*{year2}.*"

        query = """
        // Find connectors mentioned by the same company in both years
        MATCH (o:ORG {name: $org})-[r1]-(x)
        WHERE any(label IN labels(x) WHERE label IN $valid_types)
          AND r1.source_file =~ $year1_pattern

        WITH o, x, r1
        MATCH (o)-[r2]-(x)
        WHERE r2.source_file =~ $year2_pattern

        WITH x,
             type(r1) as rel1, type(r2) as rel2,
             r1.source_file as src1, r1.page_id as page1, r1.chunk_id as chunk1,
             r2.source_file as src2, r2.page_id as page2, r2.chunk_id as chunk2,
             r1.context as ctx1, r2.context as ctx2

        MATCH (x)--(any_org:ORG)
        WITH x, rel1, rel2, src1, page1, chunk1, src2, page2, chunk2, ctx1, ctx2,
             count(DISTINCT any_org) as org_degree

        WITH x, rel1, rel2, src1, page1, chunk1, src2, page2, chunk2, ctx1, ctx2,
             org_degree, log(toFloat($total_orgs) / org_degree) as idf_score
        WHERE idf_score > $min_idf

        RETURN
            x.name as connector_name,
            [l IN labels(x) WHERE l <> 'Entity'][0] as connector_type,
            org_degree,
            idf_score,
            rel1, src1, page1, chunk1, ctx1,
            rel2, src2, page2, chunk2, ctx2,
            CASE WHEN rel1 <> rel2 THEN true ELSE false END as relationship_changed
        ORDER BY relationship_changed DESC, idf_score DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org=org,
                valid_types=connector_types,
                total_orgs=total_orgs,
                min_idf=min_idf,
                year1_pattern=year1_pattern,
                year2_pattern=year2_pattern,
                limit=limit
            )

            # Track seen connectors + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_connectors = set()

            for record in records:
                # Deduplicate by (connector, page1, page2)
                key = (record["connector_name"], record["page1"], record["page2"])
                if key in seen_connectors:
                    continue
                seen_connectors.add(key)

                result = ConnectorResult(
                    connector_name=record["connector_name"],
                    connector_type=record["connector_type"],
                    org_degree=record["org_degree"],
                    idf_score=record["idf_score"],
                    org1_relationship=record["rel1"],
                    org1_source_file=record["src1"],
                    org1_page_id=record["page1"],
                    org1_chunk_id=record["chunk1"],
                    org2_relationship=record["rel2"],
                    org2_source_file=record["src2"],
                    org2_page_id=record["page2"],
                    org2_chunk_id=record["chunk2"],
                    org1_context=record["ctx1"],
                    org2_context=record["ctx2"]
                )
                results.append(result)

        changed_count = sum(1 for r in results if r.has_asymmetric_relationships)
        logger.info(f"Found {len(results)} unique temporal connectors for {org} ({year1} vs {year2}), {changed_count} with relationship changes")
        return results

    def find_intra_doc_connectors(
        self,
        org: str,
        year: int,
        connector_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ConnectorResult]:
        """
        Find connectors for same company, same year, different pages/chunks (Type 3 / intra-doc pairs).

        This pattern finds entities mentioned multiple times within the same 10-K document
        but in different sections/pages, enabling questions about how the company discusses
        the same topic in different contexts.

        Args:
            org: Company ticker
            year: Year of the document (2022, 2023, or 2024)
            connector_types: List of connector types to consider
            min_idf: Minimum IDF threshold
            limit: Maximum number of connectors to return

        Returns:
            List of ConnectorResult objects where org1 and org2 are the same company,
            same year, but from different pages/chunks
        """
        if connector_types is None:
            connector_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION',
                            'RAW_MATERIAL', 'FIN_INST', 'REGULATORY_REQUIREMENT', 'ORG_REG']

        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs
        year_pattern = f".*{year}.*"

        query = """
        // Find connectors mentioned by the same company in the same year but different pages/chunks
        MATCH (o:ORG {name: $org})-[r1]-(x)-[r2]-(o)
        WHERE any(label IN labels(x) WHERE label IN $valid_types)
          AND r1.source_file =~ $year_pattern
          AND r2.source_file =~ $year_pattern
          AND r1.source_file = r2.source_file
          AND (
            r1.page_id <> r2.page_id
            OR (r1.page_id = r2.page_id AND r1.chunk_id <> r2.chunk_id)
          )
          AND elementId(r1) < elementId(r2)

        WITH x,
             type(r1) as rel1, type(r2) as rel2,
             r1.source_file as src1, r1.page_id as page1, r1.chunk_id as chunk1,
             r2.source_file as src2, r2.page_id as page2, r2.chunk_id as chunk2,
             r1.context as ctx1, r2.context as ctx2

        MATCH (x)--(any_org:ORG)
        WITH x, rel1, rel2, src1, page1, chunk1, src2, page2, chunk2, ctx1, ctx2,
             count(DISTINCT any_org) as org_degree

        WITH x, rel1, rel2, src1, page1, chunk1, src2, page2, chunk2, ctx1, ctx2,
             org_degree, log(toFloat($total_orgs) / org_degree) as idf_score
        WHERE idf_score > $min_idf

        RETURN
            x.name as connector_name,
            [l IN labels(x) WHERE l <> 'Entity'][0] as connector_type,
            org_degree,
            idf_score,
            rel1, src1, page1, chunk1, ctx1,
            rel2, src2, page2, chunk2, ctx2,
            CASE WHEN rel1 <> rel2 THEN true ELSE false END as relationship_changed,
            CASE WHEN page1 <> page2 THEN 'page' ELSE 'chunk' END as difference_type
        ORDER BY relationship_changed DESC, idf_score DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org=org,
                valid_types=connector_types,
                total_orgs=total_orgs,
                min_idf=min_idf,
                year_pattern=year_pattern,
                limit=limit
            )

            # Track seen connectors + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_connectors = set()

            for record in records:
                # Deduplicate by (connector, page1, page2)
                key = (record["connector_name"], record["page1"], record["page2"])
                if key in seen_connectors:
                    continue
                seen_connectors.add(key)

                result = ConnectorResult(
                    connector_name=record["connector_name"],
                    connector_type=record["connector_type"],
                    org_degree=record["org_degree"],
                    idf_score=record["idf_score"],
                    org1_relationship=record["rel1"],
                    org1_source_file=record["src1"],
                    org1_page_id=record["page1"],
                    org1_chunk_id=record["chunk1"],
                    org2_relationship=record["rel2"],
                    org2_source_file=record["src2"],
                    org2_page_id=record["page2"],
                    org2_chunk_id=record["chunk2"],
                    org1_context=record["ctx1"],
                    org2_context=record["ctx2"]
                )
                results.append(result)

        changed_count = sum(1 for r in results if r.has_asymmetric_relationships)
        logger.info(f"Found {len(results)} unique intra-doc connectors for {org} ({year}), {changed_count} with relationship changes")
        return results

    def get_intra_doc_stats(self, org: str, year: int) -> Dict[str, Any]:
        """
        Get statistics about intra-document connectors for a company.

        Returns:
            Dictionary with connector counts and relationship change stats
        """
        query = """
        MATCH (o:ORG {name: $org})-[r1]-(x)-[r2]-(o)
        WHERE (x:COMP OR x:RISK_FACTOR OR x:MACRO_CONDITION OR x:RAW_MATERIAL
               OR x:FIN_INST OR x:REGULATORY_REQUIREMENT OR x:ORG_REG)
          AND r1.source_file =~ $year_pattern
          AND r2.source_file =~ $year_pattern
          AND r1.source_file = r2.source_file
          AND (
            r1.page_id <> r2.page_id
            OR (r1.page_id = r2.page_id AND r1.chunk_id <> r2.chunk_id)
          )
          AND elementId(r1) < elementId(r2)

        WITH DISTINCT x, type(r1) as rel1, type(r2) as rel2

        MATCH (x)--(any_org:ORG)
        WITH x, rel1, rel2, count(DISTINCT any_org) as org_degree
        WITH x, rel1, rel2, org_degree, log(toFloat($total_orgs) / org_degree) as idf

        RETURN
            count(CASE WHEN idf > 5.0 THEN 1 END) as hard_count,
            count(CASE WHEN idf > 4.0 AND idf <= 5.0 THEN 1 END) as medium_count,
            count(CASE WHEN idf > 5.0 AND rel1 <> rel2 THEN 1 END) as hard_changed,
            count(CASE WHEN idf > 4.0 AND rel1 <> rel2 THEN 1 END) as medium_plus_changed,
            count(*) as total_count
        """

        year_pattern = f".*{year}.*"

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(
                query,
                org=org,
                year_pattern=year_pattern,
                total_orgs=self.config.total_orgs
            )
            record = result.single()

            if record:
                return {
                    'hard': record['hard_count'],
                    'medium': record['medium_count'],
                    'hard_changed': record['hard_changed'],
                    'medium_plus_changed': record['medium_plus_changed'],
                    'total': record['total_count']
                }
            return {'hard': 0, 'medium': 0, 'hard_changed': 0, 'medium_plus_changed': 0, 'total': 0}

    def get_temporal_stats(self, org: str, year1: int, year2: int) -> Dict[str, Any]:
        """
        Get statistics about temporal connectors for a company.

        Returns:
            Dictionary with connector counts and relationship change stats
        """
        query = """
        MATCH (o:ORG {name: $org})-[r1]-(x)
        WHERE (x:COMP OR x:RISK_FACTOR OR x:MACRO_CONDITION OR x:RAW_MATERIAL
               OR x:FIN_INST OR x:REGULATORY_REQUIREMENT OR x:ORG_REG)
          AND r1.source_file =~ $year1_pattern

        WITH o, x, r1
        MATCH (o)-[r2]-(x)
        WHERE r2.source_file =~ $year2_pattern

        WITH DISTINCT x, type(r1) as rel1, type(r2) as rel2

        MATCH (x)--(any_org:ORG)
        WITH x, rel1, rel2, count(DISTINCT any_org) as org_degree
        WITH x, rel1, rel2, org_degree, log(toFloat($total_orgs) / org_degree) as idf

        RETURN
            count(CASE WHEN idf > 5.0 THEN 1 END) as hard_count,
            count(CASE WHEN idf > 4.0 AND idf <= 5.0 THEN 1 END) as medium_count,
            count(CASE WHEN idf > 5.0 AND rel1 <> rel2 THEN 1 END) as hard_changed,
            count(CASE WHEN idf > 4.0 AND rel1 <> rel2 THEN 1 END) as medium_plus_changed,
            count(*) as total_count
        """

        year1_pattern = f".*{year1}.*"
        year2_pattern = f".*{year2}.*"

        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(
                query,
                org=org,
                year1_pattern=year1_pattern,
                year2_pattern=year2_pattern,
                total_orgs=self.config.total_orgs
            )
            record = result.single()

            if record:
                return {
                    'hard': record['hard_count'],
                    'medium': record['medium_count'],
                    'hard_changed': record['hard_changed'],
                    'medium_plus_changed': record['medium_plus_changed'],
                    'total': record['total_count']
                }
            return {'hard': 0, 'medium': 0, 'hard_changed': 0, 'medium_plus_changed': 0, 'total': 0}

    # =========================================================================
    # 3-HOP DISCOVERY METHODS
    # =========================================================================

    def find_3hop_cross_company_type1(
        self,
        org1: str,
        org2: str,
        year1: Optional[int] = None,
        year2: Optional[int] = None,
        connector1_types: Optional[List[str]] = None,
        connector2_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ThreeHopResult]:
        """
        Find 3-hop paths: ORG1 → Connector1 → Connector2 ← ORG2

        This pattern finds paths where two companies connect through a chain
        of two intermediate entities.

        Example: NVDA → TSMC → Silicon_Wafers ← AMD
        """
        if connector1_types is None:
            connector1_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL']
        if connector2_types is None:
            connector2_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL', 'FIN_INST']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs

        # Build year patterns if specified
        year1_filter = f"AND r1.source_file =~ '.*{year1}.*'" if year1 else ""
        year2_filter = f"AND r3.source_file =~ '.*{year2}.*'" if year2 else ""

        query = f"""
        // 3-hop: ORG1 → C1 → C2 ← ORG2
        // IMPORTANT: All relationships must come from either ORG1's or ORG2's document (no third-party)
        MATCH (o1:ORG {{name: $org1}})-[r1]-(c1)-[r2]-(c2)-[r3]-(o2:ORG {{name: $org2}})
        WHERE any(label IN labels(c1) WHERE label IN $c1_types)
          AND any(label IN labels(c2) WHERE label IN $c2_types)
          AND c1 <> c2
          AND (r1.source_file CONTAINS $org1 OR r1.source_file CONTAINS $org2)
          AND (r2.source_file CONTAINS $org1 OR r2.source_file CONTAINS $org2)
          AND (r3.source_file CONTAINS $org1 OR r3.source_file CONTAINS $org2)
          {year1_filter}
          {year2_filter}

        // Calculate IDF for both connectors
        WITH c1, c2, r1, r2, r3, o1, o2
        MATCH (c1)--(org1_all:ORG)
        WITH c1, c2, r1, r2, r3, o1, o2, count(DISTINCT org1_all) as c1_degree
        MATCH (c2)--(org2_all:ORG)
        WITH c1, c2, r1, r2, r3, o1, o2, c1_degree, count(DISTINCT org2_all) as c2_degree

        WITH c1, c2, r1, r2, r3, o1, o2,
             c1_degree, c2_degree,
             log(toFloat($total_orgs) / c1_degree) as c1_idf,
             log(toFloat($total_orgs) / c2_degree) as c2_idf

        WHERE c1_idf > $min_idf AND c2_idf > $min_idf

        RETURN
            c1.name as c1_name, [l IN labels(c1) WHERE l <> 'Entity'][0] as c1_type,
            c1_degree, c1_idf,
            c2.name as c2_name, [l IN labels(c2) WHERE l <> 'Entity'][0] as c2_type,
            c2_degree, c2_idf,
            type(r1) as rel1, r1.source_file as src1, r1.page_id as page1, r1.chunk_id as chunk1,
            type(r2) as rel2, r2.source_file as src2, r2.page_id as page2, r2.chunk_id as chunk2,
            type(r3) as rel3, r3.source_file as src3, r3.page_id as page3, r3.chunk_id as chunk3,
            (c1_idf + c2_idf) / 2 as combined_idf
        ORDER BY combined_idf DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org1=org1, org2=org2,
                c1_types=connector1_types, c2_types=connector2_types,
                total_orgs=total_orgs, min_idf=min_idf, limit=limit
            )

            # Track seen connector pairs + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_pairs = set()

            for rec in records:
                # Deduplicate by (connector1, connector2, hop1_page, hop3_page)
                # This keeps different evidence pages for the same connector pair
                pair_key = (rec["c1_name"], rec["c2_name"], rec["page1"], rec["page3"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                result = ThreeHopResult(
                    pattern_type="cross_company_type1",
                    connector1_name=rec["c1_name"],
                    connector1_type=rec["c1_type"],
                    connector1_idf=rec["c1_idf"],
                    connector1_org_degree=rec["c1_degree"],
                    hop1_org=org1,
                    hop1_relationship=rec["rel1"],
                    hop1_source_file=rec["src1"],
                    hop1_page_id=rec["page1"],
                    hop1_chunk_id=rec["chunk1"],
                    hop2_entity_name=rec["c2_name"],
                    hop2_entity_type=rec["c2_type"],
                    hop2_relationship=rec["rel2"],
                    hop2_source_file=rec["src2"],
                    hop2_page_id=rec["page2"],
                    hop2_chunk_id=rec["chunk2"],
                    hop3_entity_name=rec["c2_name"],
                    hop3_entity_type=rec["c2_type"],
                    hop3_org=org2,
                    hop3_relationship=rec["rel3"],
                    hop3_source_file=rec["src3"],
                    hop3_page_id=rec["page3"],
                    hop3_chunk_id=rec["chunk3"],
                    combined_idf_score=rec["combined_idf"],
                    hop3_entity_idf=rec["c2_idf"]
                )
                results.append(result)

        logger.info(f"Found {len(results)} unique 3-hop cross_company_type1 paths: {org1} → C1 → C2 ← {org2}")
        return results

    def find_3hop_cross_year_type1(
        self,
        org: str,
        year1: int,
        year2: int,
        connector1_types: Optional[List[str]] = None,
        connector2_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ThreeHopResult]:
        """
        Find 3-hop paths: ORG(y1) → Connector1 → Connector2 ← ORG(y2)

        Same company, different years, connector chain pattern.
        The C1-C2 link must be from the same year as r1 OR r2.

        Example: NVDA(2022) → TSMC → Silicon_Wafers ← NVDA(2024)
        """
        if connector1_types is None:
            connector1_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL']
        if connector2_types is None:
            connector2_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL', 'FIN_INST']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs

        query = f"""
        // 3-hop Cross Year Type 1: ORG(y1) → C1 → C2 ← ORG(y2)
        // IMPORTANT: All relationships must come from this ORG's documents (no third-party)
        MATCH (o:ORG {{name: $org}})-[r1]-(c1)-[r2]-(c2)-[r3]-(o)
        WHERE any(label IN labels(c1) WHERE label IN $c1_types)
          AND any(label IN labels(c2) WHERE label IN $c2_types)
          AND c1 <> c2
          AND r1.source_file CONTAINS $org
          AND r1.source_file =~ '.*{year1}.*'
          AND r3.source_file CONTAINS $org
          AND r3.source_file =~ '.*{year2}.*'
          AND r2.source_file CONTAINS $org
          AND (r2.source_file =~ '.*{year1}.*' OR r2.source_file =~ '.*{year2}.*')

        // Calculate IDF for both connectors
        WITH c1, c2, r1, r2, r3, o
        MATCH (c1)--(org1_all:ORG)
        WITH c1, c2, r1, r2, r3, o, count(DISTINCT org1_all) as c1_degree
        MATCH (c2)--(org2_all:ORG)
        WITH c1, c2, r1, r2, r3, o, c1_degree, count(DISTINCT org2_all) as c2_degree

        WITH c1, c2, r1, r2, r3, o,
             c1_degree, c2_degree,
             log(toFloat($total_orgs) / c1_degree) as c1_idf,
             log(toFloat($total_orgs) / c2_degree) as c2_idf

        WHERE c1_idf > $min_idf AND c2_idf > $min_idf

        RETURN
            c1.name as c1_name, [l IN labels(c1) WHERE l <> 'Entity'][0] as c1_type,
            c1_degree, c1_idf,
            c2.name as c2_name, [l IN labels(c2) WHERE l <> 'Entity'][0] as c2_type,
            c2_degree, c2_idf,
            type(r1) as rel1, r1.source_file as src1, r1.page_id as page1, r1.chunk_id as chunk1,
            type(r2) as rel2, r2.source_file as src2, r2.page_id as page2, r2.chunk_id as chunk2,
            type(r3) as rel3, r3.source_file as src3, r3.page_id as page3, r3.chunk_id as chunk3,
            (c1_idf + c2_idf) / 2 as combined_idf
        ORDER BY combined_idf DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org=org,
                c1_types=connector1_types, c2_types=connector2_types,
                total_orgs=total_orgs, min_idf=min_idf, limit=limit
            )

            # Track seen connector pairs + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_pairs = set()

            for rec in records:
                # Deduplicate by (connector1, connector2, hop1_page, hop3_page)
                pair_key = (rec["c1_name"], rec["c2_name"], rec["page1"], rec["page3"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                result = ThreeHopResult(
                    pattern_type="cross_year_type1",
                    connector1_name=rec["c1_name"],
                    connector1_type=rec["c1_type"],
                    connector1_idf=rec["c1_idf"],
                    connector1_org_degree=rec["c1_degree"],
                    hop1_org=org,
                    hop1_relationship=rec["rel1"],
                    hop1_source_file=rec["src1"],
                    hop1_page_id=rec["page1"],
                    hop1_chunk_id=rec["chunk1"],
                    hop2_entity_name=rec["c2_name"],
                    hop2_entity_type=rec["c2_type"],
                    hop2_relationship=rec["rel2"],
                    hop2_source_file=rec["src2"],
                    hop2_page_id=rec["page2"],
                    hop2_chunk_id=rec["chunk2"],
                    hop3_entity_name=rec["c2_name"],
                    hop3_entity_type=rec["c2_type"],
                    hop3_org=org,
                    hop3_relationship=rec["rel3"],
                    hop3_source_file=rec["src3"],
                    hop3_page_id=rec["page3"],
                    hop3_chunk_id=rec["chunk3"],
                    combined_idf_score=rec["combined_idf"],
                    hop3_entity_idf=rec["c2_idf"]
                )
                results.append(result)

        logger.info(f"Found {len(results)} unique 3-hop cross_year_type1 paths: {org}({year1}) → C1 → C2 ← {org}({year2})")
        return results

    def find_3hop_intra_doc_type1(
        self,
        org: str,
        year: int,
        connector1_types: Optional[List[str]] = None,
        connector2_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ThreeHopResult]:
        """
        Find 3-hop paths: ORG(p1) → Connector1 → Connector2 ← ORG(p2)

        Same company, same year, different pages, connector chain pattern.
        All relationships must be from the same document.

        Example: NVDA(page5) → TSMC → Silicon_Wafers ← NVDA(page42)
        """
        if connector1_types is None:
            connector1_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL']
        if connector2_types is None:
            connector2_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL', 'FIN_INST']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs
        year_pattern = f".*{year}.*"

        query = f"""
        // 3-hop Intra Doc Type 1: ORG(p1) → C1 → C2 ← ORG(p2)
        // IMPORTANT: All relationships must come from this ORG's document (same file, different pages)
        MATCH (o:ORG {{name: $org}})-[r1]-(c1)-[r2]-(c2)-[r3]-(o)
        WHERE any(label IN labels(c1) WHERE label IN $c1_types)
          AND any(label IN labels(c2) WHERE label IN $c2_types)
          AND c1 <> c2
          AND r1.source_file CONTAINS $org
          AND r1.source_file =~ $year_pattern
          AND r3.source_file =~ $year_pattern
          AND r1.source_file = r3.source_file
          AND r2.source_file = r1.source_file
          AND (r1.page_id <> r3.page_id OR r1.chunk_id <> r3.chunk_id)

        // Calculate IDF for both connectors
        WITH c1, c2, r1, r2, r3, o
        MATCH (c1)--(org1_all:ORG)
        WITH c1, c2, r1, r2, r3, o, count(DISTINCT org1_all) as c1_degree
        MATCH (c2)--(org2_all:ORG)
        WITH c1, c2, r1, r2, r3, o, c1_degree, count(DISTINCT org2_all) as c2_degree

        WITH c1, c2, r1, r2, r3, o,
             c1_degree, c2_degree,
             log(toFloat($total_orgs) / c1_degree) as c1_idf,
             log(toFloat($total_orgs) / c2_degree) as c2_idf

        WHERE c1_idf > $min_idf AND c2_idf > $min_idf

        RETURN
            c1.name as c1_name, [l IN labels(c1) WHERE l <> 'Entity'][0] as c1_type,
            c1_degree, c1_idf,
            c2.name as c2_name, [l IN labels(c2) WHERE l <> 'Entity'][0] as c2_type,
            c2_degree, c2_idf,
            type(r1) as rel1, r1.source_file as src1, r1.page_id as page1, r1.chunk_id as chunk1,
            type(r2) as rel2, r2.source_file as src2, r2.page_id as page2, r2.chunk_id as chunk2,
            type(r3) as rel3, r3.source_file as src3, r3.page_id as page3, r3.chunk_id as chunk3,
            (c1_idf + c2_idf) / 2 as combined_idf
        ORDER BY combined_idf DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org=org,
                c1_types=connector1_types, c2_types=connector2_types,
                total_orgs=total_orgs, min_idf=min_idf, limit=limit,
                year_pattern=year_pattern
            )

            # Track seen connector pairs + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_pairs = set()

            for rec in records:
                # Deduplicate by (connector1, connector2, hop1_page, hop3_page)
                pair_key = (rec["c1_name"], rec["c2_name"], rec["page1"], rec["page3"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                result = ThreeHopResult(
                    pattern_type="intra_doc_type1",
                    connector1_name=rec["c1_name"],
                    connector1_type=rec["c1_type"],
                    connector1_idf=rec["c1_idf"],
                    connector1_org_degree=rec["c1_degree"],
                    hop1_org=org,
                    hop1_relationship=rec["rel1"],
                    hop1_source_file=rec["src1"],
                    hop1_page_id=rec["page1"],
                    hop1_chunk_id=rec["chunk1"],
                    hop2_entity_name=rec["c2_name"],
                    hop2_entity_type=rec["c2_type"],
                    hop2_relationship=rec["rel2"],
                    hop2_source_file=rec["src2"],
                    hop2_page_id=rec["page2"],
                    hop2_chunk_id=rec["chunk2"],
                    hop3_entity_name=rec["c2_name"],
                    hop3_entity_type=rec["c2_type"],
                    hop3_org=org,
                    hop3_relationship=rec["rel3"],
                    hop3_source_file=rec["src3"],
                    hop3_page_id=rec["page3"],
                    hop3_chunk_id=rec["chunk3"],
                    combined_idf_score=rec["combined_idf"],
                    hop3_entity_idf=rec["c2_idf"]
                )
                results.append(result)

        logger.info(f"Found {len(results)} unique 3-hop intra_doc_type1 paths: {org}({year}) → C1 → C2 ← {org} (diff pages)")
        return results

    # =========================================================================
    # TYPE 2 PATTERNS: Multi-Branch (3 anchors → 1 connector)
    # =========================================================================

    def find_3hop_cross_company_type2(
        self,
        org1: str,
        org2: str,
        org3: str,
        year: int,
        connector_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ThreeHopType2Result]:
        """
        Find 3-hop Type 2 paths: 3 different ORGs → 1 Connector (same year)

        Pattern:        ORG1
                         │
                       [r1]
                         │
                ORG2 ─[r2]─ Connector ─[r3]─ ORG3

        All 3 ORGs must be different, all from the same year.
        """
        if connector_types is None:
            connector_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL', 'PRODUCT']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs
        year_pattern = f".*{year}.*"

        query = """
        // Type 2: 3 ORGs → 1 Connector
        // IMPORTANT: Each relationship must come from its respective ORG's document
        MATCH (o1:ORG {name: $org1})-[r1]-(c)-[r2]-(o2:ORG {name: $org2})
        MATCH (c)-[r3]-(o3:ORG {name: $org3})
        WHERE any(label IN labels(c) WHERE label IN $valid_types)
          AND r1.source_file =~ $year_pattern
          AND r2.source_file =~ $year_pattern
          AND r3.source_file =~ $year_pattern
          AND r1.source_file CONTAINS $org1
          AND r2.source_file CONTAINS $org2
          AND r3.source_file CONTAINS $org3
          AND elementId(r1) <> elementId(r2) AND elementId(r2) <> elementId(r3) AND elementId(r1) <> elementId(r3)

        // IDF calculation
        WITH c, r1, r2, r3
        MATCH (c)--(any_org:ORG)
        WITH c, r1, r2, r3, count(DISTINCT any_org) as org_degree
        WITH c, r1, r2, r3, org_degree, log(toFloat($total_orgs) / org_degree) as idf_score
        WHERE idf_score >= $min_idf

        RETURN
            c.name as connector_name,
            [l IN labels(c) WHERE l <> 'Entity'][0] as connector_type,
            org_degree, idf_score,
            type(r1) as rel1, r1.source_file as src1, r1.page_id as page1, r1.chunk_id as chunk1,
            type(r2) as rel2, r2.source_file as src2, r2.page_id as page2, r2.chunk_id as chunk2,
            type(r3) as rel3, r3.source_file as src3, r3.page_id as page3, r3.chunk_id as chunk3
        ORDER BY idf_score DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org1=org1,
                org2=org2,
                org3=org3,
                valid_types=connector_types,
                year_pattern=year_pattern,
                total_orgs=total_orgs,
                min_idf=min_idf,
                limit=limit
            )

            # Track seen connectors + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_connectors = set()

            for rec in records:
                # Deduplicate by (connector, page1, page2, page3)
                key = (rec["connector_name"], rec["page1"], rec["page2"], rec["page3"])
                if key in seen_connectors:
                    continue
                seen_connectors.add(key)

                result = ThreeHopType2Result(
                    pattern_type='cross_company_type2',
                    connector_name=rec["connector_name"],
                    connector_type=rec["connector_type"],
                    connector_idf=rec["idf_score"],
                    connector_org_degree=rec["org_degree"],
                    hop1_org=org1,
                    hop1_relationship=rec["rel1"],
                    hop1_source_file=rec["src1"],
                    hop1_page_id=rec["page1"],
                    hop1_chunk_id=rec["chunk1"],
                    hop2_org=org2,
                    hop2_relationship=rec["rel2"],
                    hop2_source_file=rec["src2"],
                    hop2_page_id=rec["page2"],
                    hop2_chunk_id=rec["chunk2"],
                    hop3_org=org3,
                    hop3_relationship=rec["rel3"],
                    hop3_source_file=rec["src3"],
                    hop3_page_id=rec["page3"],
                    hop3_chunk_id=rec["chunk3"]
                )
                results.append(result)

        logger.info(f"Found {len(results)} unique 3-hop cross_company_type2 paths: {org1}, {org2}, {org3} → Connector ({year})")
        return results

    def find_3hop_cross_year_type2(
        self,
        org: str,
        year1: int,
        year2: int,
        year3: int,
        connector_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ThreeHopType2Result]:
        """
        Find 3-hop Type 2 paths: Same ORG, 3 different years → 1 Connector

        Pattern:        ORG(y1)
                         │
                       [r1]
                         │
                ORG(y2) ─[r2]─ Connector ─[r3]─ ORG(y3)

        Same ORG in all 3 hops, but from 3 different years.
        """
        if connector_types is None:
            connector_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'FIN_METRIC', 'SEGMENT']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs
        year1_pattern = f".*{year1}.*"
        year2_pattern = f".*{year2}.*"
        year3_pattern = f".*{year3}.*"

        query = """
        // Type 2: Same ORG, 3 different years → 1 Connector
        // CONNECTOR-FIRST approach: Find connectors, then get relationships by year
        // Step 1: Find connectors connected to this ORG with relationships from all 3 years
        MATCH (o:ORG {name: $org})-[r]-(c)
        WHERE any(label IN labels(c) WHERE label IN $valid_types)
          AND r.source_file CONTAINS $org

        // Collect relationships grouped by year pattern
        WITH c, o, collect({
            rel: r,
            relType: type(r),
            src: r.source_file,
            page: r.page_id,
            chunk: r.chunk_id,
            isYear1: r.source_file =~ $year1_pattern,
            isYear2: r.source_file =~ $year2_pattern,
            isYear3: r.source_file =~ $year3_pattern
        }) as all_rels

        // Filter for year1, year2, year3 relationships
        WITH c, o,
             [r IN all_rels WHERE r.isYear1][0] as r1_data,
             [r IN all_rels WHERE r.isYear2][0] as r2_data,
             [r IN all_rels WHERE r.isYear3][0] as r3_data

        // Must have relationships from all 3 years
        WHERE r1_data IS NOT NULL AND r2_data IS NOT NULL AND r3_data IS NOT NULL

        // IDF calculation
        WITH c, r1_data, r2_data, r3_data
        MATCH (c)--(any_org:ORG)
        WITH c, r1_data, r2_data, r3_data, count(DISTINCT any_org) as org_degree
        WITH c, r1_data, r2_data, r3_data, org_degree, log(toFloat($total_orgs) / org_degree) as idf_score
        WHERE idf_score >= $min_idf

        RETURN
            c.name as connector_name,
            [l IN labels(c) WHERE l <> 'Entity'][0] as connector_type,
            org_degree, idf_score,
            r1_data.relType as rel1, r1_data.src as src1, r1_data.page as page1, r1_data.chunk as chunk1,
            r2_data.relType as rel2, r2_data.src as src2, r2_data.page as page2, r2_data.chunk as chunk2,
            r3_data.relType as rel3, r3_data.src as src3, r3_data.page as page3, r3_data.chunk as chunk3
        ORDER BY idf_score DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org=org,
                valid_types=connector_types,
                year1_pattern=year1_pattern,
                year2_pattern=year2_pattern,
                year3_pattern=year3_pattern,
                total_orgs=total_orgs,
                min_idf=min_idf,
                limit=limit
            )

            # Track seen connectors + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_connectors = set()

            for rec in records:
                # Deduplicate by (connector, page1, page2, page3)
                key = (rec["connector_name"], rec["page1"], rec["page2"], rec["page3"])
                if key in seen_connectors:
                    continue
                seen_connectors.add(key)

                result = ThreeHopType2Result(
                    pattern_type='cross_year_type2',
                    connector_name=rec["connector_name"],
                    connector_type=rec["connector_type"],
                    connector_idf=rec["idf_score"],
                    connector_org_degree=rec["org_degree"],
                    hop1_org=org,
                    hop1_relationship=rec["rel1"],
                    hop1_source_file=rec["src1"],
                    hop1_page_id=rec["page1"],
                    hop1_chunk_id=rec["chunk1"],
                    hop2_org=org,
                    hop2_relationship=rec["rel2"],
                    hop2_source_file=rec["src2"],
                    hop2_page_id=rec["page2"],
                    hop2_chunk_id=rec["chunk2"],
                    hop3_org=org,
                    hop3_relationship=rec["rel3"],
                    hop3_source_file=rec["src3"],
                    hop3_page_id=rec["page3"],
                    hop3_chunk_id=rec["chunk3"]
                )
                results.append(result)

        logger.info(f"Found {len(results)} unique 3-hop cross_year_type2 paths: {org}({year1}, {year2}, {year3}) → Connector")
        return results

    def find_3hop_intra_doc_type2(
        self,
        org: str,
        year: int,
        connector_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        limit: int = 50
    ) -> List[ThreeHopType2Result]:
        """
        Find 3-hop Type 2 paths: Same ORG, same year, 3 different pages → 1 Connector

        Pattern:        ORG(p1)
                         │
                       [r1]
                         │
                ORG(p2) ─[r2]─ Connector ─[r3]─ ORG(p3)

        Same ORG, same document/year, but 3 different pages.
        """
        if connector_types is None:
            connector_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'PRODUCT', 'FIN_INST']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs
        year_pattern = f".*{year}.*"

        query = """
        // Type 2: Same ORG, same year, 3 different pages → 1 Connector
        // CONNECTOR-FIRST approach: Find connectors, then get relationships by page
        // Step 1: Find connectors connected to this ORG from this year's document
        MATCH (o:ORG {name: $org})-[r]-(c)
        WHERE any(label IN labels(c) WHERE label IN $valid_types)
          AND r.source_file =~ $year_pattern
          AND r.source_file CONTAINS $org

        // Collect relationships grouped by page (one per page)
        WITH c, r.source_file as doc_file, r.page_id as page, collect({
            relType: type(r),
            src: r.source_file,
            page: r.page_id,
            chunk: r.chunk_id
        })[0] as page_rel

        // Now collect unique page relationships per connector
        WITH c, doc_file, collect(page_rel) as page_rels

        // Need at least 3 different pages
        WHERE size(page_rels) >= 3

        // Pick first 3 different page relationships
        WITH c,
             page_rels[0] as r1_data,
             page_rels[1] as r2_data,
             page_rels[2] as r3_data

        // IDF calculation
        MATCH (c)--(any_org:ORG)
        WITH c, r1_data, r2_data, r3_data, count(DISTINCT any_org) as org_degree
        WITH c, r1_data, r2_data, r3_data, org_degree, log(toFloat($total_orgs) / org_degree) as idf_score
        WHERE idf_score >= $min_idf

        RETURN
            c.name as connector_name,
            [l IN labels(c) WHERE l <> 'Entity'][0] as connector_type,
            org_degree, idf_score,
            r1_data.relType as rel1, r1_data.src as src1, r1_data.page as page1, r1_data.chunk as chunk1,
            r2_data.relType as rel2, r2_data.src as src2, r2_data.page as page2, r2_data.chunk as chunk2,
            r3_data.relType as rel3, r3_data.src as src3, r3_data.page as page3, r3_data.chunk as chunk3
        ORDER BY idf_score DESC
        LIMIT $limit
        """

        results = []
        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(
                query,
                org=org,
                valid_types=connector_types,
                year_pattern=year_pattern,
                total_orgs=total_orgs,
                min_idf=min_idf,
                limit=limit
            )

            # Track seen connectors + evidence pages to avoid duplicates
            # Keep different evidence pages for same connector (different context)
            seen_connectors = set()

            for rec in records:
                # Deduplicate by (connector, page1, page2, page3)
                key = (rec["connector_name"], rec["page1"], rec["page2"], rec["page3"])
                if key in seen_connectors:
                    continue
                seen_connectors.add(key)

                result = ThreeHopType2Result(
                    pattern_type='intra_doc_type2',
                    connector_name=rec["connector_name"],
                    connector_type=rec["connector_type"],
                    connector_idf=rec["idf_score"],
                    connector_org_degree=rec["org_degree"],
                    hop1_org=org,
                    hop1_relationship=rec["rel1"],
                    hop1_source_file=rec["src1"],
                    hop1_page_id=rec["page1"],
                    hop1_chunk_id=rec["chunk1"],
                    hop2_org=org,
                    hop2_relationship=rec["rel2"],
                    hop2_source_file=rec["src2"],
                    hop2_page_id=rec["page2"],
                    hop2_chunk_id=rec["chunk2"],
                    hop3_org=org,
                    hop3_relationship=rec["rel3"],
                    hop3_source_file=rec["src3"],
                    hop3_page_id=rec["page3"],
                    hop3_chunk_id=rec["chunk3"]
                )
                results.append(result)

        logger.info(f"Found {len(results)} unique 3-hop intra_doc_type2 paths: {org}({year}) p1, p2, p3 → Connector")
        return results

    # =========================================================================
    # 3-HOP CAUSAL PATTERNS (CROSS-COMPANY, SAME YEAR)
    # =========================================================================

    def find_3hop_causal_patterns(
        self,
        org1: str,
        org2: str,
        year: int,
        pattern_family: str,
        motifs: Optional[List[Tuple[str, str, str]]] = None,
        limit: int = 50
    ) -> List[ThreeHopResult]:
        """
        Find causal 3-hop patterns for cross-company (same year).

        pattern_family:
          - "shared_driver" (A): ORG1 -> x1 <- ORG2 and x1 -p2-> x2
          - "shared_outcome" (B): ORG1 -> x2 <- ORG2 and x1 -p2-> x2
          - "cross_anchor" (C): ORG1 -> x1 -p2-> x2 <- ORG2
        """
        if motifs is None:
            motifs = CAUSAL_2HOP_MOTIFS

        results: List[ThreeHopResult] = []
        year_pattern = f".*{year}.*"
        total_orgs = self.config.total_orgs

        causal_edge_match = """
            MATCH (x1)-[p2]-(x2)
            WHERE $x1_label IN labels(x1)
              AND $x2_label IN labels(x2)
              AND type(p2) = $p2_type
              AND p2.source_file =~ $year_pattern
        """
        anchor_filters = """
            WHERE o1.name = $org1
              AND o2.name = $org2
              AND p1.source_file =~ $year_pattern
              AND p3.source_file =~ $year_pattern
              AND p1.source_file <> p3.source_file
              AND (
                (toUpper(split(p1.source_file, '_')[0]) = toUpper($org1) AND toUpper(split(p3.source_file, '_')[0]) = toUpper($org2))
                OR
                (toUpper(split(p1.source_file, '_')[0]) = toUpper($org2) AND toUpper(split(p3.source_file, '_')[0]) = toUpper($org1))
              )
              AND (p2.source_file = p1.source_file OR p2.source_file = p3.source_file)
              AND o1 <> o2
        """

        if pattern_family == "shared_driver":
            query = f"""
            {causal_edge_match}
            WITH DISTINCT x1, x2, p2
            MATCH (o1:ORG)-[p1]-(x1)
            WHERE o1.name = $org1
                AND p1.source_file =~ $year_pattern
            MATCH (o2:ORG)-[p3]-(x1)
            WHERE o2.name = $org2
                AND p3.source_file =~ $year_pattern
            AND (
                (toUpper(split(p1.source_file, '_')[0]) = toUpper($org1) AND toUpper(split(p3.source_file, '_')[0]) = toUpper($org2))
                OR
                (toUpper(split(p1.source_file, '_')[0]) = toUpper($org2) AND toUpper(split(p3.source_file, '_')[0]) = toUpper($org1))
            )
            AND (p2.source_file = p1.source_file OR p2.source_file = p3.source_file)
            WITH DISTINCT o1, o2, x1, x2, p1, p2, p3
            RETURN
              o1.name as org1, o2.name as org2,
              x1.name as x1_name, [l IN labels(x1) WHERE l <> 'Entity'][0] as x1_type,
              x2.name as x2_name, [l IN labels(x2) WHERE l <> 'Entity'][0] as x2_type,
              type(p1) as rel1, p1.source_file as src1, p1.page_id as page1, p1.chunk_id as chunk1, p1.context as ctx1,
              type(p2) as rel2, p2.source_file as src2, p2.page_id as page2, p2.chunk_id as chunk2, p2.context as ctx2,
              type(p3) as rel3, p3.source_file as src3, p3.page_id as page3, p3.chunk_id as chunk3, p3.context as ctx3,
              0 as x1_degree, 0 as x2_degree,
              0.0 as x1_idf,
              0.0 as x2_idf
            LIMIT $limit
            """
        elif pattern_family == "shared_outcome":
            query = f"""
            {causal_edge_match}
            WITH DISTINCT x1, x2, p2
            MATCH (o1:ORG)-[p1]-(x2)
            WHERE o1.name = $org1
                AND p1.source_file =~ $year_pattern
            MATCH (o2:ORG)-[p3]-(x2)
            WHERE o2.name = $org2
                AND p3.source_file =~ $year_pattern
            AND (
                (toUpper(split(p1.source_file, '_')[0]) = toUpper($org1) AND toUpper(split(p3.source_file, '_')[0]) = toUpper($org2))
                OR
                (toUpper(split(p1.source_file, '_')[0]) = toUpper($org2) AND toUpper(split(p3.source_file, '_')[0]) = toUpper($org1))
            )
            AND (p2.source_file = p1.source_file OR p2.source_file = p3.source_file)
            WITH DISTINCT o1, o2, x1, x2, p1, p2, p3
            RETURN
              o1.name as org1, o2.name as org2,
              x1.name as x1_name, [l IN labels(x1) WHERE l <> 'Entity'][0] as x1_type,
              x2.name as x2_name, [l IN labels(x2) WHERE l <> 'Entity'][0] as x2_type,
              type(p1) as rel1, p1.source_file as src1, p1.page_id as page1, p1.chunk_id as chunk1, p1.context as ctx1,
              type(p2) as rel2, p2.source_file as src2, p2.page_id as page2, p2.chunk_id as chunk2, p2.context as ctx2,
              type(p3) as rel3, p3.source_file as src3, p3.page_id as page3, p3.chunk_id as chunk3, p3.context as ctx3,
              0 as x1_degree, 0 as x2_degree,
              0.0 as x1_idf,
              0.0 as x2_idf
            LIMIT $limit
            """
        elif pattern_family == "cross_anchor":
            query = f"""
            {causal_edge_match}
            WITH DISTINCT x1, x2, p2
            MATCH (o1:ORG)-[p1]-(x1)
            WHERE o1.name = $org1
                AND p1.source_file =~ $year_pattern
            MATCH (o2:ORG)-[p3]-(x2)
            WHERE o2.name = $org2
                AND p3.source_file =~ $year_pattern
            AND p1.source_file <> p3.source_file
            AND (
                (toUpper(split(p1.source_file, '_')[0]) = toUpper($org1) AND toUpper(split(p3.source_file, '_')[0]) = toUpper($org2))
                OR
                (toUpper(split(p1.source_file, '_')[0]) = toUpper($org2) AND toUpper(split(p3.source_file, '_')[0]) = toUpper($org1))
            )
            AND (p2.source_file = p1.source_file OR p2.source_file = p3.source_file)
            RETURN
              o1.name as org1, o2.name as org2,
              x1.name as x1_name, [l IN labels(x1) WHERE l <> 'Entity'][0] as x1_type,
              x2.name as x2_name, [l IN labels(x2) WHERE l <> 'Entity'][0] as x2_type,
              type(p1) as rel1, p1.source_file as src1, p1.page_id as page1, p1.chunk_id as chunk1, p1.context as ctx1,
              type(p2) as rel2, p2.source_file as src2, p2.page_id as page2, p2.chunk_id as chunk2, p2.context as ctx2,
              type(p3) as rel3, p3.source_file as src3, p3.page_id as page3, p3.chunk_id as chunk3, p3.context as ctx3,
              0 as x1_degree, 0 as x2_degree,
              0.0 as x1_idf,
              0.0 as x2_idf
            LIMIT $limit
            """
        else:
            raise ValueError(f"Unknown pattern_family: {pattern_family}")

        with self.driver.session(database=self.config.neo4j_database) as session:
            for x1_label, p2_type, x2_label in motifs:
                records = session.run(
                    query,
                    org1=org1,
                    org2=org2,
                    x1_label=x1_label,
                    x2_label=x2_label,
                    p2_type=p2_type,
                    year_pattern=year_pattern,
                    total_orgs=total_orgs,
                    limit=max(1, limit - len(results))
                )
                for rec in records:
                    x1_idf = rec["x1_idf"]
                    x2_idf = rec["x2_idf"]
                    combined_idf = (x1_idf + x2_idf) / 2

                    if pattern_family == "shared_outcome":
                        connector_name = rec["x2_name"]
                        connector_type = rec["x2_type"]
                        connector_idf = x2_idf
                        connector_degree = rec["x2_degree"]
                        hop3_entity_name = rec["x2_name"]
                        hop3_entity_type = rec["x2_type"]
                        hop2_entity_name = rec["x1_name"]
                        hop2_entity_type = rec["x1_type"]
                        hop3_entity_idf = x2_idf
                    else:
                        connector_name = rec["x1_name"]
                        connector_type = rec["x1_type"]
                        connector_idf = x1_idf
                        connector_degree = rec["x1_degree"]
                        hop3_entity_name = rec["x1_name"] if pattern_family == "shared_driver" else rec["x2_name"]
                        hop3_entity_type = rec["x1_type"] if pattern_family == "shared_driver" else rec["x2_type"]
                        hop2_entity_name = rec["x2_name"]
                        hop2_entity_type = rec["x2_type"]
                        hop3_entity_idf = x2_idf

                    result = ThreeHopResult(
                        pattern_type=f"causal_{pattern_family}",
                        connector1_name=connector_name,
                        connector1_type=connector_type,
                        connector1_idf=connector_idf,
                        connector1_org_degree=connector_degree,
                        hop1_org=rec["org1"],
                        hop1_relationship=rec["rel1"],
                        hop1_source_file=rec["src1"],
                        hop1_page_id=rec["page1"],
                        hop1_chunk_id=rec["chunk1"],
                        hop2_entity_name=hop2_entity_name,
                        hop2_entity_type=hop2_entity_type,
                        hop2_relationship=rec["rel2"],
                        hop2_source_file=rec["src2"],
                        hop2_page_id=rec["page2"],
                        hop2_chunk_id=rec["chunk2"],
                        hop3_entity_name=hop3_entity_name,
                        hop3_entity_type=hop3_entity_type,
                        hop3_org=rec["org2"],
                        hop3_relationship=rec["rel3"],
                        hop3_source_file=rec["src3"],
                        hop3_page_id=rec["page3"],
                        hop3_chunk_id=rec["chunk3"],
                        combined_idf_score=combined_idf,
                        hop3_entity_idf=hop3_entity_idf
                    )
                    results.append(result)
                    if len(results) >= limit:
                        return results

        return results

    # =========================================================================
    # CONNECTOR-FIRST APPROACH FOR TYPE 1 (Efficient 3-hop cross_company_type1)
    # Pattern: ORG1 → C1 → C2 ← ORG2 (Connector Chain)
    # =========================================================================

    def find_connector_chains_with_orgs(
        self,
        year: int,
        connector1_types: Optional[List[str]] = None,
        connector2_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        org_filter: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Find connector chains (C1→C2) where both connectors have ORG connections.

        This is much more efficient than brute-forcing all pair combinations because:
        - Single query to find all qualifying connector chains
        - Each chain comes with its list of connected ORGs for C1 and C2
        - Pairs generated from these lists are GUARANTEED to have the chain

        Args:
            year: Year to filter relationships (e.g., 2023)
            connector1_types: List of connector types for C1
            connector2_types: List of connector types for C2
            min_idf: Minimum IDF threshold for connectors
            org_filter: Optional list of ORGs to consider (e.g., sector filter)
            limit: Maximum number of chains to return

        Returns:
            List of dicts with:
            - c1_name, c1_type, c1_idf, c1_degree
            - c2_name, c2_type, c2_idf, c2_degree
            - c1_orgs: List[str] - ORGs connected to C1
            - c2_orgs: List[str] - ORGs connected to C2
            - c1_org_evidence: Dict[str, Dict] - Evidence per ORG for C1
            - c2_org_evidence: Dict[str, Dict] - Evidence per ORG for C2
            - chain_evidence: Dict - Evidence for C1→C2 link
        """
        if connector1_types is None:
            connector1_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL']
        if connector2_types is None:
            connector2_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL', 'FIN_INST']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs
        year_pattern = f".*{year}.*"

        # Build ORG filter clauses if provided
        org_filter_clause_o1 = "AND o1.name IN $org_filter" if org_filter else ""
        org_filter_clause_o2 = "AND o2.name IN $org_filter" if org_filter else ""

        query = f"""
        // Find connector chains (C1→C2) with their ORG connections
        // Step 1: Find C1→C2 links
        MATCH (c1)-[r_chain]-(c2)
        WHERE any(label IN labels(c1) WHERE label IN $c1_types)
          AND any(label IN labels(c2) WHERE label IN $c2_types)
          AND c1 <> c2
          AND r_chain.source_file =~ $year_pattern

        // Step 2: Find ORGs connected to C1 (from their own docs)
        WITH c1, c2, r_chain
        MATCH (o1:ORG)-[r1]-(c1)
        WHERE r1.source_file =~ $year_pattern
          AND r1.source_file CONTAINS o1.name
          {org_filter_clause_o1}

        WITH c1, c2, r_chain,
             collect(DISTINCT o1.name) as c1_orgs,
             collect(DISTINCT {{
                 org: o1.name,
                 rel: type(r1),
                 src: r1.source_file,
                 page: r1.page_id,
                 chunk: r1.chunk_id
             }}) as c1_evidence

        // Step 3: Find ORGs connected to C2 (from their own docs)
        MATCH (o2:ORG)-[r3]-(c2)
        WHERE r3.source_file =~ $year_pattern
          AND r3.source_file CONTAINS o2.name
          {org_filter_clause_o2}

        WITH c1, c2, r_chain, c1_orgs, c1_evidence,
             collect(DISTINCT o2.name) as c2_orgs,
             collect(DISTINCT {{
                 org: o2.name,
                 rel: type(r3),
                 src: r3.source_file,
                 page: r3.page_id,
                 chunk: r3.chunk_id
             }}) as c2_evidence

        // Filter: need at least 1 ORG on each side
        WHERE size(c1_orgs) >= 1 AND size(c2_orgs) >= 1

        // Calculate IDF for C1
        MATCH (c1)--(any_org1:ORG)
        WITH c1, c2, r_chain, c1_orgs, c1_evidence, c2_orgs, c2_evidence,
             count(DISTINCT any_org1) as c1_degree
        WITH c1, c2, r_chain, c1_orgs, c1_evidence, c2_orgs, c2_evidence,
             c1_degree, log(toFloat($total_orgs) / c1_degree) as c1_idf

        // Calculate IDF for C2
        MATCH (c2)--(any_org2:ORG)
        WITH c1, c2, r_chain, c1_orgs, c1_evidence, c2_orgs, c2_evidence,
             c1_degree, c1_idf, count(DISTINCT any_org2) as c2_degree
        WITH c1, c2, r_chain, c1_orgs, c1_evidence, c2_orgs, c2_evidence,
             c1_degree, c1_idf, c2_degree,
             log(toFloat($total_orgs) / c2_degree) as c2_idf

        WHERE c1_idf > $min_idf AND c2_idf > $min_idf

        RETURN
            c1.name as c1_name, [l IN labels(c1) WHERE l <> 'Entity'][0] as c1_type,
            c1_idf, c1_degree, c1_orgs, c1_evidence,
            c2.name as c2_name, [l IN labels(c2) WHERE l <> 'Entity'][0] as c2_type,
            c2_idf, c2_degree, c2_orgs, c2_evidence,
            type(r_chain) as chain_rel, r_chain.source_file as chain_src,
            r_chain.page_id as chain_page, r_chain.chunk_id as chain_chunk,
            (c1_idf + c2_idf) / 2 as combined_idf
        ORDER BY combined_idf DESC
        LIMIT $limit
        """

        results = []
        params = {
            'c1_types': connector1_types,
            'c2_types': connector2_types,
            'year_pattern': year_pattern,
            'total_orgs': total_orgs,
            'min_idf': min_idf,
            'limit': limit
        }
        if org_filter:
            params['org_filter'] = org_filter

        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(query, **params)

            for rec in records:
                # Build org evidence dicts for quick lookup
                c1_org_evidence = {}
                for ev in rec["c1_evidence"]:
                    org = ev["org"]
                    if org not in c1_org_evidence:
                        c1_org_evidence[org] = {
                            'relationship': ev["rel"],
                            'source_file': ev["src"],
                            'page_id': ev["page"],
                            'chunk_id': ev["chunk"]
                        }

                c2_org_evidence = {}
                for ev in rec["c2_evidence"]:
                    org = ev["org"]
                    if org not in c2_org_evidence:
                        c2_org_evidence[org] = {
                            'relationship': ev["rel"],
                            'source_file': ev["src"],
                            'page_id': ev["page"],
                            'chunk_id': ev["chunk"]
                        }

                results.append({
                    'c1_name': rec["c1_name"],
                    'c1_type': rec["c1_type"],
                    'c1_idf': rec["c1_idf"],
                    'c1_degree': rec["c1_degree"],
                    'c1_orgs': rec["c1_orgs"],
                    'c1_org_evidence': c1_org_evidence,
                    'c2_name': rec["c2_name"],
                    'c2_type': rec["c2_type"],
                    'c2_idf': rec["c2_idf"],
                    'c2_degree': rec["c2_degree"],
                    'c2_orgs': rec["c2_orgs"],
                    'c2_org_evidence': c2_org_evidence,
                    'chain_evidence': {
                        'relationship': rec["chain_rel"],
                        'source_file': rec["chain_src"],
                        'page_id': rec["chain_page"],
                        'chunk_id': rec["chain_chunk"]
                    },
                    'combined_idf': rec["combined_idf"]
                })

        logger.info(f"Found {len(results)} connector chains with ORG connections for year {year}")
        return results

    def build_type1_result_from_chain(
        self,
        chain: Dict[str, Any],
        org1: str,
        org2: str
    ) -> Optional[ThreeHopResult]:
        """
        Build a ThreeHopResult from a pre-fetched connector chain and pair.

        This is used with the connector-first approach - the chain already has
        evidence for all connected ORGs, so we just need to assemble the result.

        Args:
            chain: Dict from find_connector_chains_with_orgs()
            org1, org2: The pair of ORGs

        Returns:
            ThreeHopResult if both ORGs have evidence, None otherwise
        """
        c1_evidence = chain.get('c1_org_evidence', {})
        c2_evidence = chain.get('c2_org_evidence', {})

        # Check org1 connects to C1 and org2 connects to C2
        if org1 not in c1_evidence or org2 not in c2_evidence:
            return None

        ev1 = c1_evidence[org1]
        ev2 = chain['chain_evidence']
        ev3 = c2_evidence[org2]

        return ThreeHopResult(
            pattern_type='cross_company_type1',
            connector1_name=chain['c1_name'],
            connector1_type=chain['c1_type'],
            connector1_idf=chain['c1_idf'],
            connector1_org_degree=chain['c1_degree'],
            hop1_org=org1,
            hop1_relationship=ev1['relationship'],
            hop1_source_file=ev1['source_file'],
            hop1_page_id=ev1['page_id'],
            hop1_chunk_id=ev1['chunk_id'],
            hop2_entity_name=chain['c2_name'],
            hop2_entity_type=chain['c2_type'],
            hop2_relationship=ev2['relationship'],
            hop2_source_file=ev2['source_file'],
            hop2_page_id=ev2['page_id'],
            hop2_chunk_id=ev2['chunk_id'],
            hop3_entity_name=chain['c2_name'],
            hop3_entity_type=chain['c2_type'],
            hop3_org=org2,
            hop3_relationship=ev3['relationship'],
            hop3_source_file=ev3['source_file'],
            hop3_page_id=ev3['page_id'],
            hop3_chunk_id=ev3['chunk_id'],
            combined_idf_score=chain['combined_idf'],
            hop3_entity_idf=chain['c2_idf']
        )

    def generate_type1_pairs_from_chains(
        self,
        chains: List[Dict[str, Any]],
        org_filter: Optional[List[str]] = None,
        max_pairs_per_chain: int = 50,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Generate valid pairs from connector chains found by find_connector_chains_with_orgs().

        Each returned pair is GUARANTEED to have the connector chain (100% hit rate).

        Args:
            chains: List from find_connector_chains_with_orgs()
            org_filter: Optional list of ORGs to include (e.g., sector filter)
            max_pairs_per_chain: Limit pairs generated per chain
            shuffle: Whether to shuffle results for diversity
            seed: Random seed for reproducibility

        Returns:
            List of (org1, org2, chain_dict) tuples
        """
        import itertools
        import random

        if shuffle:
            random.seed(seed)

        all_pairs = []

        for chain in chains:
            c1_orgs = chain['c1_orgs']
            c2_orgs = chain['c2_orgs']

            # Apply org filter if provided
            if org_filter:
                c1_orgs = [o for o in c1_orgs if o in org_filter]
                c2_orgs = [o for o in c2_orgs if o in org_filter]

            if len(c1_orgs) < 1 or len(c2_orgs) < 1:
                continue

            # Generate pairs: ORGs(C1) × ORGs(C2) where org1 != org2
            pair_combos = [
                (o1, o2) for o1 in c1_orgs for o2 in c2_orgs
                if o1 != o2
            ]

            if shuffle:
                random.shuffle(pair_combos)

            # Limit pairs per chain to avoid over-representation
            pair_combos = pair_combos[:max_pairs_per_chain]

            for org1, org2 in pair_combos:
                all_pairs.append((org1, org2, chain))

        if shuffle:
            random.shuffle(all_pairs)

        logger.info(f"Generated {len(all_pairs)} valid pairs from {len(chains)} connector chains")
        return all_pairs

    def find_connector_chains_cross_year(
        self,
        year1: int,
        year2: int,
        connector1_types: Optional[List[str]] = None,
        connector2_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        org_filter: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Find connector chains for cross_year_type1: Same ORG connects to C1 in year1 and C2 in year2.

        Args:
            year1, year2: The two years to compare
            connector1_types, connector2_types: Connector types for C1 and C2
            min_idf: Minimum IDF threshold
            org_filter: Optional list of ORGs to consider
            limit: Maximum number of chains to return

        Returns:
            List of dicts with chain info and ORGs that appear on both sides
        """
        if connector1_types is None:
            connector1_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL']
        if connector2_types is None:
            connector2_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL', 'FIN_INST']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs
        year1_pattern = f".*{year1}.*"
        year2_pattern = f".*{year2}.*"

        org_filter_clause = ""
        if org_filter:
            org_filter_clause = "AND o.name IN $org_filter"

        query = f"""
        // Find chains where same ORG connects to C1 in year1 and C2 in year2
        MATCH (c1)-[r_chain]-(c2)
        WHERE any(label IN labels(c1) WHERE label IN $c1_types)
          AND any(label IN labels(c2) WHERE label IN $c2_types)
          AND c1 <> c2

        // ORGs connected to C1 in year1 (from their own docs)
        WITH c1, c2, r_chain
        MATCH (o:ORG)-[r1]-(c1)
        WHERE r1.source_file =~ $year1_pattern
          AND r1.source_file CONTAINS o.name
          {org_filter_clause}

        WITH c1, c2, r_chain,
             collect(DISTINCT o.name) as c1_orgs_y1,
             collect(DISTINCT {{
                 org: o.name,
                 rel: type(r1),
                 src: r1.source_file,
                 page: r1.page_id,
                 chunk: r1.chunk_id
             }}) as c1_evidence

        // ORGs connected to C2 in year2 (from their own docs)
        MATCH (o2:ORG)-[r3]-(c2)
        WHERE r3.source_file =~ $year2_pattern
          AND r3.source_file CONTAINS o2.name
          {'AND o2.name IN $org_filter' if org_filter else ''}

        WITH c1, c2, r_chain, c1_orgs_y1, c1_evidence,
             collect(DISTINCT o2.name) as c2_orgs_y2,
             collect(DISTINCT {{
                 org: o2.name,
                 rel: type(r3),
                 src: r3.source_file,
                 page: r3.page_id,
                 chunk: r3.chunk_id
             }}) as c2_evidence

        // Find ORGs that appear on BOTH sides (same ORG, different years)
        WITH c1, c2, r_chain, c1_orgs_y1, c1_evidence, c2_orgs_y2, c2_evidence,
             [x IN c1_orgs_y1 WHERE x IN c2_orgs_y2] as common_orgs

        WHERE size(common_orgs) >= 1

        // Get chain evidence (must be from one of the years)
        WITH c1, c2, r_chain, c1_orgs_y1, c1_evidence, c2_orgs_y2, c2_evidence, common_orgs
        WHERE r_chain.source_file =~ $year1_pattern OR r_chain.source_file =~ $year2_pattern

        // Calculate IDF
        MATCH (c1)--(any_org1:ORG)
        WITH c1, c2, r_chain, c1_orgs_y1, c1_evidence, c2_orgs_y2, c2_evidence, common_orgs,
             count(DISTINCT any_org1) as c1_degree
        MATCH (c2)--(any_org2:ORG)
        WITH c1, c2, r_chain, c1_orgs_y1, c1_evidence, c2_orgs_y2, c2_evidence, common_orgs,
             c1_degree, count(DISTINCT any_org2) as c2_degree

        WITH c1, c2, r_chain, c1_orgs_y1, c1_evidence, c2_orgs_y2, c2_evidence, common_orgs,
             c1_degree, c2_degree,
             log(toFloat($total_orgs) / c1_degree) as c1_idf,
             log(toFloat($total_orgs) / c2_degree) as c2_idf

        WHERE c1_idf > $min_idf AND c2_idf > $min_idf

        RETURN
            c1.name as c1_name, [l IN labels(c1) WHERE l <> 'Entity'][0] as c1_type,
            c1_idf, c1_degree, c1_orgs_y1, c1_evidence,
            c2.name as c2_name, [l IN labels(c2) WHERE l <> 'Entity'][0] as c2_type,
            c2_idf, c2_degree, c2_orgs_y2, c2_evidence,
            common_orgs,
            type(r_chain) as chain_rel, r_chain.source_file as chain_src,
            r_chain.page_id as chain_page, r_chain.chunk_id as chain_chunk,
            (c1_idf + c2_idf) / 2 as combined_idf
        ORDER BY combined_idf DESC, size(common_orgs) DESC
        LIMIT $limit
        """

        results = []
        params = {
            'c1_types': connector1_types,
            'c2_types': connector2_types,
            'year1_pattern': year1_pattern,
            'year2_pattern': year2_pattern,
            'total_orgs': total_orgs,
            'min_idf': min_idf,
            'limit': limit
        }
        if org_filter:
            params['org_filter'] = org_filter

        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(query, **params)

            for rec in records:
                # Build org evidence dicts
                c1_org_evidence = {}
                for ev in rec["c1_evidence"]:
                    org = ev["org"]
                    if org not in c1_org_evidence:
                        c1_org_evidence[org] = {
                            'relationship': ev["rel"],
                            'source_file': ev["src"],
                            'page_id': ev["page"],
                            'chunk_id': ev["chunk"]
                        }

                c2_org_evidence = {}
                for ev in rec["c2_evidence"]:
                    org = ev["org"]
                    if org not in c2_org_evidence:
                        c2_org_evidence[org] = {
                            'relationship': ev["rel"],
                            'source_file': ev["src"],
                            'page_id': ev["page"],
                            'chunk_id': ev["chunk"]
                        }

                results.append({
                    'c1_name': rec["c1_name"],
                    'c1_type': rec["c1_type"],
                    'c1_idf': rec["c1_idf"],
                    'c1_degree': rec["c1_degree"],
                    'c1_orgs': rec["c1_orgs_y1"],
                    'c1_org_evidence': c1_org_evidence,
                    'c2_name': rec["c2_name"],
                    'c2_type': rec["c2_type"],
                    'c2_idf': rec["c2_idf"],
                    'c2_degree': rec["c2_degree"],
                    'c2_orgs': rec["c2_orgs_y2"],
                    'c2_org_evidence': c2_org_evidence,
                    'common_orgs': rec["common_orgs"],  # ORGs that appear on both sides
                    'chain_evidence': {
                        'relationship': rec["chain_rel"],
                        'source_file': rec["chain_src"],
                        'page_id': rec["chain_page"],
                        'chunk_id': rec["chain_chunk"]
                    },
                    'combined_idf': rec["combined_idf"]
                })

        logger.info(f"Found {len(results)} connector chains for cross_year ({year1} vs {year2})")
        return results

    def generate_type1_cross_year_from_chains(
        self,
        chains: List[Dict[str, Any]],
        org_filter: Optional[List[str]] = None,
        max_orgs_per_chain: int = 50,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate valid ORGs from cross_year chains.

        Each returned ORG is GUARANTEED to have the connector chain (100% hit rate).

        Returns:
            List of (org, chain_dict) tuples
        """
        import random

        if shuffle:
            random.seed(seed)

        all_orgs = []

        for chain in chains:
            common_orgs = chain['common_orgs']

            # Apply org filter if provided
            if org_filter:
                common_orgs = [o for o in common_orgs if o in org_filter]

            if len(common_orgs) < 1:
                continue

            if shuffle:
                random.shuffle(common_orgs)

            # Limit ORGs per chain
            common_orgs = common_orgs[:max_orgs_per_chain]

            for org in common_orgs:
                all_orgs.append((org, chain))

        if shuffle:
            random.shuffle(all_orgs)

        logger.info(f"Generated {len(all_orgs)} valid ORGs from {len(chains)} cross_year chains")
        return all_orgs

    def build_type1_cross_year_result(
        self,
        chain: Dict[str, Any],
        org: str,
        year1: int,
        year2: int
    ) -> Optional[ThreeHopResult]:
        """
        Build a ThreeHopResult for cross_year_type1 from pre-fetched chain data.

        Returns:
            ThreeHopResult if ORG has evidence on both sides, None otherwise
        """
        c1_evidence = chain.get('c1_org_evidence', {})
        c2_evidence = chain.get('c2_org_evidence', {})

        if org not in c1_evidence or org not in c2_evidence:
            return None

        ev1 = c1_evidence[org]
        ev2 = chain['chain_evidence']
        ev3 = c2_evidence[org]

        return ThreeHopResult(
            pattern_type='cross_year_type1',
            connector1_name=chain['c1_name'],
            connector1_type=chain['c1_type'],
            connector1_idf=chain['c1_idf'],
            connector1_org_degree=chain['c1_degree'],
            hop1_org=org,
            hop1_relationship=ev1['relationship'],
            hop1_source_file=ev1['source_file'],
            hop1_page_id=ev1['page_id'],
            hop1_chunk_id=ev1['chunk_id'],
            hop2_entity_name=chain['c2_name'],
            hop2_entity_type=chain['c2_type'],
            hop2_relationship=ev2['relationship'],
            hop2_source_file=ev2['source_file'],
            hop2_page_id=ev2['page_id'],
            hop2_chunk_id=ev2['chunk_id'],
            hop3_entity_name=chain['c2_name'],
            hop3_entity_type=chain['c2_type'],
            hop3_org=org,
            hop3_relationship=ev3['relationship'],
            hop3_source_file=ev3['source_file'],
            hop3_page_id=ev3['page_id'],
            hop3_chunk_id=ev3['chunk_id'],
            combined_idf_score=chain['combined_idf'],
            hop3_entity_idf=chain['c2_idf']
        )

    def find_connector_chains_intra_doc(
        self,
        year: int,
        connector1_types: Optional[List[str]] = None,
        connector2_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        org_filter: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Find connector chains for intra_doc_type1: Same ORG, same document, different pages.

        Pattern: ORG(page1) → C1 → C2 ← ORG(page2)
        All relationships must be from the same document.

        Args:
            year: Year to filter
            connector1_types, connector2_types: Connector types
            min_idf: Minimum IDF threshold
            org_filter: Optional list of ORGs to consider
            limit: Maximum chains to return

        Returns:
            List of dicts with chain info and ORGs that have intra-doc connections
        """
        if connector1_types is None:
            connector1_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL']
        if connector2_types is None:
            connector2_types = ['COMP', 'RISK_FACTOR', 'MACRO_CONDITION', 'RAW_MATERIAL', 'FIN_INST']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs
        year_pattern = f".*{year}.*"

        org_filter_clause = ""
        if org_filter:
            org_filter_clause = "AND o.name IN $org_filter"

        query = f"""
        // Find chains where same ORG connects to C1 and C2 from same document, different pages
        MATCH (c1)-[r_chain]-(c2)
        WHERE any(label IN labels(c1) WHERE label IN $c1_types)
          AND any(label IN labels(c2) WHERE label IN $c2_types)
          AND c1 <> c2
          AND r_chain.source_file =~ $year_pattern

        // ORGs connected to C1 (from their own docs)
        WITH c1, c2, r_chain
        MATCH (o:ORG)-[r1]-(c1)
        WHERE r1.source_file =~ $year_pattern
          AND r1.source_file CONTAINS o.name
          {org_filter_clause}

        WITH c1, c2, r_chain, o,
             collect({{
                 rel: type(r1),
                 src: r1.source_file,
                 page: r1.page_id,
                 chunk: r1.chunk_id
             }}) as c1_pages

        // Same ORG connected to C2 from same document, different page
        MATCH (o)-[r3]-(c2)
        WHERE r3.source_file =~ $year_pattern
          AND r3.source_file CONTAINS o.name
          AND r3.source_file = c1_pages[0].src  // Same document
          AND r3.page_id <> c1_pages[0].page    // Different page
          AND r_chain.source_file = c1_pages[0].src  // Chain evidence also from same doc

        WITH c1, c2, r_chain, o, c1_pages,
             collect({{
                 rel: type(r3),
                 src: r3.source_file,
                 page: r3.page_id,
                 chunk: r3.chunk_id
             }}) as c2_pages

        // Group by chain
        WITH c1, c2, r_chain,
             collect(DISTINCT {{
                 org: o.name,
                 c1_ev: c1_pages[0],
                 c2_ev: c2_pages[0]
             }}) as org_evidence

        WHERE size(org_evidence) >= 1

        // Calculate IDF
        MATCH (c1)--(any_org1:ORG)
        WITH c1, c2, r_chain, org_evidence, count(DISTINCT any_org1) as c1_degree
        MATCH (c2)--(any_org2:ORG)
        WITH c1, c2, r_chain, org_evidence, c1_degree, count(DISTINCT any_org2) as c2_degree

        WITH c1, c2, r_chain, org_evidence, c1_degree, c2_degree,
             log(toFloat($total_orgs) / c1_degree) as c1_idf,
             log(toFloat($total_orgs) / c2_degree) as c2_idf

        WHERE c1_idf > $min_idf AND c2_idf > $min_idf

        RETURN
            c1.name as c1_name, [l IN labels(c1) WHERE l <> 'Entity'][0] as c1_type,
            c1_idf, c1_degree,
            c2.name as c2_name, [l IN labels(c2) WHERE l <> 'Entity'][0] as c2_type,
            c2_idf, c2_degree,
            org_evidence,
            type(r_chain) as chain_rel, r_chain.source_file as chain_src,
            r_chain.page_id as chain_page, r_chain.chunk_id as chain_chunk,
            (c1_idf + c2_idf) / 2 as combined_idf
        ORDER BY combined_idf DESC
        LIMIT $limit
        """

        results = []
        params = {
            'c1_types': connector1_types,
            'c2_types': connector2_types,
            'year_pattern': year_pattern,
            'total_orgs': total_orgs,
            'min_idf': min_idf,
            'limit': limit
        }
        if org_filter:
            params['org_filter'] = org_filter

        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(query, **params)

            for rec in records:
                # Extract ORG evidence
                orgs = []
                org_evidence_map = {}
                for ev in rec["org_evidence"]:
                    org = ev["org"]
                    orgs.append(org)
                    org_evidence_map[org] = {
                        'c1_evidence': ev["c1_ev"],
                        'c2_evidence': ev["c2_ev"]
                    }

                results.append({
                    'c1_name': rec["c1_name"],
                    'c1_type': rec["c1_type"],
                    'c1_idf': rec["c1_idf"],
                    'c1_degree': rec["c1_degree"],
                    'c2_name': rec["c2_name"],
                    'c2_type': rec["c2_type"],
                    'c2_idf': rec["c2_idf"],
                    'c2_degree': rec["c2_degree"],
                    'orgs': list(set(orgs)),
                    'org_evidence': org_evidence_map,
                    'chain_evidence': {
                        'relationship': rec["chain_rel"],
                        'source_file': rec["chain_src"],
                        'page_id': rec["chain_page"],
                        'chunk_id': rec["chain_chunk"]
                    },
                    'combined_idf': rec["combined_idf"]
                })

        logger.info(f"Found {len(results)} connector chains for intra_doc (year {year})")
        return results

    def generate_type1_intra_doc_from_chains(
        self,
        chains: List[Dict[str, Any]],
        org_filter: Optional[List[str]] = None,
        max_orgs_per_chain: int = 50,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate valid ORGs from intra_doc chains.

        Each returned ORG is GUARANTEED to have the connector chain (100% hit rate).

        Returns:
            List of (org, chain_dict) tuples
        """
        import random

        if shuffle:
            random.seed(seed)

        all_orgs = []

        for chain in chains:
            orgs = chain['orgs']

            # Apply org filter if provided
            if org_filter:
                orgs = [o for o in orgs if o in org_filter]

            if len(orgs) < 1:
                continue

            if shuffle:
                random.shuffle(orgs)

            # Limit ORGs per chain
            orgs = orgs[:max_orgs_per_chain]

            for org in orgs:
                all_orgs.append((org, chain))

        if shuffle:
            random.shuffle(all_orgs)

        logger.info(f"Generated {len(all_orgs)} valid ORGs from {len(chains)} intra_doc chains")
        return all_orgs

    def build_type1_intra_doc_result(
        self,
        chain: Dict[str, Any],
        org: str
    ) -> Optional[ThreeHopResult]:
        """
        Build a ThreeHopResult for intra_doc_type1 from pre-fetched chain data.

        Returns:
            ThreeHopResult if ORG has evidence, None otherwise
        """
        org_evidence = chain.get('org_evidence', {})

        if org not in org_evidence:
            return None

        ev = org_evidence[org]
        ev1 = ev['c1_evidence']
        ev2 = chain['chain_evidence']
        ev3 = ev['c2_evidence']

        return ThreeHopResult(
            pattern_type='intra_doc_type1',
            connector1_name=chain['c1_name'],
            connector1_type=chain['c1_type'],
            connector1_idf=chain['c1_idf'],
            connector1_org_degree=chain['c1_degree'],
            hop1_org=org,
            hop1_relationship=ev1['rel'],
            hop1_source_file=ev1['src'],
            hop1_page_id=ev1['page'],
            hop1_chunk_id=ev1['chunk'],
            hop2_entity_name=chain['c2_name'],
            hop2_entity_type=chain['c2_type'],
            hop2_relationship=ev2['relationship'],
            hop2_source_file=ev2['source_file'],
            hop2_page_id=ev2['page_id'],
            hop2_chunk_id=ev2['chunk_id'],
            hop3_entity_name=chain['c2_name'],
            hop3_entity_type=chain['c2_type'],
            hop3_org=org,
            hop3_relationship=ev3['rel'],
            hop3_source_file=ev3['src'],
            hop3_page_id=ev3['page'],
            hop3_chunk_id=ev3['chunk'],
            combined_idf_score=chain['combined_idf'],
            hop3_entity_idf=chain['c2_idf']
        )

    # =========================================================================
    # CONNECTOR-FIRST APPROACH FOR TYPE 2 (Efficient 3-hop cross_company_type2)
    # =========================================================================

    def find_connectors_with_multiple_orgs(
        self,
        year: int,
        connector_types: Optional[List[str]] = None,
        min_idf: Optional[float] = None,
        min_org_count: int = 3,
        org_filter: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Find connectors that have connections to multiple ORGs (connector-first approach).

        This is much more efficient than brute-forcing all triplet combinations because:
        - Single query to find all qualifying connectors
        - Each connector comes with its list of connected ORGs
        - Triplets generated from these lists are GUARANTEED to share the connector

        Args:
            year: Year to filter relationships (e.g., 2023)
            connector_types: List of connector types (e.g., ['FIN_METRIC', 'SEGMENT'])
            min_idf: Minimum IDF threshold for connectors
            min_org_count: Minimum number of ORGs a connector must have (default: 3 for triplets)
            org_filter: Optional list of ORGs to consider (e.g., S&P 100 tickers)
            limit: Maximum number of connectors to return

        Returns:
            List of dicts with:
            - connector_name: str
            - connector_type: str
            - idf_score: float
            - org_degree: int
            - connected_orgs: List[str] - ORGs that mention this connector
            - org_evidence: Dict[str, Dict] - Evidence per ORG (source_file, page_id, chunk_id, relationship)
        """
        if connector_types is None:
            connector_types = ['FIN_METRIC', 'SEGMENT', 'ECON_IND']
        if min_idf is None:
            min_idf = self.config.default_min_idf

        total_orgs = self.config.total_orgs
        year_pattern = f".*{year}.*"

        # Build ORG filter clause if provided
        org_filter_clause = ""
        if org_filter:
            org_filter_clause = "AND o.name IN $org_filter"

        query = f"""
        // Find connectors with 3+ ORG connections (connector-first approach)
        MATCH (c)-[r]-(o:ORG)
        WHERE any(label IN labels(c) WHERE label IN $valid_types)
          AND r.source_file =~ $year_pattern
          AND r.source_file CONTAINS o.name
          {org_filter_clause}

        // Aggregate by connector
        WITH c,
             collect(DISTINCT o.name) as orgs,
             collect(DISTINCT {{
                 org: o.name,
                 rel: type(r),
                 src: r.source_file,
                 page: r.page_id,
                 chunk: r.chunk_id
             }}) as evidence_list

        // Filter to connectors with min_org_count+ ORGs
        WHERE size(orgs) >= $min_org_count

        // Calculate IDF
        MATCH (c)--(any_org:ORG)
        WITH c, orgs, evidence_list, count(DISTINCT any_org) as org_degree
        WITH c, orgs, evidence_list, org_degree,
             log(toFloat($total_orgs) / org_degree) as idf_score
        WHERE idf_score >= $min_idf

        RETURN
            c.name as connector_name,
            [l IN labels(c) WHERE l <> 'Entity'][0] as connector_type,
            idf_score,
            org_degree,
            orgs as connected_orgs,
            evidence_list
        ORDER BY idf_score DESC, size(orgs) DESC
        LIMIT $limit
        """

        results = []
        params = {
            'valid_types': connector_types,
            'year_pattern': year_pattern,
            'min_org_count': min_org_count,
            'total_orgs': total_orgs,
            'min_idf': min_idf,
            'limit': limit
        }
        if org_filter:
            params['org_filter'] = org_filter

        with self.driver.session(database=self.config.neo4j_database) as session:
            records = session.run(query, **params)

            for rec in records:
                # Build org_evidence dict for quick lookup
                org_evidence = {}
                for ev in rec["evidence_list"]:
                    org = ev["org"]
                    if org not in org_evidence:
                        org_evidence[org] = {
                            'relationship': ev["rel"],
                            'source_file': ev["src"],
                            'page_id': ev["page"],
                            'chunk_id': ev["chunk"]
                        }

                results.append({
                    'connector_name': rec["connector_name"],
                    'connector_type': rec["connector_type"],
                    'idf_score': rec["idf_score"],
                    'org_degree': rec["org_degree"],
                    'connected_orgs': rec["connected_orgs"],
                    'org_evidence': org_evidence
                })

        logger.info(f"Found {len(results)} connectors with {min_org_count}+ ORG connections for year {year}")
        return results

    def build_type2_result_from_connector(
        self,
        connector: Dict[str, Any],
        org1: str,
        org2: str,
        org3: str
    ) -> Optional[ThreeHopType2Result]:
        """
        Build a ThreeHopType2Result from a pre-fetched connector and triplet.

        This is used with the connector-first approach - the connector already has
        evidence for all connected ORGs, so we just need to assemble the result.

        Args:
            connector: Dict from find_connectors_with_multiple_orgs()
            org1, org2, org3: The triplet of ORGs

        Returns:
            ThreeHopType2Result if all 3 ORGs have evidence, None otherwise
        """
        org_evidence = connector.get('org_evidence', {})

        # Check all 3 ORGs have evidence
        if org1 not in org_evidence or org2 not in org_evidence or org3 not in org_evidence:
            return None

        ev1 = org_evidence[org1]
        ev2 = org_evidence[org2]
        ev3 = org_evidence[org3]

        return ThreeHopType2Result(
            pattern_type='cross_company_type2',
            connector_name=connector['connector_name'],
            connector_type=connector['connector_type'],
            connector_idf=connector['idf_score'],
            connector_org_degree=connector['org_degree'],
            hop1_org=org1,
            hop1_relationship=ev1['relationship'],
            hop1_source_file=ev1['source_file'],
            hop1_page_id=ev1['page_id'],
            hop1_chunk_id=ev1['chunk_id'],
            hop2_org=org2,
            hop2_relationship=ev2['relationship'],
            hop2_source_file=ev2['source_file'],
            hop2_page_id=ev2['page_id'],
            hop2_chunk_id=ev2['chunk_id'],
            hop3_org=org3,
            hop3_relationship=ev3['relationship'],
            hop3_source_file=ev3['source_file'],
            hop3_page_id=ev3['page_id'],
            hop3_chunk_id=ev3['chunk_id']
        )

    def generate_type2_triplets_from_connectors(
        self,
        connectors: List[Dict[str, Any]],
        org_filter: Optional[List[str]] = None,
        max_triplets_per_connector: int = 50,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Generate valid triplets from connectors found by find_connectors_with_multiple_orgs().

        Each returned triplet is GUARANTEED to share the connector (100% hit rate).

        Args:
            connectors: List from find_connectors_with_multiple_orgs()
            org_filter: Optional list of ORGs to include (e.g., sector filter)
            max_triplets_per_connector: Limit triplets generated per connector
            shuffle: Whether to shuffle results for diversity
            seed: Random seed for reproducibility

        Returns:
            List of (org1, org2, org3, connector_dict) tuples
        """
        import itertools
        import random

        if shuffle:
            random.seed(seed)

        all_triplets = []

        for conn in connectors:
            orgs = conn['connected_orgs']

            # Apply org filter if provided
            if org_filter:
                orgs = [o for o in orgs if o in org_filter]

            if len(orgs) < 3:
                continue

            # Generate triplets from this connector's ORGs
            triplet_combos = list(itertools.combinations(sorted(orgs), 3))

            if shuffle:
                random.shuffle(triplet_combos)

            # Limit triplets per connector to avoid over-representation
            triplet_combos = triplet_combos[:max_triplets_per_connector]

            for org1, org2, org3 in triplet_combos:
                all_triplets.append((org1, org2, org3, conn))

        if shuffle:
            random.shuffle(all_triplets)

        logger.info(f"Generated {len(all_triplets)} valid triplets from {len(connectors)} connectors")
        return all_triplets


def main():
    """Test connector discovery."""
    discovery = ConnectorDiscovery()

    try:
        # Test with NVDA-AMD pair
        print("=" * 80)
        print("Testing Connector Discovery: NVDA - AMD")
        print("=" * 80)

        # Get stats first
        stats = discovery.get_connector_stats('NVDA', 'AMD')
        print(f"\nConnector Statistics:")
        print(f"  Hard (IDF > 5.0): {stats['hard']}")
        print(f"  Medium (IDF 4.0-5.0): {stats['medium']}")
        print(f"  Easy (IDF 3.5-4.0): {stats['easy']}")
        print(f"  Generic (IDF < 3.5): {stats['generic']}")
        print(f"  Total: {stats['total']}")

        # Find hard connectors
        print("\n" + "-" * 40)
        print("Hard Connectors (IDF > 5.0):")
        print("-" * 40)
        connectors = discovery.find_connectors_by_difficulty('NVDA', 'AMD', 'hard', limit=10)
        for c in connectors:
            asymmetry = " [ASYMMETRIC]" if c.has_asymmetric_relationships else ""
            print(f"  {c.connector_name} ({c.connector_type})")
            print(f"    IDF: {c.idf_score:.2f}, Org Degree: {c.org_degree}")
            print(f"    NVDA: {c.org1_relationship} -> {c.org1_source_file}:{c.org1_page_id}")
            print(f"    AMD: {c.org2_relationship} -> {c.org2_source_file}:{c.org2_page_id}{asymmetry}")
            print()

        # Test Banks pair
        print("\n" + "=" * 80)
        print("Testing Connector Discovery: JPM - GS")
        print("=" * 80)

        stats = discovery.get_connector_stats('JPM', 'GS')
        print(f"\nConnector Statistics:")
        print(f"  Hard: {stats['hard']}, Medium: {stats['medium']}, Easy: {stats['easy']}")

        connectors = discovery.find_connectors(
            'JPM', 'GS',
            connector_types=['FIN_INST', 'REGULATORY_REQUIREMENT', 'ORG_REG'],
            min_idf=5.0,
            limit=5
        )
        print("\nTop FIN_INST/REGULATORY connectors (IDF > 5.0):")
        for c in connectors:
            print(f"  {c.connector_name}: IDF={c.idf_score:.2f}")
            print(f"    JPM: {c.org1_relationship}, GS: {c.org2_relationship}")

    finally:
        discovery.close()


if __name__ == "__main__":
    main()
