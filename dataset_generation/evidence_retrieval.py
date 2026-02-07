"""
Evidence Retrieval Module

This module retrieves chunk text from Neo4j for evidence grounding.
Given a connector result, it fetches the actual text from both companies'
10-K filings to provide context for QA generation.

Usage:
    from evidence_retrieval import EvidenceRetriever, Evidence

    retriever = EvidenceRetriever()
    evidence = retriever.get_evidence_for_connector(connector_result)
    print(evidence.org1_chunk_text)
    print(evidence.org2_chunk_text)
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkEvidence:
    """Evidence from a single chunk."""
    source_file: str
    page_id: str
    chunk_id: Optional[str]
    text: str
    ticker: Optional[str] = None

    @property
    def reference(self) -> str:
        """Return a citation reference string."""
        chunk_str = f":{self.chunk_id}" if self.chunk_id else ""
        return f"{self.source_file}:{self.page_id}{chunk_str}"


@dataclass
class ConnectorEvidence:
    """Complete evidence package for a connector between two companies."""
    connector_name: str
    connector_type: str
    idf_score: float

    org1_ticker: str
    org1_relationship: str
    org1_evidence: ChunkEvidence

    org2_ticker: str
    org2_relationship: str
    org2_evidence: ChunkEvidence

    @property
    def has_asymmetric_relationships(self) -> bool:
        """Check if relationships are different."""
        return self.org1_relationship != self.org2_relationship

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'connector': {
                'name': self.connector_name,
                'type': self.connector_type,
                'idf_score': self.idf_score
            },
            'org1': {
                'ticker': self.org1_ticker,
                'relationship': self.org1_relationship,
                'evidence': {
                    'source_file': self.org1_evidence.source_file,
                    'page_id': self.org1_evidence.page_id,
                    'chunk_id': self.org1_evidence.chunk_id,
                    'text': self.org1_evidence.text
                }
            },
            'org2': {
                'ticker': self.org2_ticker,
                'relationship': self.org2_relationship,
                'evidence': {
                    'source_file': self.org2_evidence.source_file,
                    'page_id': self.org2_evidence.page_id,
                    'chunk_id': self.org2_evidence.chunk_id,
                    'text': self.org2_evidence.text
                }
            }
        }


class EvidenceRetriever:
    """
    Retrieves chunk text from Neo4j for evidence grounding.

    Example:
        retriever = EvidenceRetriever()
        evidence = retriever.get_chunk_text('NVDA_10k_2024.pdf', 'page_41', 'chunk_1')
        print(evidence.text)
    """

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None
    ):
        # Load from environment or use defaults
        neo4j_auth = os.getenv("NEO4J_AUTH", "")
        if neo4j_auth:
            parts = neo4j_auth.split("/")
            if len(parts) == 2:
                neo4j_username = neo4j_username or parts[0]
                neo4j_password = neo4j_password or parts[1]

        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "<YOUR_NEO4J_URI>")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME", "<YOUR_NEO4J_USERNAME>")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "<YOUR_NEO4J_PASSWORD>")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j."""
        logger.info(f"Connecting to Neo4j at {self.neo4j_uri}...")
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password),
            max_connection_pool_size=100,  # Handle parallel workers
            connection_acquisition_timeout=60,  # Wait up to 60s for connection from pool
            connection_timeout=30,  # Connection establishment timeout
        )
        logger.info("Connected to Neo4j successfully")

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def get_chunk_text(
        self,
        source_file: str,
        page_id: str,
        chunk_id: Optional[str] = None
    ) -> Optional[ChunkEvidence]:
        """
        Retrieve chunk text from Neo4j.

        Args:
            source_file: The source file (e.g., 'NVDA_10k_2024.pdf')
            page_id: The page ID (e.g., 'page_41')
            chunk_id: Optional chunk ID (e.g., 'chunk_1')

        Returns:
            ChunkEvidence object with the text, or None if not found
        """
        if chunk_id:
            # Specific chunk
            query = """
            MATCH (c:Chunk {source_file: $source_file, page_id: $page_id, chunk_id: $chunk_id})
            RETURN c.text as text, c.ticker as ticker
            LIMIT 1
            """
            params = {'source_file': source_file, 'page_id': page_id, 'chunk_id': chunk_id}
        else:
            # All chunks from the page, concatenated
            query = """
            MATCH (c:Chunk {source_file: $source_file, page_id: $page_id})
            RETURN c.chunk_id as chunk_id, c.text as text, c.ticker as ticker
            ORDER BY c.chunk_id
            """
            params = {'source_file': source_file, 'page_id': page_id}

        with self.driver.session(database=self.neo4j_database) as session:
            result = session.run(query, **params)
            records = list(result)

            if not records:
                logger.warning(f"No chunk found for {source_file}:{page_id}:{chunk_id}")
                return None

            if chunk_id:
                # Single chunk
                record = records[0]
                return ChunkEvidence(
                    source_file=source_file,
                    page_id=page_id,
                    chunk_id=chunk_id,
                    text=record['text'] or "",
                    ticker=record['ticker']
                )
            else:
                # Concatenate all chunks from page
                texts = [r['text'] for r in records if r['text']]
                combined_text = "\n\n".join(texts)
                ticker = records[0]['ticker'] if records else None
                return ChunkEvidence(
                    source_file=source_file,
                    page_id=page_id,
                    chunk_id=None,
                    text=combined_text,
                    ticker=ticker
                )

    def get_page_context(
        self,
        source_file: str,
        page_id: str,
        context_pages: int = 0
    ) -> str:
        """
        Get text from a page with optional surrounding context.

        Args:
            source_file: The source file
            page_id: The target page ID (e.g., 'page_41')
            context_pages: Number of pages before/after to include

        Returns:
            Concatenated text from the page(s)
        """
        # Extract page number
        page_num = int(page_id.replace('page_', ''))

        # Build list of page IDs to fetch
        page_ids = []
        for i in range(max(1, page_num - context_pages), page_num + context_pages + 1):
            page_ids.append(f"page_{i}")

        query = """
        MATCH (c:Chunk {source_file: $source_file})
        WHERE c.page_id IN $page_ids
        RETURN c.page_id as page_id, c.chunk_id as chunk_id, c.text as text
        ORDER BY c.page_id, c.chunk_id
        """

        with self.driver.session(database=self.neo4j_database) as session:
            result = session.run(query, source_file=source_file, page_ids=page_ids)
            records = list(result)

            texts = []
            current_page = None
            for record in records:
                if record['page_id'] != current_page:
                    if current_page is not None:
                        texts.append("")  # Page break
                    current_page = record['page_id']
                    texts.append(f"[{record['page_id']}]")
                if record['text']:
                    texts.append(record['text'])

            return "\n".join(texts)

    def get_evidence_for_connector(
        self,
        connector_result,  # ConnectorResult from connector_discovery
        org1_ticker: str,
        org2_ticker: str
    ) -> Optional[ConnectorEvidence]:
        """
        Get complete evidence for a connector result.

        Args:
            connector_result: ConnectorResult from ConnectorDiscovery
            org1_ticker: First company ticker
            org2_ticker: Second company ticker

        Returns:
            ConnectorEvidence with text from both companies, or None if evidence not found
        """
        # Get evidence for org1
        org1_evidence = self.get_chunk_text(
            connector_result.org1_source_file,
            connector_result.org1_page_id,
            connector_result.org1_chunk_id
        )

        # Get evidence for org2
        org2_evidence = self.get_chunk_text(
            connector_result.org2_source_file,
            connector_result.org2_page_id,
            connector_result.org2_chunk_id
        )

        if not org1_evidence or not org2_evidence:
            logger.warning(f"Could not retrieve evidence for connector {connector_result.connector_name}")
            return None

        return ConnectorEvidence(
            connector_name=connector_result.connector_name,
            connector_type=connector_result.connector_type,
            idf_score=connector_result.idf_score,
            org1_ticker=org1_ticker,
            org1_relationship=connector_result.org1_relationship,
            org1_evidence=org1_evidence,
            org2_ticker=org2_ticker,
            org2_relationship=connector_result.org2_relationship,
            org2_evidence=org2_evidence
        )


def main():
    """Test evidence retrieval."""
    from connector_discovery import ConnectorDiscovery

    # Initialize
    discovery = ConnectorDiscovery()
    retriever = EvidenceRetriever()

    try:
        print("=" * 80)
        print("Testing Evidence Retrieval")
        print("=" * 80)

        # Find connectors
        connectors = discovery.find_connectors('NVDA', 'AMD', min_idf=5.0, limit=3)

        for connector in connectors:
            print(f"\n{'='*60}")
            print(f"Connector: {connector.connector_name} ({connector.connector_type})")
            print(f"IDF: {connector.idf_score:.2f}")
            print(f"{'='*60}")

            # Get evidence
            evidence = retriever.get_evidence_for_connector(connector, 'NVDA', 'AMD')

            if evidence:
                print(f"\n[NVDA - {evidence.org1_relationship}]")
                print(f"Source: {evidence.org1_evidence.reference}")
                print(f"Text preview: {evidence.org1_evidence.text[:500]}...")

                print(f"\n[AMD - {evidence.org2_relationship}]")
                print(f"Source: {evidence.org2_evidence.reference}")
                print(f"Text preview: {evidence.org2_evidence.text[:500]}...")

                if evidence.has_asymmetric_relationships:
                    print("\n[ASYMMETRIC RELATIONSHIPS - Good for comparative questions!]")
            else:
                print("Could not retrieve evidence")

    finally:
        discovery.close()
        retriever.close()


if __name__ == "__main__":
    main()
