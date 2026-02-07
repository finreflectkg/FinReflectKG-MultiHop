"""
Multi-hop QA Generation Module

This module generates multi-hop questions and answers from company pairs
using connectors discovered in the Neo4j knowledge graph.

Supports both 2-hop and 3-hop question generation:

2-Hop Patterns:
- cross_company: ORG1 → Connector ← ORG2
- cross_year (temporal): ORG(year1) → Connector ← ORG(year2)
- intra_doc: ORG → Connector ← ORG (same company, same year, different pages)

3-Hop Patterns:
- cross_company_type1: ORG1 → Connector1 → Connector2 ← ORG2
- entity_extension: ORG1 → Connector ← ORG2 → Entity
- temporal_extension: ORG(year1) → Connector ← ORG(year2) → Entity

Usage:
    from qa_generator import QAGenerator
    from pair_generator import load_config

    config = load_config()
    qa_gen = QAGenerator(config=config)

    # Generate 2-hop QA
    qa_items_2hop = qa_gen.generate_2hop_qa(pair)

    # Generate 3-hop QA
    qa_items_3hop = qa_gen.generate_3hop_qa(pair, pattern="cross_company_type1")
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from llm import LLMClient
from connector_discovery import ConnectorDiscovery, ConnectorDiscoveryConfig, ConnectorResult, ThreeHopResult, ThreeHopType2Result
from evidence_retrieval import EvidenceRetriever
from pair_generator import CompanyPair, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# LOAD PROMPTS FROM YAML
# =============================================================================

def load_prompts() -> Dict[str, Any]:
    """Load prompts from prompts.yaml"""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    with open(prompts_path) as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QAItem:
    """A generated QA item supporting 2-hop and 3-hop patterns."""
    question: str
    answer: str
    reasoning_steps: List[str]
    difficulty: str
    evidence_ids: List[str]

    # Hop configuration
    hop_count: int  # 2 or 3
    hop_pattern: str  # 'cross_company', 'cross_year', 'intra_doc', 'cross_company_type1', 'entity_extension', 'temporal_extension'

    # Pair metadata
    org1: str
    org2: str
    org1_year: int
    org2_year: int
    sector: str

    # Connector info (for 2-hop, this is the single connector; for 3-hop, this is connector1)
    connector_name: str
    connector_type: str
    connector_idf: float

    # Optional fields (must come after required fields in dataclass)
    sub_industry: Optional[str] = None

    # ORG1 evidence (ORG1 -> Connector relationship)
    org1_source_file: Optional[str] = None
    org1_page_id: Optional[str] = None
    org1_chunk_id: Optional[str] = None
    org1_chunk_text: Optional[str] = None
    org1_relationship: Optional[str] = None

    # ORG2 evidence (ORG2 -> Connector relationship)
    org2_source_file: Optional[str] = None
    org2_page_id: Optional[str] = None
    org2_chunk_id: Optional[str] = None
    org2_chunk_text: Optional[str] = None
    org2_relationship: Optional[str] = None

    # 3-hop specific: additional entity evidence
    ext_entity_source_file: Optional[str] = None
    ext_entity_page_id: Optional[str] = None
    ext_entity_chunk_id: Optional[str] = None
    ext_entity_chunk_text: Optional[str] = None
    ext_entity_relationship: Optional[str] = None

    # 3-hop specific fields
    connector2_name: Optional[str] = None  # For cross_company_type1
    connector2_type: Optional[str] = None
    connector2_idf: Optional[float] = None
    extended_entity_name: Optional[str] = None  # For entity_extension/temporal_extension
    extended_entity_type: Optional[str] = None
    extended_entity_idf: Optional[float] = None

    # Optional fields
    temporal_change: Optional[str] = None
    combined_idf_score: Optional[float] = None
    quality_score: Optional[int] = None  # Validation score from reflection (0-50)

    def to_dict(self, question_id: int = 0) -> Dict[str, Any]:
        """
        Convert to dictionary using old format naming convention.

        Old format structure:
        - pattern: Shows the path with relationships (e.g., "ORG ──[rel1]──→ COMP ←──[rel2]── ORG")
        - entities: {start, intermediate, end} for 2-hop; {start, node_2, node_3, end} for 3-hop
        - entity_types: Matching types for each entity
        - path_data: Contains start_node, hop_X_rel, intermediate_node/node_X, end_node
        - document_relationship: inter_document_cross_company or inter_document_same_company

        New additions kept:
        - reasoning_steps, difficulty, idf_score
        """
        # Determine document_relationship based on hop_pattern
        if self.hop_pattern in ['cross_company', 'cross_company_type1', 'cross_company_type2', 'entity_extension']:
            document_relationship = "inter_document_cross_company"
        elif self.hop_pattern in ['cross_year', 'cross_year_type1', 'cross_year_type2', 'temporal_extension']:
            document_relationship = "inter_document_same_company"
        elif self.hop_pattern in ['intra_doc', 'intra_doc_type1', 'intra_doc_type2']:
            document_relationship = "intra_document"
        elif self.hop_pattern in ['causal_shared_driver', 'causal_shared_outcome', 'causal_cross_anchor']:
            document_relationship = "inter_document_cross_company"  # causal patterns are cross-company
        else:
            document_relationship = "inter_document_cross_company"  # default

        # Build result based on hop_count
        if self.hop_count == 2:
            return self._to_dict_2hop(question_id, document_relationship)
        elif self.hop_count == 3:
            return self._to_dict_3hop(question_id, document_relationship)
        else:
            # Fallback for unknown hop_count
            return self._to_dict_2hop(question_id, document_relationship)

    def _to_dict_2hop(self, question_id: int, document_relationship: str) -> Dict[str, Any]:
        """
        Convert 2-hop QA item to old format.

        Pattern: ORG1 ──[rel1]──→ CONNECTOR ←──[rel2]── ORG2

        Mapping:
        - start = ORG1
        - intermediate = CONNECTOR (bridge entity)
        - end = ORG2
        """
        # Build pattern string showing the bridge structure
        rel1 = self.org1_relationship or "RELATES_TO"
        rel2 = self.org2_relationship or "RELATES_TO"
        connector_type = self.connector_type or "ENTITY"

        pattern = f"ORG -[{rel1}]-> {connector_type} <-[{rel2}]- ORG"

        # Build entities and entity_types
        entities = {
            "start": self.org1,
            "intermediate": self.connector_name,
            "end": self.org2
        }

        entity_types = {
            "start": "ORG",
            "intermediate": connector_type,
            "end": "ORG"
        }

        # Build path_data with evidence
        path_data = {
            "start_node": {
                "id": self.org1,
                "name": self.org1,
                "type": "ORG",
                "year": self.org1_year
            },
            "hop_1_rel": {
                "source_file": self.org1_source_file,
                "page_id": self.org1_page_id,
                "chunk_id": self.org1_chunk_id,
                "chunk_text": self.org1_chunk_text,
                "relationship": self.org1_relationship
            },
            "intermediate_node": {
                "id": self.connector_name.replace(" ", "_") if self.connector_name else None,
                "name": self.connector_name,
                "type": connector_type,
                "idf_score": self.connector_idf
            },
            "hop_2_rel": {
                "source_file": self.org2_source_file,
                "page_id": self.org2_page_id,
                "chunk_id": self.org2_chunk_id,
                "chunk_text": self.org2_chunk_text,
                "relationship": self.org2_relationship
            },
            "end_node": {
                "id": self.org2,
                "name": self.org2,
                "type": "ORG",
                "year": self.org2_year
            }
        }

        result = {
            "question_id": question_id,
            "question": self.question,
            "answer": self.answer,
            "reasoning_steps": self.reasoning_steps,
            "difficulty": self.difficulty,
            "idf_score": self.connector_idf,
            "sector": self.sector,
            "sub_industry": self.sub_industry,
            "pattern": pattern,
            "entities": entities,
            "entity_types": entity_types,
            "hop_count": self.hop_count,
            "document_relationship": document_relationship,
            "path_data": path_data
        }

        # Add temporal_change if present
        if self.temporal_change:
            result["temporal_change"] = self.temporal_change

        # Add quality_score if present
        if self.quality_score is not None:
            result["quality_score"] = self.quality_score

        return result

    def _to_dict_3hop(self, question_id: int, document_relationship: str) -> Dict[str, Any]:
        """
        Convert 3-hop QA item to old format.

        Patterns:
        - cross_company_type1: ORG1 ──→ Connector1 ──→ Connector2 ←── ORG2
        - entity_extension: ORG1 ──→ Connector ←── ORG2 ──→ Entity
        - temporal_extension: ORG(y1) ──→ Connector ←── ORG(y2) ──→ Entity

        Mapping:
        - start = ORG1
        - node_2 = Connector1 / Connector
        - node_3 = Connector2 / ORG2 / Entity (depends on pattern)
        - end = ORG2 / Entity
        """
        rel1 = self.org1_relationship or "RELATES_TO"
        rel2 = self.ext_entity_relationship or "RELATES_TO"
        rel3 = self.org2_relationship or "RELATES_TO"
        connector1_type = self.connector_type or "ENTITY"

        if self.hop_pattern in ['cross_company_type1', 'cross_year_type1', 'intra_doc_type1',
                                 'causal_shared_driver', 'causal_shared_outcome', 'causal_cross_anchor']:
            # Pattern: ORG1 ──→ Connector1 ──→ Connector2 ←── ORG2 (all Type 1 and causal patterns)
            connector2_type = self.connector2_type or "ENTITY"

            pattern = f"ORG -[{rel1}]-> {connector1_type} -[{rel2}]-> {connector2_type} <-[{rel3}]- ORG"

            entities = {
                "start": self.org1,
                "node_2": self.connector_name,
                "node_3": self.connector2_name,
                "end": self.org2
            }

            entity_types = {
                "start": "ORG",
                "node_2": connector1_type,
                "node_3": connector2_type,
                "end": "ORG"
            }

            path_data = {
                "start_node": {
                    "id": self.org1,
                    "name": self.org1,
                    "type": "ORG",
                    "year": self.org1_year
                },
                "hop_1_rel": {
                    "source_file": self.org1_source_file,
                    "page_id": self.org1_page_id,
                    "chunk_id": self.org1_chunk_id,
                    "chunk_text": self.org1_chunk_text,
                    "relationship": self.org1_relationship
                },
                "node_2": {
                    "id": self.connector_name.replace(" ", "_") if self.connector_name else None,
                    "name": self.connector_name,
                    "type": connector1_type,
                    "idf_score": self.connector_idf
                },
                "hop_2_rel": {
                    "source_file": self.ext_entity_source_file,
                    "page_id": self.ext_entity_page_id,
                    "chunk_id": self.ext_entity_chunk_id,
                    "chunk_text": self.ext_entity_chunk_text,
                    "relationship": self.ext_entity_relationship
                },
                "node_3": {
                    "id": self.connector2_name.replace(" ", "_") if self.connector2_name else None,
                    "name": self.connector2_name,
                    "type": connector2_type,
                    "idf_score": self.connector2_idf
                },
                "hop_3_rel": {
                    "source_file": self.org2_source_file,
                    "page_id": self.org2_page_id,
                    "chunk_id": self.org2_chunk_id,
                    "chunk_text": self.org2_chunk_text,
                    "relationship": self.org2_relationship
                },
                "end_node": {
                    "id": self.org2,
                    "name": self.org2,
                    "type": "ORG",
                    "year": self.org2_year
                }
            }

            idf_score = self.combined_idf_score or self.connector_idf

        elif self.hop_pattern in ['cross_company_type2', 'cross_year_type2', 'intra_doc_type2']:
            # Pattern: ORG1 → Connector ← ORG2, Connector ← ORG3 (all Type 2 patterns - multi-branch)
            connector2_type = self.connector_type or "ENTITY"  # Same connector for all 3 orgs

            pattern = f"ORG -[{rel1}]-> {connector1_type} <-[{rel2}]- ORG <-[{rel3}]- ORG"

            entities = {
                "start": self.org1,
                "node_2": self.connector_name,
                "node_3": self.org2 if hasattr(self, 'org3') else self.connector_name,
                "end": getattr(self, 'org3', self.org2)
            }

            entity_types = {
                "start": "ORG",
                "node_2": connector1_type,
                "node_3": "ORG",
                "end": "ORG"
            }

            path_data = {
                "start_node": {
                    "id": self.org1,
                    "name": self.org1,
                    "type": "ORG",
                    "year": self.org1_year
                },
                "hop_1_rel": {
                    "source_file": self.org1_source_file,
                    "page_id": self.org1_page_id,
                    "chunk_id": self.org1_chunk_id,
                    "chunk_text": self.org1_chunk_text,
                    "relationship": self.org1_relationship
                },
                "connector_node": {
                    "id": self.connector_name.replace(" ", "_") if self.connector_name else None,
                    "name": self.connector_name,
                    "type": connector1_type,
                    "idf_score": self.connector_idf
                },
                "hop_2_rel": {
                    "source_file": self.org2_source_file,
                    "page_id": self.org2_page_id,
                    "chunk_id": self.org2_chunk_id,
                    "chunk_text": self.org2_chunk_text,
                    "relationship": self.org2_relationship
                },
                "hop_3_rel": {
                    "source_file": getattr(self, 'ext_entity_source_file', None),
                    "page_id": getattr(self, 'ext_entity_page_id', None),
                    "chunk_id": getattr(self, 'ext_entity_chunk_id', None),
                    "chunk_text": getattr(self, 'ext_entity_chunk_text', None),
                    "relationship": getattr(self, 'ext_entity_relationship', None)
                }
            }

            idf_score = self.connector_idf

        elif self.hop_pattern in ['entity_extension', 'temporal_extension']:
            # Pattern: ORG1 ──→ Connector ←── ORG2 ──→ Entity
            extended_type = self.extended_entity_type or "ENTITY"

            pattern = f"ORG -[{rel1}]-> {connector1_type} <-[{rel2}]- ORG -[{rel3}]-> {extended_type}"

            entities = {
                "start": self.org1,
                "node_2": self.connector_name,
                "node_3": self.org2,
                "end": self.extended_entity_name
            }

            entity_types = {
                "start": "ORG",
                "node_2": connector1_type,
                "node_3": "ORG",
                "end": extended_type
            }

            path_data = {
                "start_node": {
                    "id": self.org1,
                    "name": self.org1,
                    "type": "ORG",
                    "year": self.org1_year
                },
                "hop_1_rel": {
                    "source_file": self.org1_source_file,
                    "page_id": self.org1_page_id,
                    "chunk_id": self.org1_chunk_id,
                    "chunk_text": self.org1_chunk_text,
                    "relationship": self.org1_relationship
                },
                "node_2": {
                    "id": self.connector_name.replace(" ", "_") if self.connector_name else None,
                    "name": self.connector_name,
                    "type": connector1_type,
                    "idf_score": self.connector_idf
                },
                "hop_2_rel": {
                    "source_file": self.org2_source_file,
                    "page_id": self.org2_page_id,
                    "chunk_id": self.org2_chunk_id,
                    "chunk_text": self.org2_chunk_text,
                    "relationship": self.org2_relationship
                },
                "node_3": {
                    "id": self.org2,
                    "name": self.org2,
                    "type": "ORG",
                    "year": self.org2_year
                },
                "hop_3_rel": {
                    "source_file": self.ext_entity_source_file,
                    "page_id": self.ext_entity_page_id,
                    "chunk_id": self.ext_entity_chunk_id,
                    "chunk_text": self.ext_entity_chunk_text,
                    "relationship": self.ext_entity_relationship
                },
                "end_node": {
                    "id": self.extended_entity_name.replace(" ", "_") if self.extended_entity_name else None,
                    "name": self.extended_entity_name,
                    "type": extended_type,
                    "idf_score": self.extended_entity_idf
                }
            }

            idf_score = self.combined_idf_score or self.connector_idf

        else:
            # Fallback for unknown 3-hop pattern
            pattern = f"ORG -[{rel1}]-> {connector1_type} -[{rel2}]-> ENTITY <-[{rel3}]- ORG"
            entities = {"start": self.org1, "node_2": self.connector_name, "node_3": "UNKNOWN", "end": self.org2}
            entity_types = {"start": "ORG", "node_2": connector1_type, "node_3": "ENTITY", "end": "ORG"}
            path_data = {}
            idf_score = self.connector_idf

        result = {
            "question_id": question_id,
            "question": self.question,
            "answer": self.answer,
            "reasoning_steps": self.reasoning_steps,
            "difficulty": self.difficulty,
            "idf_score": idf_score,
            "sector": self.sector,
            "sub_industry": self.sub_industry,
            "pattern": pattern,
            "entities": entities,
            "entity_types": entity_types,
            "hop_count": self.hop_count,
            "document_relationship": document_relationship,
            "path_data": path_data
        }

        # Add temporal_change if present
        if self.temporal_change:
            result["temporal_change"] = self.temporal_change

        # Add quality_score if present
        if self.quality_score is not None:
            result["quality_score"] = self.quality_score

        return result


# =============================================================================
# QA GENERATOR
# =============================================================================

class QAGenerator:
    """
    Generates multi-hop QA items from company pairs.

    Example:
        config = load_config()
        qa_gen = QAGenerator(config=config)

        for pair in pairs:
            qa_items = qa_gen.generate_qa_for_pair(pair, max_questions=3)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_model: Optional[str] = None
    ):
        """
        Initialize the QA generator.

        Args:
            config: Configuration dictionary (from load_config())
            llm_model: Name of LLM model to use (default: from config.yaml)
        """
        self.config = config or load_config()

        # Neo4j settings (from config.yaml if provided)
        neo_cfg = (self.config or {}).get("neo4j", {})

        # Initialize connector discovery for finding connectors
        discovery_cfg = ConnectorDiscoveryConfig(
            neo4j_uri=neo_cfg.get("uri", ConnectorDiscoveryConfig.neo4j_uri),
            neo4j_username=neo_cfg.get("username", ConnectorDiscoveryConfig.neo4j_username),
            neo4j_password=neo_cfg.get("password", ConnectorDiscoveryConfig.neo4j_password),
            neo4j_database=neo_cfg.get("database", ConnectorDiscoveryConfig.neo4j_database),
            total_orgs=self.config.get("total_orgs", ConnectorDiscoveryConfig.total_orgs),
        )
        self.discovery = ConnectorDiscovery(config=discovery_cfg)

        # Initialize evidence retriever
        self.retriever = EvidenceRetriever(
            neo4j_uri=neo_cfg.get("uri"),
            neo4j_username=neo_cfg.get("username"),
            neo4j_password=neo_cfg.get("password"),
            neo4j_database=neo_cfg.get("database"),
        )

        # Initialize LLM client (uses config.yaml, model_name optional)
        self.llm_client = LLMClient(llm_model)
        logger.info(f"Initialized QAGenerator with LLM")

        # IDF thresholds from config
        idf_config = self.config.get('idf_thresholds', {})
        self.hard_threshold = idf_config.get('hard', 5.0)
        self.medium_threshold = idf_config.get('medium', 4.0)

        # QA validation settings from config (reflection mechanism)
        qa_val_config = self.config.get('qa_validation', {})
        self.prompt_variant = qa_val_config.get('prompt_variant', 'default')  # 'default' or 'v3_binary'
        self.quality_threshold = qa_val_config.get('quality_threshold', 35)
        self.max_attempts = qa_val_config.get('max_attempts', 3)
        self.base_temperature = qa_val_config.get('base_temperature', 0.3)
        self.temperature_increment = qa_val_config.get('temperature_increment', 0.1)
        
        if self.prompt_variant == 'v3_binary':
            logger.info(f"QA validation: variant=v3_binary (pass/fail), max_attempts={self.max_attempts}")
        else:
            logger.info(f"QA validation: variant=default, threshold={self.quality_threshold}/50, max_attempts={self.max_attempts}")

    def close(self):
        """Close resources."""
        if self.discovery:
            self.discovery.close()
        if self.retriever:
            self.retriever.close()

    def _get_difficulty(self, idf_score: float) -> str:
        """Determine difficulty level from IDF score."""
        if idf_score >= self.hard_threshold:
            return "hard"
        elif idf_score >= self.medium_threshold:
            return "medium"
        else:
            return "easy"

    def _format_evidence(self, chunks: List[Dict[str, Any]], max_chunks: int = 3) -> str:
        """Format evidence chunks for prompt."""
        if not chunks:
            return "No evidence found."

        formatted = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            text = chunk.get('text', chunk.get('content', ''))
            page = chunk.get('page_number', 'N/A')
            section = chunk.get('section', 'Unknown')

            formatted.append(f"[{chunk_id}] (Page {page}, Section: {section})\n{text[:1500]}...")

        return "\n\n".join(formatted)

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON response."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Handle markdown code blocks
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            return None

    def _validate_qa(
        self,
        question: str,
        answer: str,
        pattern: str,
        entity_chain: str,
        evidence: str,
        source_files: str,
        quality_threshold: Optional[int] = None
    ) -> tuple[bool, int, str]:
        """
        Validate QA pair quality using the validation prompt (supports multiple variants).

        Args:
            question: Generated question
            answer: Generated answer
            pattern: The hop pattern (e.g., "ORG1 → Connector ← ORG2")
            entity_chain: Entity chain string (e.g., "NVDA → TSMC → AMD")
            evidence: Combined evidence text from all chunks
            source_files: Source file references
            quality_threshold: Minimum score to accept (only used for 'default' variant)

        Returns:
            Tuple of (accepted: bool, score: int, explanation: str)
            For v3_binary: score is 50 if pass, 0 if fail (for compatibility)
        """
        try:
            # Select prompt variant
            variant = self.prompt_variant
            if variant == 'v3_binary':
                prompt_key = 'validation_v3_binary'
            else:
                prompt_key = 'validation'
            
            if prompt_key not in PROMPTS:
                logger.warning(f"Prompt variant '{variant}' not found, falling back to 'default'")
                prompt_key = 'validation'
                variant = 'default'

            user_prompt = PROMPTS[prompt_key]['user'].format(
                pattern=pattern,
                entity_chain=entity_chain,
                question=question,
                answer=answer,
                evidence=evidence,
                source_files=source_files
            )

            response = self.llm_client.complete(
                system_prompt=PROMPTS[prompt_key]['system'],
                user_prompt=user_prompt
            )

            parsed = self._parse_llm_response(response)
            if not parsed:
                return False, 0, "Failed to parse validation response"

            # Parse response based on variant
            if variant == 'v3_binary':
                decision = parsed.get('decision', 'fail').lower()
                accepted = decision == 'pass'
                # Convert binary to score for compatibility: pass=50, fail=0
                score = 50 if accepted else 0
                
                failure_tags = parsed.get('failure_tags', [])
                must_fix_issues = parsed.get('must_fix_issues', [])
                rewrite_suggestions = parsed.get('rewrite_suggestions', [])
                notes = parsed.get('notes', 'No explanation provided')
                
                # Build feedback from binary format
                feedback = notes
                if failure_tags:
                    feedback += f"\n\nFailure tags: {', '.join(failure_tags)}"
                if must_fix_issues:
                    feedback += f"\n\nMust fix issues:\n- " + "\n- ".join(must_fix_issues)
                if rewrite_suggestions:
                    feedback += f"\n\nRewrite suggestions:\n- " + "\n- ".join(rewrite_suggestions)
            else:
                # Default scoring variant
                score = parsed.get('score', 0)
                decision = parsed.get('decision', 'reject').lower()
                explanation = parsed.get('explanation', 'No explanation provided')
                issues = parsed.get('issues', [])
                suggestions = parsed.get('suggestions', [])
                
                # Use provided threshold or instance default
                threshold = quality_threshold if quality_threshold is not None else self.quality_threshold
                accepted = score >= threshold
                
                # Build detailed feedback for reflection
                feedback = explanation
                if issues:
                    feedback += f"\n\nIssues found:\n- " + "\n- ".join(issues)
                if suggestions:
                    feedback += f"\n\nSuggestions:\n- " + "\n- ".join(suggestions)

            if variant == 'v3_binary':
                logger.info(f"Validation result (variant={variant}): decision={decision}, score={score}/50 (binary: {'PASS' if accepted else 'FAIL'}), accepted={accepted}")
            else:
                logger.info(f"Validation result (variant={variant}): decision={decision}, score={score}/50, accepted={accepted}")
            return accepted, score, feedback

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, 0, f"Validation error: {str(e)}"

    def _generate_with_reflection(
        self,
        system_prompt: str,
        user_prompt_template: str,
        prompt_vars: dict,
        pattern: str,
        entity_chain: str,
        evidence: str,
        source_files: str,
        max_attempts: Optional[int] = None,
        base_temperature: Optional[float] = None,
        temperature_increment: Optional[float] = None,
        quality_threshold: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate QA with reflection mechanism - validate and retry with feedback.

        Args:
            system_prompt: System prompt for generation
            user_prompt_template: User prompt template string
            prompt_vars: Variables to format into user prompt
            pattern: Pattern string for validation
            entity_chain: Entity chain for validation
            evidence: Evidence text for validation
            source_files: Source files for validation
            max_attempts: Maximum generation attempts
            base_temperature: Starting temperature
            temperature_increment: Temperature increase per retry
            quality_threshold: Minimum validation score

        Returns:
            Parsed QA dict if accepted, None if all attempts failed
        """
        # Use instance defaults if not provided
        max_attempts = max_attempts if max_attempts is not None else self.max_attempts
        base_temperature = base_temperature if base_temperature is not None else self.base_temperature
        temperature_increment = temperature_increment if temperature_increment is not None else self.temperature_increment
        quality_threshold = quality_threshold if quality_threshold is not None else self.quality_threshold

        feedback_from_previous = None

        for attempt in range(max_attempts):
            # Format user prompt, adding feedback if available
            user_prompt = user_prompt_template.format(**prompt_vars)

            if feedback_from_previous:
                user_prompt += f"""

<previous_attempt_feedback>
Your previous attempt was rejected. Please address this feedback:
{feedback_from_previous}
</previous_attempt_feedback>

Generate an improved QA pair that addresses the issues above."""

            # Generate with increasing temperature on retries
            temperature = base_temperature + (attempt * temperature_increment)

            try:
                response = self.llm_client.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature
                )

                parsed = self._parse_llm_response(response)
                if not parsed:
                    logger.warning(f"Attempt {attempt + 1}: Failed to parse response")
                    feedback_from_previous = "Response was not valid JSON. Please return only valid JSON."
                    continue

                question = parsed.get('question', '')
                answer = parsed.get('answer', '')

                # Check for "no meaningful question" response
                if 'no meaningful' in question.lower():
                    logger.info(f"LLM indicated no meaningful question possible")
                    return None

                if not question or not answer:
                    feedback_from_previous = "Question or answer was empty. Both are required."
                    continue

                # Validate the QA pair
                accepted, score, explanation = self._validate_qa(
                    question=question,
                    answer=answer,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=evidence,
                    source_files=source_files,
                    quality_threshold=quality_threshold
                )

                if accepted:
                    if self.prompt_variant == 'v3_binary':
                        # Binary variant: score is always 0 or 50
                        logger.info(f"QA accepted on attempt {attempt + 1} with score {score}/50 (binary: PASS)")
                    else:
                        logger.info(f"QA accepted on attempt {attempt + 1} with score {score}/50")
                    parsed['quality_score'] = score
                    return parsed
                else:
                    if self.prompt_variant == 'v3_binary':
                        # Binary variant: score is always 0 or 50
                        logger.info(f"Attempt {attempt + 1} rejected (score: {score}/50, binary: FAIL)")
                    else:
                        logger.info(f"Attempt {attempt + 1} rejected (score: {score}/50)")
                    feedback_from_previous = explanation

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                feedback_from_previous = f"Generation failed with error: {str(e)}"
                continue

        logger.warning(f"All {max_attempts} attempts failed validation")
        return None

    def generate_qa_for_cross_company_pair(
        self,
        pair: CompanyPair,
        max_questions: int = 3,
        connectors: Optional[List[ConnectorResult]] = None
    ) -> List[QAItem]:
        """
        Generate QA items for a Type 1 (cross-company) pair.

        Args:
            pair: CompanyPair object
            max_questions: Maximum number of questions to generate
            connectors: Optional pre-fetched connectors (if None, will fetch)

        Returns:
            List of QAItem objects
        """
        if pair.pair_category != 'cross_company':
            logger.warning(f"Pair is not cross_company: {pair.pair_category}")

        # Get connectors if not provided
        if connectors is None:
            # Use year-filtered query to get connectors from the correct year
            connectors = self.discovery.find_connectors_by_year(
                org1=pair.org1,
                org2=pair.org2,
                year1=pair.org1_year,
                year2=pair.org2_year,
                connector_types=pair.connector_types,
                min_idf=self.medium_threshold - 0.5  # Get medium and hard
            )

        if not connectors:
            logger.warning(f"No connectors found for {pair.org1}-{pair.org2}")
            return []

        # Sort by IDF (hardest first) and take top connectors
        # ConnectorResult is a dataclass, use attribute access
        connectors = sorted(connectors, key=lambda c: -c.idf_score)[:max_questions]

        qa_items = []

        for connector in connectors:
            # ConnectorResult dataclass attributes
            connector_name = connector.connector_name
            connector_type = connector.connector_type
            idf_score = connector.idf_score
            difficulty = self._get_difficulty(idf_score)

            # ==================== DEBUG: CONNECTOR INFO ====================
            print(f"\n{'='*60}")
            print(f"DEBUG: Processing Connector")
            print(f"{'='*60}")
            print(f"  Name: {connector_name}")
            print(f"  Type: {connector_type}")
            print(f"  IDF Score: {idf_score:.2f}")
            print(f"  Difficulty: {difficulty}")
            print(f"  Org1 Source: {connector.org1_source_file}:{connector.org1_page_id}")
            print(f"  Org2 Source: {connector.org2_source_file}:{connector.org2_page_id}")
            # ===============================================================

            # Get evidence for both companies using ConnectorResult
            evidence = self.retriever.get_evidence_for_connector(
                connector, pair.org1, pair.org2
            )

            if not evidence:
                logger.debug(f"Skipping connector {connector_name}: missing evidence")
                continue

            evidence1_text = evidence.org1_evidence.text if evidence.org1_evidence else ""
            evidence2_text = evidence.org2_evidence.text if evidence.org2_evidence else ""

            # ==================== DEBUG: EVIDENCE TEXT ====================
            # print(f"\n{'='*60}")
            # print(f"DEBUG: Evidence Retrieved")
            # print(f"{'='*60}")
            # print(f"  {pair.org1} Evidence ({len(evidence1_text)} chars):")
            # print(f"  Reference: {evidence.org1_evidence.reference if evidence.org1_evidence else 'N/A'}")
            # print(f"  Preview: {evidence1_text[:500]}..." if evidence1_text else "  [EMPTY]")
            # print(f"\n  {pair.org2} Evidence ({len(evidence2_text)} chars):")
            # print(f"  Reference: {evidence.org2_evidence.reference if evidence.org2_evidence else 'N/A'}")
            # print(f"  Preview: {evidence2_text[:500]}..." if evidence2_text else "  [EMPTY]")
            # ==============================================================

            if not evidence1_text or not evidence2_text:
                logger.debug(f"Skipping connector {connector_name}: incomplete evidence text")
                continue

            # Format evidence with source references
            evidence1_formatted = f"[{evidence.org1_evidence.reference}]\n{evidence1_text}"
            evidence2_formatted = f"[{evidence.org2_evidence.reference}]\n{evidence2_text}"

            # Prepare prompt variables
            prompt_vars = {
                'org1': pair.org1,
                'org2': pair.org2,
                'year1': pair.org1_year,
                'year2': pair.org2_year,
                'sector': pair.sector or 'Unknown',
                'org1_source_file': connector.org1_source_file,
                'org2_source_file': connector.org2_source_file,
                'org1_relationship': connector.org1_relationship,
                'org2_relationship': connector.org2_relationship,
                'connector_name': connector_name,
                'connector_type': connector_type,
                'idf_score': idf_score,
                'evidence1': evidence1_formatted,
                'evidence2': evidence2_formatted,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{pair.org1} -[{connector.org1_relationship}]-> {connector_name} <-[{connector.org2_relationship}]- {pair.org2}"
            entity_chain = f"{pair.org1} -> {connector_name} ({connector_type}) -> {pair.org2}"
            combined_evidence = f"Source A ({pair.org1}):\n{evidence1_formatted}\n\nSource B ({pair.org2}):\n{evidence2_formatted}"
            source_files = f"{connector.org1_source_file}, {connector.org2_source_file}"

            # Generate QA with reflection mechanism
            try:
                parsed = self._generate_with_reflection(
                    system_prompt=PROMPTS['cross_company']['system'],
                    user_prompt_template=PROMPTS['cross_company']['user'],
                    prompt_vars=prompt_vars,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=combined_evidence,
                    source_files=source_files,
                    # Uses config defaults: max_attempts, quality_threshold
                )

                if not parsed:
                    logger.info(f"Skipping connector {connector_name}: failed validation after retries")
                    continue

                # Get quality score from validation
                quality_score = parsed.get('quality_score', 0)

                qa_item = QAItem(
                    question=parsed.get('question', ''),
                    answer=parsed.get('answer', ''),
                    reasoning_steps=parsed.get('reasoning_steps', []),
                    difficulty=difficulty,
                    evidence_ids=parsed.get('evidence_ids', []),
                    hop_count=2,
                    hop_pattern='cross_company',
                    org1=pair.org1,
                    org2=pair.org2,
                    org1_year=pair.org1_year,
                    org2_year=pair.org2_year,
                    sector=pair.sector,
                    sub_industry=pair.sub_industry,
                    connector_name=connector_name,
                    connector_type=connector_type,
                    connector_idf=idf_score,
                    # ORG1 evidence
                    org1_source_file=connector.org1_source_file,
                    org1_page_id=connector.org1_page_id,
                    org1_chunk_id=connector.org1_chunk_id,
                    org1_chunk_text=evidence1_text,
                    org1_relationship=connector.org1_relationship,
                    # ORG2 evidence
                    org2_source_file=connector.org2_source_file,
                    org2_page_id=connector.org2_page_id,
                    org2_chunk_id=connector.org2_chunk_id,
                    org2_chunk_text=evidence2_text,
                    org2_relationship=connector.org2_relationship,
                    # Quality score from validation
                    quality_score=quality_score
                )

                qa_items.append(qa_item)
                logger.info(f"Generated cross_company QA for {pair.org1}-{pair.org2} ({connector_name}) [score: {quality_score}/50]")

            except Exception as e:
                logger.error(f"Failed to generate QA for connector {connector_name}: {e}")
                continue

        return qa_items

    def generate_qa_for_cross_year_pair(
        self,
        pair: CompanyPair,
        max_questions: int = 3,
        connectors: Optional[List[ConnectorResult]] = None
    ) -> List[QAItem]:
        """
        Generate QA items for a Type 2 (temporal) pair.

        Args:
            pair: CompanyPair object (same company, different years)
            max_questions: Maximum number of questions to generate
            connectors: Optional pre-fetched connectors with change info

        Returns:
            List of QAItem objects
        """
        if pair.pair_category != 'cross_year':
            logger.warning(f"Pair is not cross_year: {pair.pair_category}")

        if not pair.is_same_company:
            logger.error(f"cross_year pair must be same company: {pair.org1} vs {pair.org2}")
            return []

        # Get temporal connectors if not provided
        if connectors is None:
            connectors = self.discovery.find_temporal_connectors(
                org=pair.org1,
                year1=pair.org1_year,
                year2=pair.org2_year,
                connector_types=pair.connector_types
            )

        if not connectors:
            logger.warning(f"No temporal connectors found for {pair.org1} ({pair.org1_year}-{pair.org2_year})")
            return []

        # Prioritize connectors with relationship changes (asymmetric)
        # ConnectorResult is a dataclass, use attribute access
        changed_connectors = [c for c in connectors if c.has_asymmetric_relationships]
        stable_connectors = [c for c in connectors if not c.has_asymmetric_relationships]

        # Take mix: prioritize changed, then stable
        selected = changed_connectors[:max_questions]
        if len(selected) < max_questions:
            selected.extend(stable_connectors[:max_questions - len(selected)])

        qa_items = []

        for connector in selected:
            # ConnectorResult dataclass attributes
            connector_name = connector.connector_name
            connector_type = connector.connector_type
            idf_score = connector.idf_score
            difficulty = self._get_difficulty(idf_score)
            relationship_changed = connector.has_asymmetric_relationships

            # Get evidence for both years using ConnectorResult
            # For temporal pairs, org1 is same as org2 (same company)
            evidence = self.retriever.get_evidence_for_connector(
                connector, pair.org1, pair.org1
            )

            if not evidence:
                logger.debug(f"Skipping connector {connector_name}: no evidence found")
                continue

            evidence1_text = evidence.org1_evidence.text if evidence.org1_evidence else ""
            evidence2_text = evidence.org2_evidence.text if evidence.org2_evidence else ""

            if not evidence1_text and not evidence2_text:
                logger.debug(f"Skipping connector {connector_name}: no evidence in either year")
                continue

            # Format evidence with source references
            if evidence1_text:
                evidence1_formatted = f"[{evidence.org1_evidence.reference}]\n{evidence1_text}"
            else:
                evidence1_formatted = "No evidence for this year (connector may be new)"

            if evidence2_text:
                evidence2_formatted = f"[{evidence.org2_evidence.reference}]\n{evidence2_text}"
            else:
                evidence2_formatted = "No evidence for this year (connector may have been discontinued)"

            # Prepare prompt variables
            prompt_vars = {
                'org': pair.org1,
                'year1': pair.org1_year,
                'year2': pair.org2_year,
                'sector': pair.sector or 'Unknown',
                'connector_name': connector_name,
                'connector_type': connector_type,
                'relationship_changed': "Yes - relationship changed between years" if relationship_changed else "No - relationship stable",
                'org1_source_file': connector.org1_source_file,
                'org2_source_file': connector.org2_source_file,
                'org1_relationship': connector.org1_relationship,
                'org2_relationship': connector.org2_relationship,
                'evidence1': evidence1_formatted,
                'evidence2': evidence2_formatted,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{pair.org1}({pair.org1_year}) -[{connector.org1_relationship}]-> {connector_name} <-[{connector.org2_relationship}]- {pair.org1}({pair.org2_year})"
            entity_chain = f"{pair.org1}({pair.org1_year}) -> {connector_name} ({connector_type}) -> {pair.org1}({pair.org2_year})"
            combined_evidence = f"Year {pair.org1_year}:\n{evidence1_formatted}\n\nYear {pair.org2_year}:\n{evidence2_formatted}"
            source_files = f"{connector.org1_source_file}, {connector.org2_source_file}"

            # Generate QA with reflection mechanism
            try:
                parsed = self._generate_with_reflection(
                    system_prompt=PROMPTS['cross_year']['system'],
                    user_prompt_template=PROMPTS['cross_year']['user'],
                    prompt_vars=prompt_vars,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=combined_evidence,
                    source_files=source_files,
                    # Uses config defaults: max_attempts, quality_threshold
                )

                if not parsed:
                    logger.info(f"Skipping connector {connector_name}: failed validation after retries")
                    continue

                # Get quality score from validation
                quality_score = parsed.get('quality_score', 0)

                qa_item = QAItem(
                    question=parsed.get('question', ''),
                    answer=parsed.get('answer', ''),
                    reasoning_steps=parsed.get('reasoning_steps', []),
                    difficulty=difficulty,
                    evidence_ids=parsed.get('evidence_ids', []),
                    hop_count=2,
                    hop_pattern='cross_year',
                    org1=pair.org1,
                    org2=pair.org2,
                    org1_year=pair.org1_year,
                    org2_year=pair.org2_year,
                    sector=pair.sector,
                    sub_industry=pair.sub_industry,
                    connector_name=connector_name,
                    connector_type=connector_type,
                    connector_idf=idf_score,
                    temporal_change=parsed.get('temporal_change', 'unknown'),
                    # Year1 evidence
                    org1_source_file=connector.org1_source_file,
                    org1_page_id=connector.org1_page_id,
                    org1_chunk_id=connector.org1_chunk_id,
                    org1_chunk_text=evidence1_text,
                    org1_relationship=connector.org1_relationship,
                    # Year2 evidence
                    org2_source_file=connector.org2_source_file,
                    org2_page_id=connector.org2_page_id,
                    org2_chunk_id=connector.org2_chunk_id,
                    org2_chunk_text=evidence2_text,
                    org2_relationship=connector.org2_relationship,
                    # Quality score from validation
                    quality_score=quality_score
                )

                qa_items.append(qa_item)
                logger.info(f"Generated cross_year QA for {pair.org1} ({pair.org1_year}-{pair.org2_year}, {connector_name}) [score: {quality_score}/50]")

            except Exception as e:
                logger.error(f"Failed to generate temporal QA for connector {connector_name}: {e}")
                continue

        return qa_items

    def generate_qa_for_intra_doc_pair(
        self,
        pair: CompanyPair,
        max_questions: int = 3,
        connectors: Optional[List[ConnectorResult]] = None
    ) -> List[QAItem]:
        """
        Generate QA items for a Type 3 (intra-document) pair.

        This generates questions about how the same company discusses the same topic
        in different sections/pages of the same 10-K document.

        Args:
            pair: CompanyPair object (same company, same year, different pages/chunks)
            max_questions: Maximum number of questions to generate
            connectors: Optional pre-fetched connectors

        Returns:
            List of QAItem objects
        """
        if pair.pair_category != 'intra_doc':
            logger.warning(f"Pair is not intra_doc: {pair.pair_category}")

        if not pair.is_same_company:
            logger.error(f"Intra-doc pair must be same company: {pair.org1} vs {pair.org2}")
            return []

        if pair.org1_year != pair.org2_year:
            logger.error(f"Intra-doc pair must be same year: {pair.org1_year} vs {pair.org2_year}")
            return []

        # Get intra-doc connectors if not provided
        if connectors is None:
            connectors = self.discovery.find_intra_doc_connectors(
                org=pair.org1,
                year=pair.org1_year,
                connector_types=pair.connector_types
            )

        if not connectors:
            logger.warning(f"No intra-doc connectors found for {pair.org1} ({pair.org1_year})")
            return []

        # Prioritize connectors with relationship changes (asymmetric) - different contexts
        changed_connectors = [c for c in connectors if c.has_asymmetric_relationships]
        stable_connectors = [c for c in connectors if not c.has_asymmetric_relationships]

        # Take mix: prioritize changed (discussed differently), then stable
        selected = changed_connectors[:max_questions]
        if len(selected) < max_questions:
            selected.extend(stable_connectors[:max_questions - len(selected)])

        qa_items = []

        for connector in selected:
            connector_name = connector.connector_name
            connector_type = connector.connector_type
            idf_score = connector.idf_score
            difficulty = self._get_difficulty(idf_score)
            relationship_changed = connector.has_asymmetric_relationships

            # Get evidence for both pages/chunks
            evidence = self.retriever.get_evidence_for_connector(
                connector, pair.org1, pair.org1
            )

            if not evidence:
                logger.debug(f"Skipping connector {connector_name}: no evidence found")
                continue

            evidence1_text = evidence.org1_evidence.text if evidence.org1_evidence else ""
            evidence2_text = evidence.org2_evidence.text if evidence.org2_evidence else ""

            if not evidence1_text or not evidence2_text:
                logger.debug(f"Skipping connector {connector_name}: need evidence from both pages")
                continue

            # Format evidence with source references
            evidence1_formatted = f"[{evidence.org1_evidence.reference}]\n{evidence1_text}"
            evidence2_formatted = f"[{evidence.org2_evidence.reference}]\n{evidence2_text}"

            # Prepare prompt variables
            prompt_vars = {
                'org': pair.org1,
                'year': pair.org1_year,
                'sector': pair.sector or 'Unknown',
                'connector_name': connector_name,
                'connector_type': connector_type,
                'relationship_changed': "Yes - discussed differently across sections" if relationship_changed else "No - discussed consistently",
                'page1': connector.org1_page_id,
                'page2': connector.org2_page_id,
                'org1_relationship': connector.org1_relationship,
                'org2_relationship': connector.org2_relationship,
                'evidence1': evidence1_formatted,
                'evidence2': evidence2_formatted,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{pair.org1}(page:{connector.org1_page_id}) -[{connector.org1_relationship}]-> {connector_name} <-[{connector.org2_relationship}]- {pair.org1}(page:{connector.org2_page_id})"
            entity_chain = f"{pair.org1} (page {connector.org1_page_id}) -> {connector_name} ({connector_type}) -> {pair.org1} (page {connector.org2_page_id})"
            combined_evidence = f"Section 1 (Page {connector.org1_page_id}):\n{evidence1_formatted}\n\nSection 2 (Page {connector.org2_page_id}):\n{evidence2_formatted}"
            source_files = connector.org1_source_file  # Same file for both

            # Generate QA with reflection mechanism
            # Use intra_doc prompts, fallback to cross_year if not available
            prompt_key = 'intra_doc' if 'intra_doc' in PROMPTS else 'cross_year'
            try:
                parsed = self._generate_with_reflection(
                    system_prompt=PROMPTS[prompt_key]['system'],
                    user_prompt_template=PROMPTS[prompt_key]['user'],
                    prompt_vars=prompt_vars,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=combined_evidence,
                    source_files=source_files,
                )

                if not parsed:
                    logger.info(f"Skipping connector {connector_name}: failed validation after retries")
                    continue

                quality_score = parsed.get('quality_score', 0)

                qa_item = QAItem(
                    question=parsed.get('question', ''),
                    answer=parsed.get('answer', ''),
                    reasoning_steps=parsed.get('reasoning_steps', []),
                    difficulty=difficulty,
                    evidence_ids=parsed.get('evidence_ids', []),
                    hop_count=2,
                    hop_pattern='intra_doc',
                    org1=pair.org1,
                    org2=pair.org2,
                    org1_year=pair.org1_year,
                    org2_year=pair.org2_year,
                    sector=pair.sector,
                    sub_industry=pair.sub_industry,
                    connector_name=connector_name,
                    connector_type=connector_type,
                    connector_idf=idf_score,
                    temporal_change=parsed.get('temporal_change', 'N/A'),
                    # Page1 evidence
                    org1_source_file=connector.org1_source_file,
                    org1_page_id=connector.org1_page_id,
                    org1_chunk_id=connector.org1_chunk_id,
                    org1_chunk_text=evidence1_text,
                    org1_relationship=connector.org1_relationship,
                    # Page2 evidence
                    org2_source_file=connector.org2_source_file,
                    org2_page_id=connector.org2_page_id,
                    org2_chunk_id=connector.org2_chunk_id,
                    org2_chunk_text=evidence2_text,
                    org2_relationship=connector.org2_relationship,
                    quality_score=quality_score
                )

                qa_items.append(qa_item)
                logger.info(f"Generated intra-doc QA for {pair.org1} ({pair.org1_year}, {connector_name}, pages {connector.org1_page_id}/{connector.org2_page_id}) [score: {quality_score}/50]")

            except Exception as e:
                logger.error(f"Failed to generate intra-doc QA for connector {connector_name}: {e}")
                continue

        return qa_items

    # =========================================================================
    # 3-HOP GENERATION METHODS
    # =========================================================================

    def generate_3hop_cross_company_type1(
        self,
        pair: CompanyPair,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA using cross_company_type1 pattern.
        Pattern: ORG1 → Connector1 → Connector2 ← ORG2
        """
        # Find 3-hop paths
        three_hop_results = self.discovery.find_3hop_cross_company_type1(
            org1=pair.org1,
            org2=pair.org2,
            year1=pair.org1_year,
            year2=pair.org2_year,
            connector1_types=pair.connector_types,
            connector2_types=pair.connector_types,
            min_idf=self.medium_threshold
        )

        if not three_hop_results:
            logger.warning(f"No 3-hop cross_company_type1 paths found for {pair.org1}-{pair.org2}")
            return []

        qa_items = []
        for result in three_hop_results[:max_questions]:
            difficulty = self._get_difficulty(result.combined_idf_score)

            # Get evidence for all 3 hops
            # get_chunk_text returns ChunkEvidence with .text attribute
            ev1 = self.retriever.get_chunk_text(
                result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
            ) if result.hop1_source_file else None
            ev2 = self.retriever.get_chunk_text(
                result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
            ) if result.hop2_source_file else None
            ev3 = self.retriever.get_chunk_text(
                result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
            ) if result.hop3_source_file else None

            evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
            evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
            evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

            # Prepare prompt variables
            prompt_vars = {
                'org1': pair.org1,
                'org2': pair.org2,
                'year1': pair.org1_year,
                'year2': pair.org2_year,
                'sector': pair.sector or 'Unknown',
                'connector1_name': result.connector1_name,
                'connector1_type': result.connector1_type,
                'connector2_name': result.hop2_entity_name,
                'connector2_type': result.hop2_entity_type,
                'hop1_relationship': result.hop1_relationship,
                'hop2_relationship': result.hop2_relationship,
                'hop3_relationship': result.hop3_relationship,
                'org1_source_file': result.hop1_source_file,
                'org2_source_file': result.hop3_source_file,
                'evidence1': evidence1,
                'evidence2': evidence2,
                'evidence3': evidence3,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{pair.org1} -[{result.hop1_relationship}]-> {result.connector1_name} -[{result.hop2_relationship}]-> {result.hop2_entity_name} <-[{result.hop3_relationship}]- {pair.org2}"
            entity_chain = f"{pair.org1} -> {result.connector1_name} -> {result.hop2_entity_name} -> {pair.org2}"
            combined_evidence = f"Hop 1 ({pair.org1}):\n{evidence1}\n\nHop 2 (link):\n{evidence2}\n\nHop 3 ({pair.org2}):\n{evidence3}"
            source_files = f"{result.hop1_source_file}, {result.hop2_source_file}, {result.hop3_source_file}"

            try:
                parsed = self._generate_with_reflection(
                    system_prompt=PROMPTS['three_hop_cross_company_type1']['system'],
                    user_prompt_template=PROMPTS['three_hop_cross_company_type1']['user'],
                    prompt_vars=prompt_vars,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=combined_evidence,
                    source_files=source_files,
                    # Uses config defaults: max_attempts, quality_threshold
                )

                if not parsed:
                    logger.info(f"Skipping 3-hop cross_company_type1: failed validation after retries")
                    continue

                quality_score = parsed.get('quality_score', 0)

                qa_item = QAItem(
                    question=parsed.get('question', ''),
                    answer=parsed.get('answer', ''),
                    reasoning_steps=parsed.get('reasoning_steps', []),
                    difficulty=difficulty,
                    evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                    hop_count=3,
                    hop_pattern='cross_company_type1',
                    org1=pair.org1,
                    org2=pair.org2,
                    org1_year=pair.org1_year,
                    org2_year=pair.org2_year,
                    sector=pair.sector,
                    sub_industry=pair.sub_industry,
                    connector_name=result.connector1_name,
                    connector_type=result.connector1_type,
                    connector_idf=result.connector1_idf,
                    connector2_name=result.hop2_entity_name,
                    connector2_type=result.hop2_entity_type,
                    connector2_idf=result.hop3_entity_idf,
                    combined_idf_score=result.combined_idf_score,
                    # ORG1 -> Connector1 evidence
                    org1_source_file=result.hop1_source_file,
                    org1_page_id=result.hop1_page_id,
                    org1_chunk_id=result.hop1_chunk_id,
                    org1_chunk_text=ev1.text if ev1 else None,
                    org1_relationship=result.hop1_relationship,
                    # ORG2 -> Connector2 evidence
                    org2_source_file=result.hop3_source_file,
                    org2_page_id=result.hop3_page_id,
                    org2_chunk_id=result.hop3_chunk_id,
                    org2_chunk_text=ev3.text if ev3 else None,
                    org2_relationship=result.hop3_relationship,
                    # Connector1 -> Connector2 evidence (extended)
                    ext_entity_source_file=result.hop2_source_file,
                    ext_entity_page_id=result.hop2_page_id,
                    ext_entity_chunk_id=result.hop2_chunk_id,
                    ext_entity_chunk_text=ev2.text if ev2 else None,
                    ext_entity_relationship=result.hop2_relationship,
                    quality_score=quality_score
                )
                qa_items.append(qa_item)
                logger.info(f"Generated 3-hop cross_company_type1 QA: {pair.org1} → {result.connector1_name} → {result.hop2_entity_name} ← {pair.org2} [score: {quality_score}/50]")

            except Exception as e:
                logger.error(f"Failed to generate 3-hop cross_company_type1 QA: {e}")
                continue

        return qa_items

    def generate_3hop_cross_company_type1_from_result(
        self,
        type1_result: 'ThreeHopResult',
        year: int,
        sector: Optional[str] = None,
        sub_industry: Optional[str] = None,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA from a pre-built ThreeHopResult (connector-first approach).

        This method skips the Neo4j discovery query since the result is already available.
        Used by the connector-first approach in pipeline.py for efficient processing.

        Pattern: ORG1 → Connector1 → Connector2 ← ORG2

        Args:
            type1_result: Pre-built ThreeHopResult from connector-first discovery
            year: Year for the QA
            sector: Sector name (optional)
            sub_industry: Sub-industry name (optional)
            max_questions: Maximum QA items to generate (usually 1 per result)

        Returns:
            List of QAItem objects
        """
        qa_items = []
        result = type1_result

        try:
            difficulty = self._get_difficulty(result.combined_idf_score)

            # Get evidence for all 3 hops
            ev1 = self.retriever.get_chunk_text(
                result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
            ) if result.hop1_source_file else None
            ev2 = self.retriever.get_chunk_text(
                result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
            ) if result.hop2_source_file else None
            ev3 = self.retriever.get_chunk_text(
                result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
            ) if result.hop3_source_file else None

            evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
            evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
            evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

            if evidence1 == "No direct evidence found" and evidence2 == "No direct evidence found" and evidence3 == "No direct evidence found":
                logger.warning(f"Missing evidence for 3-hop cross_company_type1 path (connector-first)")
                return []

            # Prepare prompt variables
            prompt_vars = {
                'org1': result.hop1_org,
                'org2': result.hop3_org,
                'year1': year,
                'year2': year,
                'sector': sector or 'Unknown',
                'connector1_name': result.connector1_name,
                'connector1_type': result.connector1_type,
                'connector2_name': result.hop2_entity_name,
                'connector2_type': result.hop2_entity_type,
                'hop1_relationship': result.hop1_relationship,
                'hop2_relationship': result.hop2_relationship,
                'hop3_relationship': result.hop3_relationship,
                'org1_source_file': result.hop1_source_file,
                'org2_source_file': result.hop3_source_file,
                'evidence1': evidence1,
                'evidence2': evidence2,
                'evidence3': evidence3,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{result.hop1_org} -[{result.hop1_relationship}]-> {result.connector1_name} -[{result.hop2_relationship}]-> {result.hop2_entity_name} <-[{result.hop3_relationship}]- {result.hop3_org}"
            entity_chain = f"{result.hop1_org} -> {result.connector1_name} -> {result.hop2_entity_name} -> {result.hop3_org}"
            combined_evidence = f"Hop 1 ({result.hop1_org}):\n{evidence1}\n\nHop 2 (link):\n{evidence2}\n\nHop 3 ({result.hop3_org}):\n{evidence3}"
            source_files = f"{result.hop1_source_file}, {result.hop2_source_file}, {result.hop3_source_file}"

            parsed = self._generate_with_reflection(
                system_prompt=PROMPTS['three_hop_cross_company_type1']['system'],
                user_prompt_template=PROMPTS['three_hop_cross_company_type1']['user'],
                prompt_vars=prompt_vars,
                pattern=pattern,
                entity_chain=entity_chain,
                evidence=combined_evidence,
                source_files=source_files,
            )

            if not parsed:
                logger.info(f"Skipping 3-hop cross_company_type1: failed validation after retries")
                return []

            quality_score = parsed.get('quality_score', 0)

            qa_item = QAItem(
                question=parsed.get('question', ''),
                answer=parsed.get('answer', ''),
                reasoning_steps=parsed.get('reasoning_steps', []),
                difficulty=difficulty,
                evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                hop_count=3,
                hop_pattern='cross_company_type1',
                org1=result.hop1_org,
                org2=result.hop3_org,
                org1_year=year,
                org2_year=year,
                sector=sector,
                sub_industry=sub_industry,
                connector_name=result.connector1_name,
                connector_type=result.connector1_type,
                connector_idf=result.connector1_idf,
                connector2_name=result.hop2_entity_name,
                connector2_type=result.hop2_entity_type,
                connector2_idf=result.hop3_entity_idf,
                combined_idf_score=result.combined_idf_score,
                # ORG1 -> Connector1 evidence
                org1_source_file=result.hop1_source_file,
                org1_page_id=result.hop1_page_id,
                org1_chunk_id=result.hop1_chunk_id,
                org1_chunk_text=ev1.text if ev1 else None,
                org1_relationship=result.hop1_relationship,
                # ORG2 -> Connector2 evidence
                org2_source_file=result.hop3_source_file,
                org2_page_id=result.hop3_page_id,
                org2_chunk_id=result.hop3_chunk_id,
                org2_chunk_text=ev3.text if ev3 else None,
                org2_relationship=result.hop3_relationship,
                # Connector1 -> Connector2 evidence (extended)
                ext_entity_source_file=result.hop2_source_file,
                ext_entity_page_id=result.hop2_page_id,
                ext_entity_chunk_id=result.hop2_chunk_id,
                ext_entity_chunk_text=ev2.text if ev2 else None,
                ext_entity_relationship=result.hop2_relationship,
                quality_score=quality_score
            )
            qa_items.append(qa_item)
            logger.info(f"Generated 3-hop cross_company_type1 QA: {result.hop1_org} → {result.connector1_name} → {result.hop2_entity_name} ← {result.hop3_org} [score: {quality_score}/50]")

        except Exception as e:
            logger.error(f"Failed to generate 3-hop cross_company_type1 QA (connector-first): {e}")

        return qa_items

    def generate_3hop_cross_year_type1(
        self,
        pair: CompanyPair,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA using cross_year_type1 pattern.
        Pattern: ORG(y1) → Connector1 → Connector2 ← ORG(y2)
        Same company, different years, connector chain.
        """
        # Find 3-hop paths
        three_hop_results = self.discovery.find_3hop_cross_year_type1(
            org=pair.org1,  # Same company
            year1=pair.org1_year,
            year2=pair.org2_year,
            connector1_types=pair.connector_types,
            connector2_types=pair.connector_types,
            min_idf=self.medium_threshold
        )

        if not three_hop_results:
            logger.warning(f"No 3-hop cross_year_type1 paths found for {pair.org1} ({pair.org1_year}-{pair.org2_year})")
            return []

        qa_items = []
        for result in three_hop_results[:max_questions]:
            difficulty = self._get_difficulty(result.combined_idf_score)

            # Get evidence for all 3 hops
            ev1 = self.retriever.get_chunk_text(
                result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
            ) if result.hop1_source_file else None
            ev2 = self.retriever.get_chunk_text(
                result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
            ) if result.hop2_source_file else None
            ev3 = self.retriever.get_chunk_text(
                result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
            ) if result.hop3_source_file else None

            evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
            evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
            evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

            # Prepare prompt variables
            prompt_vars = {
                'org': pair.org1,
                'year1': pair.org1_year,
                'year2': pair.org2_year,
                'sector': pair.sector or 'Unknown',
                'connector1_name': result.connector1_name,
                'connector1_type': result.connector1_type,
                'connector2_name': result.hop2_entity_name,
                'connector2_type': result.hop2_entity_type,
                'hop1_relationship': result.hop1_relationship,
                'hop2_relationship': result.hop2_relationship,
                'hop3_relationship': result.hop3_relationship,
                'org1_source_file': result.hop1_source_file,
                'org2_source_file': result.hop3_source_file,
                'evidence1': evidence1,
                'evidence2': evidence2,
                'evidence3': evidence3,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{pair.org1}({pair.org1_year}) -[{result.hop1_relationship}]-> {result.connector1_name} -[{result.hop2_relationship}]-> {result.hop2_entity_name} <-[{result.hop3_relationship}]- {pair.org1}({pair.org2_year})"
            entity_chain = f"{pair.org1}({pair.org1_year}) -> {result.connector1_name} -> {result.hop2_entity_name} -> {pair.org1}({pair.org2_year})"
            combined_evidence = f"Hop 1 ({pair.org1} {pair.org1_year}):\n{evidence1}\n\nHop 2 (link):\n{evidence2}\n\nHop 3 ({pair.org1} {pair.org2_year}):\n{evidence3}"
            source_files = f"{result.hop1_source_file}, {result.hop2_source_file}, {result.hop3_source_file}"

            try:
                # Use cross_company_type1 prompt for now (similar structure)
                parsed = self._generate_with_reflection(
                    system_prompt=PROMPTS.get('three_hop_cross_year_type1', PROMPTS['three_hop_cross_company_type1'])['system'],
                    user_prompt_template=PROMPTS.get('three_hop_cross_year_type1', PROMPTS['three_hop_cross_company_type1'])['user'],
                    prompt_vars=prompt_vars,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=combined_evidence,
                    source_files=source_files,
                )

                if not parsed:
                    logger.info(f"Skipping 3-hop cross_year_type1: failed validation after retries")
                    continue

                quality_score = parsed.get('quality_score', 0)

                qa_item = QAItem(
                    question=parsed.get('question', ''),
                    answer=parsed.get('answer', ''),
                    reasoning_steps=parsed.get('reasoning_steps', []),
                    difficulty=difficulty,
                    evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                    hop_count=3,
                    hop_pattern='cross_year_type1',
                    org1=pair.org1,
                    org2=pair.org1,  # Same company
                    org1_year=pair.org1_year,
                    org2_year=pair.org2_year,
                    sector=pair.sector,
                    sub_industry=pair.sub_industry,
                    connector_name=result.connector1_name,
                    connector_type=result.connector1_type,
                    connector_idf=result.connector1_idf,
                    connector2_name=result.hop2_entity_name,
                    connector2_type=result.hop2_entity_type,
                    connector2_idf=result.hop3_entity_idf,
                    combined_idf_score=result.combined_idf_score,
                    org1_source_file=result.hop1_source_file,
                    org1_page_id=result.hop1_page_id,
                    org1_chunk_id=result.hop1_chunk_id,
                    org1_chunk_text=ev1.text if ev1 else None,
                    org1_relationship=result.hop1_relationship,
                    org2_source_file=result.hop3_source_file,
                    org2_page_id=result.hop3_page_id,
                    org2_chunk_id=result.hop3_chunk_id,
                    org2_chunk_text=ev3.text if ev3 else None,
                    org2_relationship=result.hop3_relationship,
                    ext_entity_source_file=result.hop2_source_file,
                    ext_entity_page_id=result.hop2_page_id,
                    ext_entity_chunk_id=result.hop2_chunk_id,
                    ext_entity_chunk_text=ev2.text if ev2 else None,
                    ext_entity_relationship=result.hop2_relationship,
                    quality_score=quality_score
                )
                qa_items.append(qa_item)
                logger.info(f"Generated 3-hop cross_year_type1 QA: {pair.org1}({pair.org1_year}) → {result.connector1_name} → {result.hop2_entity_name} ← {pair.org1}({pair.org2_year}) [score: {quality_score}/50]")

            except Exception as e:
                logger.error(f"Failed to generate 3-hop cross_year_type1 QA: {e}")
                continue

        return qa_items

    def generate_3hop_cross_year_type1_from_result(
        self,
        type1_result: 'ThreeHopResult',
        org: str,
        year1: int,
        year2: int,
        sector: Optional[str] = None,
        sub_industry: Optional[str] = None,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA from a pre-built ThreeHopResult for cross_year_type1 (connector-first).

        Pattern: ORG(y1) → Connector1 → Connector2 ← ORG(y2)
        Same company, different years, connector chain.
        """
        qa_items = []
        result = type1_result

        try:
            difficulty = self._get_difficulty(result.combined_idf_score)

            # Get evidence for all 3 hops
            ev1 = self.retriever.get_chunk_text(
                result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
            ) if result.hop1_source_file else None
            ev2 = self.retriever.get_chunk_text(
                result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
            ) if result.hop2_source_file else None
            ev3 = self.retriever.get_chunk_text(
                result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
            ) if result.hop3_source_file else None

            evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
            evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
            evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

            if evidence1 == "No direct evidence found" and evidence2 == "No direct evidence found" and evidence3 == "No direct evidence found":
                logger.warning(f"Missing evidence for 3-hop cross_year_type1 path (connector-first)")
                return []

            # Prepare prompt variables
            prompt_vars = {
                'org': org,
                'year1': year1,
                'year2': year2,
                'sector': sector or 'Unknown',
                'connector1_name': result.connector1_name,
                'connector1_type': result.connector1_type,
                'connector2_name': result.hop2_entity_name,
                'connector2_type': result.hop2_entity_type,
                'hop1_relationship': result.hop1_relationship,
                'hop2_relationship': result.hop2_relationship,
                'hop3_relationship': result.hop3_relationship,
                'org1_source_file': result.hop1_source_file,
                'org2_source_file': result.hop3_source_file,
                'evidence1': evidence1,
                'evidence2': evidence2,
                'evidence3': evidence3,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{org}({year1}) -[{result.hop1_relationship}]-> {result.connector1_name} -[{result.hop2_relationship}]-> {result.hop2_entity_name} <-[{result.hop3_relationship}]- {org}({year2})"
            entity_chain = f"{org}({year1}) -> {result.connector1_name} -> {result.hop2_entity_name} -> {org}({year2})"
            combined_evidence = f"Hop 1 ({org} {year1}):\n{evidence1}\n\nHop 2 (link):\n{evidence2}\n\nHop 3 ({org} {year2}):\n{evidence3}"
            source_files = f"{result.hop1_source_file}, {result.hop2_source_file}, {result.hop3_source_file}"

            parsed = self._generate_with_reflection(
                system_prompt=PROMPTS.get('three_hop_cross_year_type1', PROMPTS['three_hop_cross_company_type1'])['system'],
                user_prompt_template=PROMPTS.get('three_hop_cross_year_type1', PROMPTS['three_hop_cross_company_type1'])['user'],
                prompt_vars=prompt_vars,
                pattern=pattern,
                entity_chain=entity_chain,
                evidence=combined_evidence,
                source_files=source_files,
            )

            if not parsed:
                logger.info(f"Skipping 3-hop cross_year_type1: failed validation after retries")
                return []

            quality_score = parsed.get('quality_score', 0)

            qa_item = QAItem(
                question=parsed.get('question', ''),
                answer=parsed.get('answer', ''),
                reasoning_steps=parsed.get('reasoning_steps', []),
                difficulty=difficulty,
                evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                hop_count=3,
                hop_pattern='cross_year_type1',
                org1=org,
                org2=org,  # Same company
                org1_year=year1,
                org2_year=year2,
                sector=sector,
                sub_industry=sub_industry,
                connector_name=result.connector1_name,
                connector_type=result.connector1_type,
                connector_idf=result.connector1_idf,
                connector2_name=result.hop2_entity_name,
                connector2_type=result.hop2_entity_type,
                connector2_idf=result.hop3_entity_idf,
                combined_idf_score=result.combined_idf_score,
                org1_source_file=result.hop1_source_file,
                org1_page_id=result.hop1_page_id,
                org1_chunk_id=result.hop1_chunk_id,
                org1_chunk_text=ev1.text if ev1 else None,
                org1_relationship=result.hop1_relationship,
                org2_source_file=result.hop3_source_file,
                org2_page_id=result.hop3_page_id,
                org2_chunk_id=result.hop3_chunk_id,
                org2_chunk_text=ev3.text if ev3 else None,
                org2_relationship=result.hop3_relationship,
                ext_entity_source_file=result.hop2_source_file,
                ext_entity_page_id=result.hop2_page_id,
                ext_entity_chunk_id=result.hop2_chunk_id,
                ext_entity_chunk_text=ev2.text if ev2 else None,
                ext_entity_relationship=result.hop2_relationship,
                quality_score=quality_score
            )
            qa_items.append(qa_item)
            logger.info(f"Generated 3-hop cross_year_type1 QA: {org}({year1}) → {result.connector1_name} → {result.hop2_entity_name} ← {org}({year2}) [score: {quality_score}/50]")

        except Exception as e:
            logger.error(f"Failed to generate 3-hop cross_year_type1 QA (connector-first): {e}")

        return qa_items

    def generate_3hop_intra_doc_type1(
        self,
        pair: CompanyPair,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA using intra_doc_type1 pattern.
        Pattern: ORG(p1) → Connector1 → Connector2 ← ORG(p2)
        Same company, same year, different pages, connector chain.
        """
        # Find 3-hop paths
        three_hop_results = self.discovery.find_3hop_intra_doc_type1(
            org=pair.org1,  # Same company
            year=pair.org1_year,
            connector1_types=pair.connector_types,
            connector2_types=pair.connector_types,
            min_idf=self.medium_threshold
        )

        if not three_hop_results:
            logger.warning(f"No 3-hop intra_doc_type1 paths found for {pair.org1} ({pair.org1_year})")
            return []

        qa_items = []
        for result in three_hop_results[:max_questions]:
            difficulty = self._get_difficulty(result.combined_idf_score)

            # Get evidence for all 3 hops
            ev1 = self.retriever.get_chunk_text(
                result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
            ) if result.hop1_source_file else None
            ev2 = self.retriever.get_chunk_text(
                result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
            ) if result.hop2_source_file else None
            ev3 = self.retriever.get_chunk_text(
                result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
            ) if result.hop3_source_file else None

            evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
            evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
            evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

            # Prepare prompt variables
            prompt_vars = {
                'org': pair.org1,
                'year': pair.org1_year,
                'sector': pair.sector or 'Unknown',
                'page1': result.hop1_page_id,
                'page2': result.hop3_page_id,
                'connector1_name': result.connector1_name,
                'connector1_type': result.connector1_type,
                'connector2_name': result.hop2_entity_name,
                'connector2_type': result.hop2_entity_type,
                'hop1_relationship': result.hop1_relationship,
                'hop2_relationship': result.hop2_relationship,
                'hop3_relationship': result.hop3_relationship,
                'evidence1': evidence1,
                'evidence2': evidence2,
                'evidence3': evidence3,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{pair.org1}(page {result.hop1_page_id}) -[{result.hop1_relationship}]-> {result.connector1_name} -[{result.hop2_relationship}]-> {result.hop2_entity_name} <-[{result.hop3_relationship}]- {pair.org1}(page {result.hop3_page_id})"
            entity_chain = f"{pair.org1}(p{result.hop1_page_id}) -> {result.connector1_name} -> {result.hop2_entity_name} -> {pair.org1}(p{result.hop3_page_id})"
            combined_evidence = f"Hop 1 ({pair.org1} page {result.hop1_page_id}):\n{evidence1}\n\nHop 2 (link):\n{evidence2}\n\nHop 3 ({pair.org1} page {result.hop3_page_id}):\n{evidence3}"
            source_files = f"{result.hop1_source_file}"

            try:
                # Use cross_company_type1 prompt for now (similar structure)
                parsed = self._generate_with_reflection(
                    system_prompt=PROMPTS.get('three_hop_intra_doc_type1', PROMPTS['three_hop_cross_company_type1'])['system'],
                    user_prompt_template=PROMPTS.get('three_hop_intra_doc_type1', PROMPTS['three_hop_cross_company_type1'])['user'],
                    prompt_vars=prompt_vars,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=combined_evidence,
                    source_files=source_files,
                )

                if not parsed:
                    logger.info(f"Skipping 3-hop intra_doc_type1: failed validation after retries")
                    continue

                quality_score = parsed.get('quality_score', 0)

                qa_item = QAItem(
                    question=parsed.get('question', ''),
                    answer=parsed.get('answer', ''),
                    reasoning_steps=parsed.get('reasoning_steps', []),
                    difficulty=difficulty,
                    evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                    hop_count=3,
                    hop_pattern='intra_doc_type1',
                    org1=pair.org1,
                    org2=pair.org1,  # Same company
                    org1_year=pair.org1_year,
                    org2_year=pair.org1_year,  # Same year
                    sector=pair.sector,
                    sub_industry=pair.sub_industry,
                    connector_name=result.connector1_name,
                    connector_type=result.connector1_type,
                    connector_idf=result.connector1_idf,
                    connector2_name=result.hop2_entity_name,
                    connector2_type=result.hop2_entity_type,
                    connector2_idf=result.hop3_entity_idf,
                    combined_idf_score=result.combined_idf_score,
                    org1_source_file=result.hop1_source_file,
                    org1_page_id=result.hop1_page_id,
                    org1_chunk_id=result.hop1_chunk_id,
                    org1_chunk_text=ev1.text if ev1 else None,
                    org1_relationship=result.hop1_relationship,
                    org2_source_file=result.hop3_source_file,
                    org2_page_id=result.hop3_page_id,
                    org2_chunk_id=result.hop3_chunk_id,
                    org2_chunk_text=ev3.text if ev3 else None,
                    org2_relationship=result.hop3_relationship,
                    ext_entity_source_file=result.hop2_source_file,
                    ext_entity_page_id=result.hop2_page_id,
                    ext_entity_chunk_id=result.hop2_chunk_id,
                    ext_entity_chunk_text=ev2.text if ev2 else None,
                    ext_entity_relationship=result.hop2_relationship,
                    quality_score=quality_score
                )
                qa_items.append(qa_item)
                logger.info(f"Generated 3-hop intra_doc_type1 QA: {pair.org1}(p{result.hop1_page_id}) → {result.connector1_name} → {result.hop2_entity_name} ← {pair.org1}(p{result.hop3_page_id}) [score: {quality_score}/50]")

            except Exception as e:
                logger.error(f"Failed to generate 3-hop intra_doc_type1 QA: {e}")
                continue

        return qa_items

    def generate_3hop_intra_doc_type1_from_result(
        self,
        type1_result: 'ThreeHopResult',
        org: str,
        year: int,
        sector: Optional[str] = None,
        sub_industry: Optional[str] = None,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA from a pre-built ThreeHopResult for intra_doc_type1 (connector-first).

        Pattern: ORG(p1) → Connector1 → Connector2 ← ORG(p2)
        Same company, same year, different pages, connector chain.
        """
        qa_items = []
        result = type1_result

        try:
            difficulty = self._get_difficulty(result.combined_idf_score)

            # Get evidence for all 3 hops
            ev1 = self.retriever.get_chunk_text(
                result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
            ) if result.hop1_source_file else None
            ev2 = self.retriever.get_chunk_text(
                result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
            ) if result.hop2_source_file else None
            ev3 = self.retriever.get_chunk_text(
                result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
            ) if result.hop3_source_file else None

            evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
            evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
            evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

            if evidence1 == "No direct evidence found" and evidence2 == "No direct evidence found" and evidence3 == "No direct evidence found":
                logger.warning(f"Missing evidence for 3-hop intra_doc_type1 path (connector-first)")
                return []

            # Prepare prompt variables
            prompt_vars = {
                'org': org,
                'year': year,
                'sector': sector or 'Unknown',
                'page1': result.hop1_page_id,
                'page2': result.hop3_page_id,
                'connector1_name': result.connector1_name,
                'connector1_type': result.connector1_type,
                'connector2_name': result.hop2_entity_name,
                'connector2_type': result.hop2_entity_type,
                'hop1_relationship': result.hop1_relationship,
                'hop2_relationship': result.hop2_relationship,
                'hop3_relationship': result.hop3_relationship,
                'evidence1': evidence1,
                'evidence2': evidence2,
                'evidence3': evidence3,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{org}(page {result.hop1_page_id}) -[{result.hop1_relationship}]-> {result.connector1_name} -[{result.hop2_relationship}]-> {result.hop2_entity_name} <-[{result.hop3_relationship}]- {org}(page {result.hop3_page_id})"
            entity_chain = f"{org}(p{result.hop1_page_id}) -> {result.connector1_name} -> {result.hop2_entity_name} -> {org}(p{result.hop3_page_id})"
            combined_evidence = f"Hop 1 ({org} page {result.hop1_page_id}):\n{evidence1}\n\nHop 2 (link):\n{evidence2}\n\nHop 3 ({org} page {result.hop3_page_id}):\n{evidence3}"
            source_files = f"{result.hop1_source_file}"

            parsed = self._generate_with_reflection(
                system_prompt=PROMPTS.get('three_hop_intra_doc_type1', PROMPTS['three_hop_cross_company_type1'])['system'],
                user_prompt_template=PROMPTS.get('three_hop_intra_doc_type1', PROMPTS['three_hop_cross_company_type1'])['user'],
                prompt_vars=prompt_vars,
                pattern=pattern,
                entity_chain=entity_chain,
                evidence=combined_evidence,
                source_files=source_files,
            )

            if not parsed:
                logger.info(f"Skipping 3-hop intra_doc_type1: failed validation after retries")
                return []

            quality_score = parsed.get('quality_score', 0)

            qa_item = QAItem(
                question=parsed.get('question', ''),
                answer=parsed.get('answer', ''),
                reasoning_steps=parsed.get('reasoning_steps', []),
                difficulty=difficulty,
                evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                hop_count=3,
                hop_pattern='intra_doc_type1',
                org1=org,
                org2=org,  # Same company
                org1_year=year,
                org2_year=year,  # Same year
                sector=sector,
                sub_industry=sub_industry,
                connector_name=result.connector1_name,
                connector_type=result.connector1_type,
                connector_idf=result.connector1_idf,
                connector2_name=result.hop2_entity_name,
                connector2_type=result.hop2_entity_type,
                connector2_idf=result.hop3_entity_idf,
                combined_idf_score=result.combined_idf_score,
                org1_source_file=result.hop1_source_file,
                org1_page_id=result.hop1_page_id,
                org1_chunk_id=result.hop1_chunk_id,
                org1_chunk_text=ev1.text if ev1 else None,
                org1_relationship=result.hop1_relationship,
                org2_source_file=result.hop3_source_file,
                org2_page_id=result.hop3_page_id,
                org2_chunk_id=result.hop3_chunk_id,
                org2_chunk_text=ev3.text if ev3 else None,
                org2_relationship=result.hop3_relationship,
                ext_entity_source_file=result.hop2_source_file,
                ext_entity_page_id=result.hop2_page_id,
                ext_entity_chunk_id=result.hop2_chunk_id,
                ext_entity_chunk_text=ev2.text if ev2 else None,
                ext_entity_relationship=result.hop2_relationship,
                quality_score=quality_score
            )
            qa_items.append(qa_item)
            logger.info(f"Generated 3-hop intra_doc_type1 QA: {org}(p{result.hop1_page_id}) → {result.connector1_name} → {result.hop2_entity_name} ← {org}(p{result.hop3_page_id}) [score: {quality_score}/50]")

        except Exception as e:
            logger.error(f"Failed to generate 3-hop intra_doc_type1 QA (connector-first): {e}")

        return qa_items

    # =========================================================================
    # TYPE 2 PATTERNS: Multi-Branch (3 anchors → 1 connector)
    # =========================================================================

    def generate_3hop_cross_company_type2(
        self,
        org1: str,
        org2: str,
        org3: str,
        year: int,
        sector: Optional[str] = None,
        connector_types: Optional[List[str]] = None,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA using cross_company_type2 pattern.
        Pattern: ORG1 → Connector ← ORG2, Connector ← ORG3
        3 different companies, same year, 1 shared connector.
        """
        type2_results = self.discovery.find_3hop_cross_company_type2(
            org1=org1,
            org2=org2,
            org3=org3,
            year=year,
            connector_types=connector_types,
            min_idf=self.medium_threshold
        )

        if not type2_results:
            logger.warning(f"No 3-hop cross_company_type2 paths found for {org1}, {org2}, {org3} ({year})")
            return []

        qa_items = []

        for result in type2_results[:max_questions]:
            try:
                # Retrieve evidence for all 3 hops using get_chunk_text
                ev1 = self.retriever.get_chunk_text(
                    result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
                ) if result.hop1_source_file else None
                ev2 = self.retriever.get_chunk_text(
                    result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
                ) if result.hop2_source_file else None
                ev3 = self.retriever.get_chunk_text(
                    result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
                ) if result.hop3_source_file else None

                evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
                evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
                evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

                if evidence1 == "No direct evidence found" and evidence2 == "No direct evidence found" and evidence3 == "No direct evidence found":
                    logger.warning(f"Missing evidence for 3-hop cross_company_type2 path")
                    continue

                difficulty = self._get_difficulty(result.connector_idf)

                # Build prompt vars like Type 1
                prompt_vars = {
                    'org1': result.hop1_org,
                    'org2': result.hop2_org,
                    'org3': result.hop3_org,
                    'year': year,
                    'sector': sector or 'Unknown',
                    'connector_name': result.connector_name,
                    'connector_type': result.connector_type,
                    'idf_score': result.connector_idf,
                    'hop1_relationship': result.hop1_relationship,
                    'hop2_relationship': result.hop2_relationship,
                    'hop3_relationship': result.hop3_relationship,
                    'org1_source_file': result.hop1_source_file,
                    'org2_source_file': result.hop2_source_file,
                    'org3_source_file': result.hop3_source_file,
                    'evidence1': evidence1,
                    'evidence2': evidence2,
                    'evidence3': evidence3,
                    'difficulty': difficulty
                }

                # Prepare validation context like Type 1
                pattern = f"{result.hop1_org} -[{result.hop1_relationship}]-> {result.connector_name} <-[{result.hop2_relationship}]- {result.hop2_org}, {result.connector_name} <-[{result.hop3_relationship}]- {result.hop3_org}"
                entity_chain = f"{result.hop1_org}, {result.hop2_org}, {result.hop3_org} -> {result.connector_name}"
                combined_evidence = f"Hop 1 ({result.hop1_org}):\n{evidence1}\n\nHop 2 ({result.hop2_org}):\n{evidence2}\n\nHop 3 ({result.hop3_org}):\n{evidence3}"
                source_files = f"{result.hop1_source_file}, {result.hop2_source_file}, {result.hop3_source_file}"

                # Try to get Type 2 specific prompt, fallback to cross_company_type1
                prompt_key = 'three_hop_cross_company_type2'
                if prompt_key not in PROMPTS:
                    prompt_key = 'three_hop_cross_company_type1'

                parsed = self._generate_with_reflection(
                    system_prompt=PROMPTS[prompt_key]['system'],
                    user_prompt_template=PROMPTS[prompt_key]['user'],
                    prompt_vars=prompt_vars,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=combined_evidence,
                    source_files=source_files,
                )

                if not parsed:
                    logger.info(f"Skipping 3-hop cross_company_type2: failed validation after retries")
                    continue

                quality_score = parsed.get('quality_score', 0)

                # Create QAItem - Type 2: 3 ORGs → 1 Connector
                qa_item = QAItem(
                    question=parsed.get('question', ''),
                    answer=parsed.get('answer', ''),
                    reasoning_steps=parsed.get('reasoning_steps', []),
                    difficulty=difficulty,
                    evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                    hop_count=3,
                    hop_pattern='cross_company_type2',
                    org1=result.hop1_org,
                    org2=result.hop2_org,
                    org1_year=year,
                    org2_year=year,
                    sector=sector or 'Unknown',
                    connector_name=result.connector_name,
                    connector_type=result.connector_type,
                    connector_idf=result.connector_idf,
                    # Store org3 info in connector2 fields (repurposed for Type 2)
                    connector2_name=result.hop3_org,
                    connector2_type='ORG',
                    connector2_idf=result.connector_idf,
                    # ORG1 -> Connector evidence
                    org1_source_file=result.hop1_source_file,
                    org1_page_id=result.hop1_page_id,
                    org1_chunk_id=result.hop1_chunk_id,
                    org1_chunk_text=ev1.text if ev1 else None,
                    org1_relationship=result.hop1_relationship,
                    # ORG2 -> Connector evidence
                    org2_source_file=result.hop2_source_file,
                    org2_page_id=result.hop2_page_id,
                    org2_chunk_id=result.hop2_chunk_id,
                    org2_chunk_text=ev2.text if ev2 else None,
                    org2_relationship=result.hop2_relationship,
                    # ORG3 -> Connector evidence (using ext_entity fields)
                    ext_entity_source_file=result.hop3_source_file,
                    ext_entity_page_id=result.hop3_page_id,
                    ext_entity_chunk_id=result.hop3_chunk_id,
                    ext_entity_chunk_text=ev3.text if ev3 else None,
                    ext_entity_relationship=result.hop3_relationship,
                    quality_score=quality_score
                )

                qa_items.append(qa_item)
                logger.info(f"Generated 3-hop cross_company_type2 QA: {result.hop1_org}, {result.hop2_org}, {result.hop3_org} → {result.connector_name} [score: {quality_score}/50]")

            except Exception as e:
                logger.error(f"Failed to generate 3-hop cross_company_type2 QA: {e}")
                continue

        return qa_items

    def generate_3hop_cross_company_type2_from_result(
        self,
        type2_result: 'ThreeHopType2Result',
        year: int,
        sector: Optional[str] = None,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA from a pre-built ThreeHopType2Result (connector-first approach).

        This method skips the Neo4j discovery query since the result is already available.
        Used by the connector-first approach in pipeline.py for efficient processing.

        Args:
            type2_result: Pre-built ThreeHopType2Result from connector-first discovery
            year: Year for the QA
            sector: Sector name (optional)
            max_questions: Maximum QA items to generate (usually 1 per result)

        Returns:
            List of QAItem objects
        """
        qa_items = []
        result = type2_result

        try:
            # Retrieve evidence for all 3 hops using get_chunk_text
            ev1 = self.retriever.get_chunk_text(
                result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
            ) if result.hop1_source_file else None
            ev2 = self.retriever.get_chunk_text(
                result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
            ) if result.hop2_source_file else None
            ev3 = self.retriever.get_chunk_text(
                result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
            ) if result.hop3_source_file else None

            evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
            evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
            evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

            if evidence1 == "No direct evidence found" and evidence2 == "No direct evidence found" and evidence3 == "No direct evidence found":
                logger.warning(f"Missing evidence for 3-hop cross_company_type2 path (connector-first)")
                return []

            difficulty = self._get_difficulty(result.connector_idf)

            # Build prompt vars
            prompt_vars = {
                'org1': result.hop1_org,
                'org2': result.hop2_org,
                'org3': result.hop3_org,
                'year': year,
                'sector': sector or 'Unknown',
                'connector_name': result.connector_name,
                'connector_type': result.connector_type,
                'idf_score': result.connector_idf,
                'hop1_relationship': result.hop1_relationship,
                'hop2_relationship': result.hop2_relationship,
                'hop3_relationship': result.hop3_relationship,
                'org1_source_file': result.hop1_source_file,
                'org2_source_file': result.hop2_source_file,
                'org3_source_file': result.hop3_source_file,
                'evidence1': evidence1,
                'evidence2': evidence2,
                'evidence3': evidence3,
                'difficulty': difficulty
            }

            # Prepare validation context
            pattern = f"{result.hop1_org} -[{result.hop1_relationship}]-> {result.connector_name} <-[{result.hop2_relationship}]- {result.hop2_org}, {result.connector_name} <-[{result.hop3_relationship}]- {result.hop3_org}"
            entity_chain = f"{result.hop1_org}, {result.hop2_org}, {result.hop3_org} -> {result.connector_name}"
            combined_evidence = f"Hop 1 ({result.hop1_org}):\n{evidence1}\n\nHop 2 ({result.hop2_org}):\n{evidence2}\n\nHop 3 ({result.hop3_org}):\n{evidence3}"
            source_files = f"{result.hop1_source_file}, {result.hop2_source_file}, {result.hop3_source_file}"

            # Try to get Type 2 specific prompt, fallback to cross_company_type1
            prompt_key = 'three_hop_cross_company_type2'
            if prompt_key not in PROMPTS:
                prompt_key = 'three_hop_cross_company_type1'

            parsed = self._generate_with_reflection(
                system_prompt=PROMPTS[prompt_key]['system'],
                user_prompt_template=PROMPTS[prompt_key]['user'],
                prompt_vars=prompt_vars,
                pattern=pattern,
                entity_chain=entity_chain,
                evidence=combined_evidence,
                source_files=source_files,
            )

            if not parsed:
                logger.info(f"Skipping 3-hop cross_company_type2: failed validation after retries")
                return []

            quality_score = parsed.get('quality_score', 0)

            # Create QAItem - Type 2: 3 ORGs → 1 Connector
            qa_item = QAItem(
                question=parsed.get('question', ''),
                answer=parsed.get('answer', ''),
                reasoning_steps=parsed.get('reasoning_steps', []),
                difficulty=difficulty,
                evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                hop_count=3,
                hop_pattern='cross_company_type2',
                org1=result.hop1_org,
                org2=result.hop2_org,
                org1_year=year,
                org2_year=year,
                sector=sector or 'Unknown',
                connector_name=result.connector_name,
                connector_type=result.connector_type,
                connector_idf=result.connector_idf,
                # Store org3 info in connector2 fields (repurposed for Type 2)
                connector2_name=result.hop3_org,
                connector2_type='ORG',
                connector2_idf=result.connector_idf,
                # ORG1 -> Connector evidence
                org1_source_file=result.hop1_source_file,
                org1_page_id=result.hop1_page_id,
                org1_chunk_id=result.hop1_chunk_id,
                org1_chunk_text=ev1.text if ev1 else None,
                org1_relationship=result.hop1_relationship,
                # ORG2 -> Connector evidence
                org2_source_file=result.hop2_source_file,
                org2_page_id=result.hop2_page_id,
                org2_chunk_id=result.hop2_chunk_id,
                org2_chunk_text=ev2.text if ev2 else None,
                org2_relationship=result.hop2_relationship,
                # ORG3 -> Connector evidence (using ext_entity fields)
                ext_entity_source_file=result.hop3_source_file,
                ext_entity_page_id=result.hop3_page_id,
                ext_entity_chunk_id=result.hop3_chunk_id,
                ext_entity_chunk_text=ev3.text if ev3 else None,
                ext_entity_relationship=result.hop3_relationship,
                quality_score=quality_score
            )

            qa_items.append(qa_item)
            logger.info(f"Generated 3-hop cross_company_type2 QA: {result.hop1_org}, {result.hop2_org}, {result.hop3_org} → {result.connector_name} [score: {quality_score}/50]")

        except Exception as e:
            logger.error(f"Failed to generate 3-hop cross_company_type2 QA (connector-first): {e}")

        return qa_items

    def generate_3hop_cross_year_type2(
        self,
        pair: CompanyPair,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA using cross_year_type2 pattern.
        Pattern: ORG(y1) → Connector ← ORG(y2), Connector ← ORG(y3)
        Same company, 3 different years, 1 shared connector.
        """
        # For Type 2 cross_year, we need 3 years. Use available years from config.
        available_years = [2022, 2023, 2024]
        year1 = pair.org1_year
        year2 = pair.org2_year
        # Find third year that's different from both
        year3 = next((y for y in available_years if y != year1 and y != year2), None)

        if year3 is None:
            logger.warning(f"Cannot find 3rd year for cross_year_type2 pattern")
            return []

        type2_results = self.discovery.find_3hop_cross_year_type2(
            org=pair.org1,
            year1=year1,
            year2=year2,
            year3=year3,
            connector_types=pair.connector_types,
            min_idf=self.medium_threshold
        )

        if not type2_results:
            logger.warning(f"No 3-hop cross_year_type2 paths found for {pair.org1} ({year1}, {year2}, {year3})")
            return []

        qa_items = []

        for result in type2_results[:max_questions]:
            try:
                # Retrieve evidence for all 3 hops using get_chunk_text
                ev1 = self.retriever.get_chunk_text(
                    result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
                ) if result.hop1_source_file else None
                ev2 = self.retriever.get_chunk_text(
                    result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
                ) if result.hop2_source_file else None
                ev3 = self.retriever.get_chunk_text(
                    result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
                ) if result.hop3_source_file else None

                evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
                evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
                evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

                if evidence1 == "No direct evidence found" and evidence2 == "No direct evidence found" and evidence3 == "No direct evidence found":
                    logger.warning(f"Missing evidence for 3-hop cross_year_type2 path")
                    continue

                difficulty = self._get_difficulty(result.connector_idf)

                # Build prompt vars like Type 1
                prompt_vars = {
                    'org': pair.org1,
                    'sector': pair.sector or 'Unknown',
                    'year1': result.hop1_year,
                    'year2': result.hop2_year,
                    'year3': result.hop3_year,
                    'connector_name': result.connector_name,
                    'connector_type': result.connector_type,
                    'idf_score': result.connector_idf,
                    'hop1_relationship': result.hop1_relationship,
                    'hop2_relationship': result.hop2_relationship,
                    'hop3_relationship': result.hop3_relationship,
                    'org1_source_file': result.hop1_source_file,
                    'org2_source_file': result.hop2_source_file,
                    'org3_source_file': result.hop3_source_file,
                    'evidence1': evidence1,
                    'evidence2': evidence2,
                    'evidence3': evidence3,
                    'difficulty': difficulty
                }

                # Prepare validation context like Type 1
                pattern = f"{pair.org1}({result.hop1_year}) -[{result.hop1_relationship}]-> {result.connector_name} <-[{result.hop2_relationship}]- {pair.org1}({result.hop2_year}), {result.connector_name} <-[{result.hop3_relationship}]- {pair.org1}({result.hop3_year})"
                entity_chain = f"{pair.org1}(y{result.hop1_year}, y{result.hop2_year}, y{result.hop3_year}) -> {result.connector_name}"
                combined_evidence = f"Year {result.hop1_year}:\n{evidence1}\n\nYear {result.hop2_year}:\n{evidence2}\n\nYear {result.hop3_year}:\n{evidence3}"
                source_files = f"{result.hop1_source_file}, {result.hop2_source_file}, {result.hop3_source_file}"

                # Try to get Type 2 specific prompt, fallback to cross_year_type1
                prompt_key = 'three_hop_cross_year_type2'
                if prompt_key not in PROMPTS:
                    prompt_key = 'three_hop_cross_year_type1'

                parsed = self._generate_with_reflection(
                    system_prompt=PROMPTS[prompt_key]['system'],
                    user_prompt_template=PROMPTS[prompt_key]['user'],
                    prompt_vars=prompt_vars,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=combined_evidence,
                    source_files=source_files,
                )

                if not parsed:
                    logger.info(f"Skipping 3-hop cross_year_type2: failed validation after retries")
                    continue

                quality_score = parsed.get('quality_score', 0)

                # Create QAItem - Type 2: 1 ORG, 3 Years → 1 Connector
                qa_item = QAItem(
                    question=parsed.get('question', ''),
                    answer=parsed.get('answer', ''),
                    reasoning_steps=parsed.get('reasoning_steps', []),
                    difficulty=difficulty,
                    evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                    hop_count=3,
                    hop_pattern='cross_year_type2',
                    org1=pair.org1,
                    org2=pair.org1,  # Same company
                    org1_year=result.hop1_year,
                    org2_year=result.hop3_year,
                    sector=pair.sector or 'Unknown',
                    sub_industry=pair.sub_industry,
                    connector_name=result.connector_name,
                    connector_type=result.connector_type,
                    connector_idf=result.connector_idf,
                    # Store year2 info in connector2 fields (repurposed for temporal Type 2)
                    connector2_name=str(result.hop2_year),
                    connector2_type='YEAR',
                    connector2_idf=result.connector_idf,
                    # Year1 -> Connector evidence
                    org1_source_file=result.hop1_source_file,
                    org1_page_id=result.hop1_page_id,
                    org1_chunk_id=result.hop1_chunk_id,
                    org1_chunk_text=ev1.text if ev1 else None,
                    org1_relationship=result.hop1_relationship,
                    # Year2 -> Connector evidence
                    org2_source_file=result.hop2_source_file,
                    org2_page_id=result.hop2_page_id,
                    org2_chunk_id=result.hop2_chunk_id,
                    org2_chunk_text=ev2.text if ev2 else None,
                    org2_relationship=result.hop2_relationship,
                    # Year3 -> Connector evidence (using ext_entity fields)
                    ext_entity_source_file=result.hop3_source_file,
                    ext_entity_page_id=result.hop3_page_id,
                    ext_entity_chunk_id=result.hop3_chunk_id,
                    ext_entity_chunk_text=ev3.text if ev3 else None,
                    ext_entity_relationship=result.hop3_relationship,
                    quality_score=quality_score
                )

                qa_items.append(qa_item)
                logger.info(f"Generated 3-hop cross_year_type2 QA: {pair.org1}({result.hop1_year}, {result.hop2_year}, {result.hop3_year}) → {result.connector_name} [score: {quality_score}/50]")

            except Exception as e:
                logger.error(f"Failed to generate 3-hop cross_year_type2 QA: {e}")
                continue

        return qa_items

    def generate_3hop_intra_doc_type2(
        self,
        pair: CompanyPair,
        max_questions: int = 3
    ) -> List[QAItem]:
        """
        Generate 3-hop QA using intra_doc_type2 pattern.
        Pattern: ORG(p1) → Connector ← ORG(p2), Connector ← ORG(p3)
        Same company, same year, 3 different pages, 1 shared connector.
        """
        type2_results = self.discovery.find_3hop_intra_doc_type2(
            org=pair.org1,
            year=pair.org1_year,
            connector_types=pair.connector_types,
            min_idf=self.medium_threshold
        )

        if not type2_results:
            logger.warning(f"No 3-hop intra_doc_type2 paths found for {pair.org1} ({pair.org1_year})")
            return []

        qa_items = []

        for result in type2_results[:max_questions]:
            try:
                # Retrieve evidence for all 3 hops using get_chunk_text
                ev1 = self.retriever.get_chunk_text(
                    result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
                ) if result.hop1_source_file else None
                ev2 = self.retriever.get_chunk_text(
                    result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
                ) if result.hop2_source_file else None
                ev3 = self.retriever.get_chunk_text(
                    result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
                ) if result.hop3_source_file else None

                evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
                evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
                evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

                if evidence1 == "No direct evidence found" and evidence2 == "No direct evidence found" and evidence3 == "No direct evidence found":
                    logger.warning(f"Missing evidence for 3-hop intra_doc_type2 path")
                    continue

                difficulty = self._get_difficulty(result.connector_idf)

                # Build prompt vars like Type 1
                prompt_vars = {
                    'org': pair.org1,
                    'year': pair.org1_year,
                    'sector': pair.sector or 'Unknown',
                    'page1': result.hop1_page_id,
                    'page2': result.hop2_page_id,
                    'page3': result.hop3_page_id,
                    'connector_name': result.connector_name,
                    'connector_type': result.connector_type,
                    'idf_score': result.connector_idf,
                    'hop1_relationship': result.hop1_relationship,
                    'hop2_relationship': result.hop2_relationship,
                    'hop3_relationship': result.hop3_relationship,
                    'evidence1': evidence1,
                    'evidence2': evidence2,
                    'evidence3': evidence3,
                    'difficulty': difficulty
                }

                # Prepare validation context like Type 1
                pattern = f"{pair.org1}(p{result.hop1_page_id}) -[{result.hop1_relationship}]-> {result.connector_name} <-[{result.hop2_relationship}]- {pair.org1}(p{result.hop2_page_id}), {result.connector_name} <-[{result.hop3_relationship}]- {pair.org1}(p{result.hop3_page_id})"
                entity_chain = f"{pair.org1}(p{result.hop1_page_id}, p{result.hop2_page_id}, p{result.hop3_page_id}) -> {result.connector_name}"
                combined_evidence = f"Page {result.hop1_page_id}:\n{evidence1}\n\nPage {result.hop2_page_id}:\n{evidence2}\n\nPage {result.hop3_page_id}:\n{evidence3}"
                source_files = f"{result.hop1_source_file}"

                # Try to get Type 2 specific prompt, fallback to intra_doc_type1
                prompt_key = 'three_hop_intra_doc_type2'
                if prompt_key not in PROMPTS:
                    prompt_key = 'three_hop_intra_doc_type1'

                parsed = self._generate_with_reflection(
                    system_prompt=PROMPTS[prompt_key]['system'],
                    user_prompt_template=PROMPTS[prompt_key]['user'],
                    prompt_vars=prompt_vars,
                    pattern=pattern,
                    entity_chain=entity_chain,
                    evidence=combined_evidence,
                    source_files=source_files,
                )

                if not parsed:
                    logger.info(f"Skipping 3-hop intra_doc_type2: failed validation after retries")
                    continue

                quality_score = parsed.get('quality_score', 0)

                # Create QAItem - Type 2: 1 ORG, 1 Year, 3 Pages → 1 Connector
                qa_item = QAItem(
                    question=parsed.get('question', ''),
                    answer=parsed.get('answer', ''),
                    reasoning_steps=parsed.get('reasoning_steps', []),
                    difficulty=difficulty,
                    evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                    hop_count=3,
                    hop_pattern='intra_doc_type2',
                    org1=pair.org1,
                    org2=pair.org1,  # Same company
                    org1_year=pair.org1_year,
                    org2_year=pair.org1_year,  # Same year
                    sector=pair.sector or 'Unknown',
                    sub_industry=pair.sub_industry,
                    connector_name=result.connector_name,
                    connector_type=result.connector_type,
                    connector_idf=result.connector_idf,
                    # Store page2 info in connector2 fields (repurposed for intra-doc Type 2)
                    connector2_name=str(result.hop2_page_id),
                    connector2_type='PAGE',
                    connector2_idf=result.connector_idf,
                    # Page1 -> Connector evidence
                    org1_source_file=result.hop1_source_file,
                    org1_page_id=result.hop1_page_id,
                    org1_chunk_id=result.hop1_chunk_id,
                    org1_chunk_text=ev1.text if ev1 else None,
                    org1_relationship=result.hop1_relationship,
                    # Page2 -> Connector evidence
                    org2_source_file=result.hop2_source_file,
                    org2_page_id=result.hop2_page_id,
                    org2_chunk_id=result.hop2_chunk_id,
                    org2_chunk_text=ev2.text if ev2 else None,
                    org2_relationship=result.hop2_relationship,
                    # Page3 -> Connector evidence (using ext_entity fields)
                    ext_entity_source_file=result.hop3_source_file,
                    ext_entity_page_id=result.hop3_page_id,
                    ext_entity_chunk_id=result.hop3_chunk_id,
                    ext_entity_chunk_text=ev3.text if ev3 else None,
                    ext_entity_relationship=result.hop3_relationship,
                    quality_score=quality_score
                )

                qa_items.append(qa_item)
                logger.info(f"Generated 3-hop intra_doc_type2 QA: {pair.org1}({pair.org1_year}) p{result.hop1_page_id}, p{result.hop2_page_id}, p{result.hop3_page_id} → {result.connector_name} [score: {quality_score}/50]")

            except Exception as e:
                logger.error(f"Failed to generate 3-hop intra_doc_type2 QA: {e}")
                continue

        return qa_items

    def _get_causal_prompt_key(self, pattern_type: str) -> str:
        mapping = {
            "causal_shared_driver": "three_hop_causal_shared_driver",
            "causal_shared_outcome": "three_hop_causal_shared_outcome",
            "causal_cross_anchor": "three_hop_causal_cross_anchor",
        }
        if pattern_type not in mapping:
            raise ValueError(f"Unknown causal pattern type: {pattern_type}")
        return mapping[pattern_type]

    def generate_3hop_causal_for_pair(
        self,
        pair: CompanyPair,
        max_questions: int = 3,
        pattern_names: Optional[List[str]] = None
    ) -> List[QAItem]:
        """
        Generate 3-hop causal QA (Pattern A/B/C) for a cross-company pair.
        Round-robin across patterns: A -> B -> C -> repeat.
        """
        if pair.pair_category != "cross_company":
            logger.warning("Causal 3-hop is only for cross_company pairs")
            return []

        year = pair.org1_year
        if pair.org1_year != pair.org2_year:
            logger.warning("Causal 3-hop expects same-year pairs; skipping")
            return []

        name_to_family = {
            "both_orgs_originator": "shared_driver",
            "both_orgs_recipient": "shared_outcome",
            "both_orgs_separate": "cross_anchor",
        }
        if pattern_names:
            pattern_sequence = [name_to_family[n] for n in pattern_names if n in name_to_family]
        else:
            pattern_sequence = ["shared_driver", "shared_outcome", "cross_anchor"]
        items_by_pattern: Dict[str, List[QAItem]] = {p: [] for p in pattern_sequence}

        for pattern_family in pattern_sequence:
            results = self.discovery.find_3hop_causal_patterns(
                org1=pair.org1,
                org2=pair.org2,
                year=year,
                pattern_family=pattern_family,
                limit=max_questions
            )

            for result in results:
                # Causal patterns do not use IDF; assign a fixed difficulty.
                difficulty = "medium"

                ev1 = self.retriever.get_chunk_text(
                    result.hop1_source_file, result.hop1_page_id, result.hop1_chunk_id
                ) if result.hop1_source_file else None
                ev2 = self.retriever.get_chunk_text(
                    result.hop2_source_file, result.hop2_page_id, result.hop2_chunk_id
                ) if result.hop2_source_file else None
                ev3 = self.retriever.get_chunk_text(
                    result.hop3_source_file, result.hop3_page_id, result.hop3_chunk_id
                ) if result.hop3_source_file else None

                evidence1 = f"[{ev1.reference}]\n{ev1.text}" if ev1 and ev1.text else "No direct evidence found"
                evidence2 = f"[{ev2.reference}]\n{ev2.text}" if ev2 and ev2.text else "No direct evidence found"
                evidence3 = f"[{ev3.reference}]\n{ev3.text}" if ev3 and ev3.text else "No direct evidence found"

                pattern_type = result.pattern_type
                prompt_key = self._get_causal_prompt_key(pattern_type)

                if pattern_type == "causal_shared_outcome":
                    x1_name = result.hop2_entity_name
                    x1_type = result.hop2_entity_type
                    x2_name = result.connector1_name
                    x2_type = result.connector1_type
                else:
                    x1_name = result.connector1_name
                    x1_type = result.connector1_type
                    x2_name = result.hop2_entity_name
                    x2_type = result.hop2_entity_type

                # Prepare prompt variables
                prompt_vars = {
                    'org1': pair.org1,
                    'org2': pair.org2,
                    'year1': pair.org1_year,
                    'year2': pair.org2_year,
                    'sector': pair.sector or 'Unknown',
                    'x1_name': x1_name,
                    'x1_type': x1_type,
                    'x2_name': x2_name,
                    'x2_type': x2_type,
                    'hop1_relationship': result.hop1_relationship,
                    'hop2_relationship': result.hop2_relationship,
                    'hop3_relationship': result.hop3_relationship,
                    'evidence1': evidence1,
                    'evidence2': evidence2,
                    'evidence3': evidence3,
                    'difficulty': difficulty
                }

                # Prepare validation context
                pattern = f"{pair.org1} -[{result.hop1_relationship}]-> {x1_name} -[{result.hop2_relationship}]-> {x2_name} <-[{result.hop3_relationship}]- {pair.org2}"
                entity_chain = f"{pair.org1} -> {x1_name} -> {x2_name} <- {pair.org2}"
                combined_evidence = f"Hop 1 ({pair.org1}):\n{evidence1}\n\nHop 2 (causal link):\n{evidence2}\n\nHop 3 ({pair.org2}):\n{evidence3}"
                source_files = f"{result.hop1_source_file}, {result.hop2_source_file}, {result.hop3_source_file}"

                try:
                    parsed = self._generate_with_reflection(
                        system_prompt=PROMPTS[prompt_key]["system"],
                        user_prompt_template=PROMPTS[prompt_key]["user"],
                        prompt_vars=prompt_vars,
                        pattern=pattern,
                        entity_chain=entity_chain,
                        evidence=combined_evidence,
                        source_files=source_files,
                        max_attempts=3,
                        quality_threshold=35
                    )

                    if not parsed:
                        logger.info(f"Skipping causal {pattern_type}: failed validation after retries")
                        continue

                    quality_score = parsed.get('quality_score', 0)

                    qa_item = QAItem(
                        question=parsed.get("question", ""),
                        answer=parsed.get("answer", ""),
                        reasoning_steps=parsed.get("reasoning_steps", []),
                        difficulty=difficulty,
                        evidence_ids=[result.hop1_chunk_id, result.hop2_chunk_id, result.hop3_chunk_id],
                        hop_count=3,
                        hop_pattern=pattern_type,
                        org1=pair.org1,
                        org2=pair.org2,
                        org1_year=pair.org1_year,
                        org2_year=pair.org2_year,
                        sector=pair.sector,
                        sub_industry=pair.sub_industry,
                        connector_name=result.connector1_name,
                        connector_type=result.connector1_type,
                        connector_idf=result.connector1_idf,
                        connector2_name=result.hop2_entity_name,
                        connector2_type=result.hop2_entity_type,
                        connector2_idf=result.hop3_entity_idf,
                        combined_idf_score=result.combined_idf_score,
                        # ORG1 evidence
                        org1_source_file=result.hop1_source_file,
                        org1_page_id=result.hop1_page_id,
                        org1_chunk_id=result.hop1_chunk_id,
                        org1_chunk_text=ev1.text if ev1 else None,
                        org1_relationship=result.hop1_relationship,
                        # ORG2 evidence
                        org2_source_file=result.hop3_source_file,
                        org2_page_id=result.hop3_page_id,
                        org2_chunk_id=result.hop3_chunk_id,
                        org2_chunk_text=ev3.text if ev3 else None,
                        org2_relationship=result.hop3_relationship,
                        # Causal link evidence (x1 -> x2)
                        ext_entity_source_file=result.hop2_source_file,
                        ext_entity_page_id=result.hop2_page_id,
                        ext_entity_chunk_id=result.hop2_chunk_id,
                        ext_entity_chunk_text=ev2.text if ev2 else None,
                        ext_entity_relationship=result.hop2_relationship,
                        quality_score=quality_score
                    )
                    items_by_pattern[pattern_family].append(qa_item)
                except Exception as e:
                    logger.error(f"Failed to generate causal QA: {e}")
                    continue

        # Round-robin across A/B/C
        interleaved: List[QAItem] = []
        idx = 0
        while len(interleaved) < max_questions and any(items_by_pattern.values()):
            pattern = pattern_sequence[idx % len(pattern_sequence)]
            if items_by_pattern[pattern]:
                interleaved.append(items_by_pattern[pattern].pop(0))
            idx += 1

        return interleaved

    def generate_3hop_causal_batch(
        self,
        pairs: List[CompanyPair],
        max_questions_per_pair: int = 3,
        max_total_questions: Optional[int] = None,
        pattern_names: Optional[List[str]] = None,
        max_workers: int = 20
    ) -> List[QAItem]:
        """Generate causal 3-hop QA across multiple pairs with parallel processing."""
        all_items: List[QAItem] = []
        lock = threading.Lock()
        stop_flag = threading.Event()
        processed_count = [0]

        def process_pair(pair_with_index):
            """Process a single pair for causal 3-hop."""
            idx, pair = pair_with_index
            if stop_flag.is_set():
                return []

            try:
                logger.info(f"Processing causal 3-hop pair {idx+1}/{len(pairs)}: {pair.org1} vs {pair.org2}")
                items = self.generate_3hop_causal_for_pair(pair, max_questions_per_pair, pattern_names)
                logger.info(f"Generated {len(items)} causal 3-hop items for {pair.org1}-{pair.org2}")
                return items
            except Exception as e:
                logger.error(f"Failed causal 3-hop for {pair.org1}-{pair.org2}: {e}")
                return []

        logger.info(f"Starting parallel causal 3-hop generation with {max_workers} workers for {len(pairs)} pairs")

        # Use batch size equal to workers to allow early stopping
        batch_size = max_workers
        pair_index = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pair_index < len(pairs):
                # Check if we've already hit the target
                if max_total_questions and len(all_items) >= max_total_questions:
                    logger.info(f"Causal target reached ({len(all_items)} >= {max_total_questions}), stopping early")
                    break

                # Submit next batch
                batch_end = min(pair_index + batch_size, len(pairs))
                batch_pairs = [(i, pairs[i]) for i in range(pair_index, batch_end)]

                logger.info(f"Submitting causal batch {pair_index+1}-{batch_end} of {len(pairs)} pairs")

                futures = {
                    executor.submit(process_pair, pair_with_idx): pair_with_idx[1]
                    for pair_with_idx in batch_pairs
                }

                # Wait for this batch to complete
                for future in as_completed(futures):
                    pair = futures[future]
                    try:
                        items = future.result()

                        with lock:
                            all_items.extend(items)
                            processed_count[0] += 1
                            current_total = len(all_items)
                            logger.info(f"Completed causal {processed_count[0]}/{len(pairs)} pairs (total: {current_total})")

                            if max_total_questions and current_total >= max_total_questions:
                                logger.info(f"Reached causal target limit ({max_total_questions}), will stop after this batch...")
                                stop_flag.set()

                    except Exception as e:
                        logger.error(f"Causal future failed for {pair.org1}-{pair.org2}: {e}")

                pair_index = batch_end

        if max_total_questions and len(all_items) > max_total_questions:
            all_items = all_items[:max_total_questions]

        logger.info(f"Generated {len(all_items)} total causal 3-hop items from {len(pairs)} pairs (parallel)")
        return all_items

    # =========================================================================
    # UNIFIED GENERATION METHODS
    # =========================================================================

    def generate_qa_for_pair(
        self,
        pair: CompanyPair,
        max_questions: int = 3,
        hop_count: int = 2,
        pattern: Optional[str] = None
    ) -> List[QAItem]:
        """
        Generate QA items for a pair with specified hop count and pattern.

        Args:
            pair: CompanyPair object
            max_questions: Maximum questions to generate
            hop_count: 2 or 3
            pattern: For 3-hop, specify 'cross_company_type1', 'entity_extension', or 'temporal_extension'
                    For 2-hop, auto-selected based on pair type

        Returns:
            List of QAItem objects
        """
        if hop_count == 2:
            if pair.pair_category == 'cross_year':
                return self.generate_qa_for_cross_year_pair(pair, max_questions)
            else:
                return self.generate_qa_for_cross_company_pair(pair, max_questions)

        elif hop_count == 3:
            # Type 1 patterns (Connector Chain: O─C1─C2─O)
            if pattern == 'cross_company_type1':
                return self.generate_3hop_cross_company_type1(pair, max_questions)
            elif pattern == 'cross_year_type1':
                return self.generate_3hop_cross_year_type1(pair, max_questions)
            elif pattern == 'intra_doc_type1':
                return self.generate_3hop_intra_doc_type1(pair, max_questions)
            # Type 2 patterns (Multi-Branch: 3 anchors → 1 connector)
            elif pattern == 'cross_company_type2':
                # Note: cross_company_type2 needs 3 ORGs - handled separately in pipeline
                logger.warning("cross_company_type2 requires 3 ORGs - use generate_3hop_cross_company_type2 directly")
                return []
            elif pattern == 'cross_year_type2':
                return self.generate_3hop_cross_year_type2(pair, max_questions)
            elif pattern == 'intra_doc_type2':
                return self.generate_3hop_intra_doc_type2(pair, max_questions)
            else:
                # Auto-select based on pair type
                if pair.pair_category == 'cross_year':
                    return self.generate_3hop_cross_year_type1(pair, max_questions)
                elif pair.pair_category == 'intra_doc':
                    return self.generate_3hop_intra_doc_type1(pair, max_questions)
                else:
                    return self.generate_3hop_cross_company_type1(pair, max_questions)

        else:
            raise ValueError(f"Invalid hop_count: {hop_count}. Must be 2 or 3.")

    def generate_qa_batch(
        self,
        pairs: List[CompanyPair],
        max_questions_per_pair: int = 3,
        max_pairs: Optional[int] = None,
        hop_count: int = 2,
        pattern: Optional[str] = None,
        max_total_questions: Optional[int] = None,
        max_workers: int = 20
    ) -> List[QAItem]:
        """
        Generate QA items for multiple pairs with parallel processing.

        Args:
            pairs: List of CompanyPair objects
            max_questions_per_pair: Max questions per pair
            max_pairs: Optional limit on number of pairs to process
            hop_count: 2 or 3
            pattern: For 3-hop, specify pattern or None for auto-selection
            max_total_questions: Stop after generating this many questions total
            max_workers: Maximum parallel workers (default 20)

        Returns:
            List of all QAItem objects
        """
        if max_pairs:
            pairs = pairs[:max_pairs]

        all_qa_items = []
        lock = threading.Lock()
        stop_flag = threading.Event()
        processed_count = [0]  # Use list for mutable counter in closure

        def process_pair(pair_with_index):
            """Process a single pair - runs in thread."""
            idx, pair = pair_with_index
            if stop_flag.is_set():
                return []

            try:
                logger.info(f"Processing pair {idx+1}/{len(pairs)}: {pair.org1} vs {pair.org2} ({pair.pair_category}) - {hop_count}-hop")
                qa_items = self.generate_qa_for_pair(pair, max_questions_per_pair, hop_count, pattern)
                logger.info(f"Generated {len(qa_items)} QA items for {pair.org1}-{pair.org2}")
                return qa_items
            except Exception as e:
                logger.error(f"Failed to process pair {pair.org1}-{pair.org2}: {e}")
                return []

        # Parallel execution with batched submission for early stopping
        logger.info(f"Starting parallel QA generation with {max_workers} workers for {len(pairs)} pairs")

        # Use batch size equal to workers to allow early stopping
        batch_size = max_workers
        pair_index = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pair_index < len(pairs):
                # Check if we've already hit the target
                if max_total_questions and len(all_qa_items) >= max_total_questions:
                    logger.info(f"Target reached ({len(all_qa_items)} >= {max_total_questions}), stopping early")
                    break

                # Submit next batch
                batch_end = min(pair_index + batch_size, len(pairs))
                batch_pairs = [(i, pairs[i]) for i in range(pair_index, batch_end)]

                logger.info(f"Submitting batch {pair_index+1}-{batch_end} of {len(pairs)} pairs")

                futures = {
                    executor.submit(process_pair, pair_with_idx): pair_with_idx[1]
                    for pair_with_idx in batch_pairs
                }

                # Wait for this batch to complete
                for future in as_completed(futures):
                    pair = futures[future]
                    try:
                        qa_items = future.result()

                        with lock:
                            all_qa_items.extend(qa_items)
                            processed_count[0] += 1
                            current_total = len(all_qa_items)
                            logger.info(f"Completed {processed_count[0]}/{len(pairs)} pairs (total QA items: {current_total})")

                            # Check if we've hit the limit
                            if max_total_questions and current_total >= max_total_questions:
                                logger.info(f"Reached target limit ({max_total_questions}), will stop after this batch...")
                                stop_flag.set()

                    except Exception as e:
                        logger.error(f"Future failed for {pair.org1}-{pair.org2}: {e}")

                pair_index = batch_end

        # Truncate to exact limit if we went over
        if max_total_questions and len(all_qa_items) > max_total_questions:
            all_qa_items = all_qa_items[:max_total_questions]

        logger.info(f"Generated {len(all_qa_items)} total {hop_count}-hop QA items from {len(pairs)} pairs (parallel)")
        return all_qa_items

    def generate_from_config(
        self,
        pairs: List[CompanyPair],
        max_questions_per_pair: int = 3,
        max_pairs: Optional[int] = None
    ) -> Dict[str, List[QAItem]]:
        """
        Generate QA items based on hop_settings in config.yaml.

        Returns:
            Dictionary with keys like '2hop_cross_company', '3hop_cross_company_type1', etc.
        """
        hop_settings = self.config.get('hop_settings', {})
        enabled_hops = hop_settings.get('enabled_hops', [2])

        results = {}

        if max_pairs:
            pairs = pairs[:max_pairs]

        # Separate pairs by type
        cross_company_pairs = [p for p in pairs if p.pair_category == 'cross_company']
        cross_year_pairs = [p for p in pairs if p.pair_category == 'cross_year']
        intra_doc_pairs = [p for p in pairs if p.pair_category == 'intra_doc']

        # Generate 2-hop if enabled
        if 2 in enabled_hops and hop_settings.get('two_hop', {}).get('enabled', True):
            two_hop_patterns = hop_settings.get('two_hop', {}).get('patterns', [])

            for pattern_cfg in two_hop_patterns:
                if not pattern_cfg.get('enabled', True):
                    continue

                pattern_name = pattern_cfg.get('name')

                if pattern_name == 'cross_company':
                    logger.info(f"Generating 2-hop cross_company QA for {len(cross_company_pairs)} pairs...")
                    items = self.generate_qa_batch(cross_company_pairs, max_questions_per_pair, hop_count=2)
                    results['2hop_cross_company'] = items

                elif pattern_name == 'cross_year':
                    logger.info(f"Generating 2-hop temporal QA for {len(cross_year_pairs)} pairs...")
                    items = self.generate_qa_batch(cross_year_pairs, max_questions_per_pair, hop_count=2)
                    results['2hop_cross_year'] = items

        # Generate 3-hop if enabled
        if 3 in enabled_hops and hop_settings.get('three_hop', {}).get('enabled', True):
            three_hop_patterns = hop_settings.get('three_hop', {}).get('patterns', [])

            for pattern_cfg in three_hop_patterns:
                if not pattern_cfg.get('enabled', True):
                    continue

                pattern_name = pattern_cfg.get('name')

                # Type 1 patterns (Connector Chain: O─C1─C2─O)
                if pattern_name == 'cross_company_type1':
                    logger.info(f"Generating 3-hop cross_company_type1 QA for {len(cross_company_pairs)} pairs...")
                    items = self.generate_qa_batch(cross_company_pairs, max_questions_per_pair, hop_count=3, pattern='cross_company_type1')
                    results['3hop_cross_company_type1'] = items

                elif pattern_name == 'cross_year_type1':
                    logger.info(f"Generating 3-hop cross_year_type1 QA for {len(cross_year_pairs)} pairs...")
                    items = self.generate_qa_batch(cross_year_pairs, max_questions_per_pair, hop_count=3, pattern='cross_year_type1')
                    results['3hop_cross_year_type1'] = items

                elif pattern_name == 'intra_doc_type1':
                    logger.info(f"Generating 3-hop intra_doc_type1 QA for {len(intra_doc_pairs)} pairs...")
                    items = self.generate_qa_batch(intra_doc_pairs, max_questions_per_pair, hop_count=3, pattern='intra_doc_type1')
                    results['3hop_intra_doc_type1'] = items

        # Summary
        total = sum(len(items) for items in results.values())
        logger.info(f"Generated {total} total QA items across {len(results)} patterns")
        for key, items in results.items():
            logger.info(f"  {key}: {len(items)} items")

        return results

    def save_qa_items(
        self,
        qa_items: List[QAItem],
        output_path: str
    ) -> None:
        """
        Save QA items to JSON file.

        Args:
            qa_items: List of QAItem objects
            output_path: Path to output file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = [item.to_dict(question_id=i+1) for i, item in enumerate(qa_items)]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(qa_items)} QA items to {output_path}")


def main():
    """Demo: Generate QA for a few sample pairs."""
    from pair_generator import PairGenerator

    print("=" * 80)
    print("Multi-hop QA Generator Demo")
    print("=" * 80)

    # Load config
    config = load_config()

    # Get validated pairs
    print("\nLoading validated pairs...")
    pair_gen = PairGenerator(config=config, validate_connectors=True)

    try:
        # Get a small sample of pairs
        cross_company_pairs = [p for p in pair_gen.generate_all_pairs_from_config()
                       if p.pair_category == 'cross_company' and p.hard_connectors >= 5][:2]
        cross_year_pairs = [p for p in pair_gen.generate_all_pairs_from_config()
                       if p.pair_category == 'cross_year' and p.hard_connectors >= 5][:2]

        sample_pairs = cross_company_pairs + cross_year_pairs
        print(f"Selected {len(sample_pairs)} sample pairs")

        # Initialize QA generator
        qa_gen = QAGenerator(config=config)

        try:
            # Generate QA
            print("\nGenerating QA items...")
            qa_items = qa_gen.generate_qa_batch(sample_pairs, max_questions_per_pair=2)

            # Display results
            print("\n" + "=" * 80)
            print("Generated QA Items")
            print("=" * 80)

            for i, item in enumerate(qa_items):
                print(f"\n--- QA Item {i+1} ({item.pair_type}) ---")
                print(f"Pair: {item.org1}({item.org1_year}) vs {item.org2}({item.org2_year})")
                print(f"Connector: {item.connector_used} ({item.connector_type})")
                print(f"Difficulty: {item.difficulty} (IDF: {item.connector_idf:.2f})")
                print(f"\nQ: {item.question}")
                print(f"\nA: {item.answer[:500]}...")
                print(f"\nReasoning: {item.reasoning_steps}")

            # Save results
            output_dir = Path(__file__).parent / "outputs"
            output_dir.mkdir(exist_ok=True)
            qa_gen.save_qa_items(qa_items, str(output_dir / "sample_qa.json"))

        finally:
            qa_gen.close()

    finally:
        pair_gen.close()


if __name__ == "__main__":
    main()
