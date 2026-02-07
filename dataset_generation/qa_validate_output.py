"""
Run the QA validation prompt (prompts.yaml -> validation) on an existing output JSON file.

This is useful for post-hoc scoring/filtering of generated questions.

Example:
  python -m dataset_generation_new.qa_validate_output \
    --input /path/to/qa_all_20260117_232433.json \
    --min-score 35
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import yaml

from llm import LLMClient

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_prompts() -> Dict[str, Any]:
    prompts_path = Path(__file__).parent / "prompts.yaml"
    with open(prompts_path) as f:
        return yaml.safe_load(f)


PROMPTS = _load_prompts()


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_strip_code_fences(response))
    except Exception:
        return None


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _iter_evidence_blobs(evidence: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Evidence schema varies a bit across outputs, but common shape is:
      evidence: { org1_evidence: {...}, org2_evidence: {...}, extended_entity_evidence: {...} }
    """
    if not isinstance(evidence, dict):
        return []
    out: List[Tuple[str, Dict[str, Any]]] = []
    for k, v in evidence.items():
        if isinstance(v, dict):
            out.append((k, v))
    return out


def _format_evidence_for_validation(item: Dict[str, Any], max_chars_per_chunk: int) -> Tuple[str, str]:
    """
    Extract evidence chunks from either:
    1. item["evidence"] dict (old format: {org1_evidence: {...}, org2_evidence: {...}})
    2. item["path_data"] dict (new format: {hop_1_rel: {...}, hop_2_rel: {...}})
    """
    blocks: List[str] = []
    source_files: List[str] = []

    # Try old format first: item["evidence"]
    evidence = item.get("evidence", {})
    if evidence and isinstance(evidence, dict):
        for label, blob in _iter_evidence_blobs(evidence):
            source_file = blob.get("source_file")
            if isinstance(source_file, str) and source_file:
                source_files.append(source_file)

            meta_bits = []
            for key in ("source_file", "page_id", "chunk_id", "relationship"):
                val = blob.get(key)
                if val is not None and val != "":
                    meta_bits.append(f"{key}={val}")
            meta = ", ".join(meta_bits) if meta_bits else "no-metadata"

            chunk_text = blob.get("chunk_text", "")
            if isinstance(chunk_text, str) and max_chars_per_chunk > 0:
                chunk_text = chunk_text[:max_chars_per_chunk]

            blocks.append(
                f"<chunk label='{label}' {meta}>\n{chunk_text}\n</chunk>"
            )

    # If no evidence found, try new format: item["path_data"]
    if not blocks:
        path_data = item.get("path_data", {})
        if isinstance(path_data, dict):
            # Extract hop_1_rel, hop_2_rel, and optionally hop_3_rel
            hop_keys = ["hop_1_rel", "hop_2_rel", "hop_3_rel"]
            for hop_key in hop_keys:
                hop_rel = path_data.get(hop_key)
                if isinstance(hop_rel, dict) and hop_rel.get("chunk_text"):
                    source_file = hop_rel.get("source_file", "")
                    if isinstance(source_file, str) and source_file:
                        source_files.append(source_file)

                    meta_bits = []
                    for key in ("source_file", "page_id", "chunk_id", "relationship"):
                        val = hop_rel.get(key)
                        if val is not None and val != "":
                            meta_bits.append(f"{key}={val}")
                    meta = ", ".join(meta_bits) if meta_bits else "no-metadata"

                    chunk_text = hop_rel.get("chunk_text", "")
                    if isinstance(chunk_text, str) and max_chars_per_chunk > 0:
                        chunk_text = chunk_text[:max_chars_per_chunk]

                    blocks.append(
                        f"<chunk label='{hop_key}' {meta}>\n{chunk_text}\n</chunk>"
                    )

    evidence_text = "\n\n".join(blocks) if blocks else "No evidence provided."
    source_files_text = "\n".join(sorted(set(source_files))) if source_files else "Unknown"
    return evidence_text, source_files_text


def _format_chain_for_validation(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract entity chain info from either:
    1. item["metadata"] / item["entities"] / item["connectors"] (old format)
    2. item["path_data"] (new format with start_node, intermediate_node, end_node)
    """
    md = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
    
    # Try to get org info from path_data first (new format)
    path_data = item.get("path_data", {})
    if isinstance(path_data, dict):
        start_node = path_data.get("start_node", {})
        end_node = path_data.get("end_node", {})
        intermediate_node = path_data.get("intermediate_node", {})
        
        if start_node.get("name") and end_node.get("name"):
            org1 = start_node.get("name", "ORG1")
            org2 = end_node.get("name", "ORG2")
            y1 = start_node.get("year")
            y2 = end_node.get("year")
            connector_name = intermediate_node.get("name", "")
            
            org1_s = f"{org1}({y1})" if y1 is not None else str(org1)
            org2_s = f"{org2}({y2})" if y2 is not None else str(org2)
            
            hop_count = item.get("hop_count", 2)
            pattern_str = item.get("pattern", "")
            
            if hop_count == 2 and connector_name:
                pattern = pattern_str or f"{org1_s} -[{path_data.get('hop_1_rel', {}).get('relationship', '')}]-> {connector_name} <-[{path_data.get('hop_2_rel', {}).get('relationship', '')}]- {org2_s}"
                entity_chain = f"{org1_s} -> {connector_name} -> {org2_s}"
            elif hop_count == 3 and connector_name:
                # For 3-hop, might need to check for additional connectors
                pattern = pattern_str or f"{org1_s} -> {connector_name} -> ... <- {org2_s}"
                entity_chain = f"{org1_s} -> {connector_name} -> ... -> {org2_s}"
            else:
                pattern = pattern_str or f"{org1_s} -> {connector_name} <- {org2_s}"
                entity_chain = f"{org1_s} -> {connector_name} -> {org2_s}"
            
            return pattern, entity_chain
    
    # Fall back to old format
    org1 = md.get("org1") or item.get("org1") or item.get("entities", {}).get("start") or "ORG1"
    org2 = md.get("org2") or item.get("org2") or item.get("entities", {}).get("end") or "ORG2"
    y1 = md.get("org1_year") or item.get("org1_year") or item.get("path_data", {}).get("start_node", {}).get("year")
    y2 = md.get("org2_year") or item.get("org2_year") or item.get("path_data", {}).get("end_node", {}).get("year")

    org1_s = f"{org1}({y1})" if y1 is not None else str(org1)
    org2_s = f"{org2}({y2})" if y2 is not None else str(org2)

    connectors = item.get("connectors", {}) if isinstance(item.get("connectors"), dict) else {}
    entities = item.get("entities", {}) if isinstance(item.get("entities"), dict) else {}
    c1 = _safe_get(connectors, ["connector1", "name"]) or entities.get("intermediate") or item.get("path_data", {}).get("intermediate_node", {}).get("name")
    c2 = _safe_get(connectors, ["connector2", "name"])

    hop_pattern = item.get("hop_pattern") or item.get("pattern") or item.get("hop_type") or ""
    hop_count = item.get("hop_count")

    # Keep "pattern" human-legible but not overly dependent on exact schema.
    if hop_count == 2 and c1:
        pattern = hop_pattern or f"{org1_s} -> {c1} <- {org2_s}"
        entity_chain = f"{org1_s} -> {c1} -> {org2_s}"
    elif hop_count == 3 and c1 and c2:
        pattern = hop_pattern or f"{org1_s} -> {c1} -> {c2} <- {org2_s}"
        entity_chain = f"{org1_s} -> {c1} -> {c2} -> {org2_s}"
    elif c1:
        pattern = hop_pattern.strip() if hop_pattern else f"{org1_s} -> {c1} <- {org2_s}"
        entity_chain = f"{org1_s} -> {c1} -> {org2_s}"
    else:
        pattern = hop_pattern.strip() if hop_pattern else "unknown_pattern"
        entity_chain = f"{org1_s} -> ... -> {org2_s}"

    return pattern, entity_chain


@dataclass
class ValidationResult:
    score: int
    decision: str
    breakdown: Dict[str, Any]
    explanation: str
    issues: List[str]
    suggestions: List[str]
    raw: Dict[str, Any]
    # Additional fields for v3_binary variant
    failure_tags: List[str] = None
    must_fix_issues: List[str] = None
    rewrite_suggestions: List[str] = None

    def __post_init__(self):
        if self.failure_tags is None:
            self.failure_tags = []
        if self.must_fix_issues is None:
            self.must_fix_issues = []
        if self.rewrite_suggestions is None:
            self.rewrite_suggestions = []

    @property
    def accepted(self) -> bool:
        return self.decision.lower() in ("accept", "pass")


def validate_item(
    llm: LLMClient,
    item: Dict[str, Any],
    *,
    min_score: Optional[int],
    max_chars_per_chunk: int,
    temperature: Optional[float] = None,
    prompt_variant: str = "default",
) -> ValidationResult:
    """
    Validate a QA item using the specified prompt variant.
    
    Args:
        llm: LLM client
        item: QA item dictionary
        min_score: Minimum score threshold (only used for 'default' variant)
        max_chars_per_chunk: Max characters per chunk
        temperature: LLM temperature
        prompt_variant: 'default' (scoring) or 'v3_binary' (pass/fail)
    """
    question = item.get("question", "")
    answer = item.get("answer", "")
    pattern, entity_chain = _format_chain_for_validation(item)
    evidence_text, source_files_text = _format_evidence_for_validation(item, max_chars_per_chunk=max_chars_per_chunk)

    # Select prompt variant
    if prompt_variant == 'v3_binary':
        prompt_key = 'validation_v3_binary'
    else:
        prompt_key = 'validation'
    
    if prompt_key not in PROMPTS:
        logger.warning(f"Prompt variant '{prompt_variant}' not found, falling back to 'default'")
        prompt_key = 'validation'
        prompt_variant = 'default'

    user_prompt = PROMPTS[prompt_key]["user"].format(
        pattern=pattern,
        entity_chain=entity_chain,
        question=question,
        answer=answer,
        evidence=evidence_text,
        source_files=source_files_text,
    )

    response = llm.complete(
        system_prompt=PROMPTS[prompt_key]["system"],
        user_prompt=user_prompt,
        temperature=temperature,
    )

    parsed = _parse_json_response(response) or {}
    
    # Parse response based on variant
    if prompt_variant == 'v3_binary':
        decision = str(parsed.get("decision", "fail") or "fail").lower()
        # Convert binary to score for compatibility: pass=50, fail=0
        score = 50 if decision == "pass" else 0
        breakdown = {}  # No breakdown for binary variant
        explanation = str(parsed.get("notes", "") or "")
        issues = parsed.get("must_fix_issues", []) if isinstance(parsed.get("must_fix_issues"), list) else []
        suggestions = parsed.get("rewrite_suggestions", []) if isinstance(parsed.get("rewrite_suggestions"), list) else []
        failure_tags = parsed.get("failure_tags", []) if isinstance(parsed.get("failure_tags"), list) else []
        
        # For binary variant, min_score is ignored (it's pass/fail)
        # But we still respect the decision
    else:
        # Default scoring variant
        score = int(parsed.get("score", 0) or 0)
        decision = str(parsed.get("decision", "reject") or "reject").lower()
        breakdown = parsed.get("breakdown", {}) if isinstance(parsed.get("breakdown"), dict) else {}
        explanation = str(parsed.get("explanation", "") or "")
        issues = parsed.get("issues", []) if isinstance(parsed.get("issues"), list) else []
        suggestions = parsed.get("suggestions", []) if isinstance(parsed.get("suggestions"), list) else []
        failure_tags = []
        
        # Enforce numeric threshold even if model decision is inconsistent.
        if min_score is not None and score < min_score:
            decision = "reject"
        elif decision not in ("accept", "reject"):
            decision = "accept" if (min_score is None or score >= min_score) else "reject"

    return ValidationResult(
        score=score,
        decision=decision,
        breakdown=breakdown,
        explanation=explanation,
        issues=[str(x) for x in issues],
        suggestions=[str(x) for x in suggestions],
        failure_tags=[str(x) for x in failure_tags],
        must_fix_issues=[str(x) for x in issues] if prompt_variant == 'v3_binary' else [],
        rewrite_suggestions=[str(x) for x in suggestions] if prompt_variant == 'v3_binary' else [],
        raw=parsed,
    )


def _load_input_json(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(path.read_text())

    # Common repo shape: {metadata: {...}, questions: [...]}
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return data, data["questions"]

    # Alternate shape: list of questions
    if isinstance(data, list):
        return {"metadata": {}, "questions": data}, data

    raise ValueError("Unsupported input JSON format; expected {questions:[...]} or a list of questions.")


def _process_single_file(
    in_path: Path,
    llm: LLMClient,
    min_score: Optional[int],
    max_chars_per_chunk: int,
    temperature: Optional[float],
    limit: Optional[int],
    output: Optional[Path],
    filtered_output: Optional[Path],
    model: Optional[str],
    prompt_variant: str,
) -> Tuple[int, int]:
    """
    Process a single JSON file and write validated/filtered outputs.
    
    Returns:
        Tuple of (total_questions, accepted_count)
    """
    base = in_path.with_suffix("")
    out_path = output if output else base.parent / f"{base.name}_validated.json"
    filtered_path = filtered_output if filtered_output else base.parent / f"{base.name}_filtered.json"

    wrapper, questions = _load_input_json(in_path)
    if limit is not None:
        questions = questions[: max(0, limit)]

    accepted: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions):
        if not isinstance(q, dict):
            continue

        vr = validate_item(
            llm,
            q,
            min_score=min_score,
            max_chars_per_chunk=max_chars_per_chunk,
            temperature=temperature,
            prompt_variant=prompt_variant,
        )

        validation_data = {
            "score": vr.score,
            "decision": vr.decision,
            "breakdown": vr.breakdown,
            "explanation": vr.explanation,
            "issues": vr.issues,
            "suggestions": vr.suggestions,
            "validated_at": _utc_now_iso(),
        }
        
        # Add v3_binary specific fields if present
        if prompt_variant == "v3_binary":
            validation_data["failure_tags"] = vr.failure_tags
            validation_data["must_fix_issues"] = vr.must_fix_issues
            validation_data["rewrite_suggestions"] = vr.rewrite_suggestions
        
        q["validation"] = validation_data

        # For v3_binary, accept if decision is "pass"; for default, use score threshold
        if prompt_variant == "v3_binary":
            if vr.decision.lower() == "pass":
                accepted.append(q)
        else:
            if min_score is None or vr.score >= min_score:
                accepted.append(q)

        # Lightweight progress (no extra deps)
        if (idx + 1) % 10 == 0:
            print(f"  [{in_path.name}] Validated {idx + 1}/{len(questions)} (accepted so far: {len(accepted)})")

    # Write outputs
    wrapper_out = dict(wrapper)
    wrapper_out["questions"] = questions
    wrapper_out.setdefault("metadata", {})
    wrapper_out["metadata"]["qa_validation"] = {
        "prompt_variant": prompt_variant,
        "min_score": min_score,
        "model": model,
        "validated_at": _utc_now_iso(),
        "max_chars_per_chunk": max_chars_per_chunk,
        "limit": limit,
    }
    out_path.write_text(json.dumps(wrapper_out, indent=2))

    filtered_out = dict(wrapper_out)
    filtered_out["questions"] = accepted
    filtered_out["metadata"] = dict(filtered_out.get("metadata") or {})
    filtered_out["metadata"]["total_questions"] = len(accepted)
    filtered_path.write_text(json.dumps(filtered_out, indent=2))

    print(f"  [{in_path.name}] Wrote: {out_path.name}, {filtered_path.name} (accepted {len(accepted)}/{len(questions)})")
    return len(questions), len(accepted)


def _find_json_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all JSON files in a directory, optionally recursively."""
    json_files = []
    if recursive:
        json_files = list(directory.rglob("*.json"))
    else:
        json_files = list(directory.glob("*.json"))
    
    # Filter out files that look like outputs (already validated/filtered)
    # This prevents re-processing outputs
    filtered = []
    for f in json_files:
        name = f.name.lower()
        if not (name.endswith("_validated.json") or name.endswith("_filtered.json")):
            filtered.append(f)
    
    return sorted(filtered)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score existing QA output JSON using prompts.yaml validation. "
        "Can process a single file or recursively process all JSON files in a directory."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to QA output JSON file or directory containing JSON files (e.g. qa_all_*.json or outputs/)"
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Path to write scored JSON (default: <input>_validated.json). "
        "For directories, outputs go to same subdirectory as input."
    )
    ap.add_argument(
        "--filtered-output",
        default=None,
        help="Path to write accepted-only JSON (default: <input>_filtered.json). "
        "For directories, outputs go to same subdirectory as input."
    )
    ap.add_argument(
        "--min-score",
        type=int,
        default=None,
        help="Minimum score out of 50 to accept (default: from config.yaml)"
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Override LLM model name from config.yaml (e.g. qwen3-235b)"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only validate first N questions per file"
    )
    ap.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=3500,
        help="Truncate each chunk_text to this many chars"
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override LLM temperature for validation calls"
    )
    ap.add_argument(
        "--prompt-variant",
        type=str,
        default=None,
        choices=["default", "v3_binary"],
        help="Validation prompt variant: 'default' (scoring 0-50) or 'v3_binary' (pass/fail). Default: from config.yaml"
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="If input is a directory, only process files in the top level (don't recurse into subdirectories)"
    )

    args = ap.parse_args()

    in_path = Path(os.path.expanduser(args.input)).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {in_path}")

    # Load config for defaults
    cfg_path = Path(__file__).parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    
    # Default min_score from config.yaml (only used for 'default' variant)
    if args.min_score is None:
        args.min_score = int(_safe_get(cfg, ["qa_validation", "quality_threshold"], 35))
    
    # Default prompt_variant from config.yaml
    if args.prompt_variant is None:
        args.prompt_variant = str(_safe_get(cfg, ["qa_validation", "prompt_variant"], "default"))
    
    # For v3_binary variant, min_score is ignored (it's pass/fail)
    if args.prompt_variant == "v3_binary":
        logger.info("Using v3_binary variant: min_score is ignored (binary pass/fail)")
        args.min_score = None

    llm = LLMClient(args.model)

    # Determine if input is a file or directory
    if in_path.is_file():
        # Single file processing
        print(f"Processing single file: {in_path}")
        total, accepted = _process_single_file(
            in_path,
            llm,
            args.min_score,
            args.max_chars_per_chunk,
            args.temperature,
            args.limit,
            Path(args.output) if args.output else None,
            Path(args.filtered_output) if args.filtered_output else None,
            args.model,
            args.prompt_variant,
        )
        print(f"\nSummary: {accepted}/{total} questions accepted (score >= {args.min_score})")
        
    elif in_path.is_dir():
        # Directory processing
        json_files = _find_json_files(in_path, recursive=not args.no_recursive)
        
        if not json_files:
            print(f"No JSON files found in {in_path}")
            return 1
        
        print(f"Found {len(json_files)} JSON file(s) in {in_path}")
        print(f"Processing with min_score={args.min_score}, limit={args.limit}\n")
        
        total_all = 0
        accepted_all = 0
        failed_files = []
        
        for idx, json_file in enumerate(json_files, 1):
            try:
                print(f"[{idx}/{len(json_files)}] Processing: {json_file.relative_to(in_path)}")
                total, accepted = _process_single_file(
                    json_file,
                    llm,
                    args.min_score,
                    args.max_chars_per_chunk,
                    args.temperature,
                    args.limit,
                    None,  # Use default naming in same directory
                    None,  # Use default naming in same directory
                    args.model,
                    args.prompt_variant,
                )
                total_all += total
                accepted_all += accepted
                print()
            except Exception as e:
                print(f"  ERROR processing {json_file}: {e}")
                failed_files.append((json_file, str(e)))
                print()
        
        print("=" * 60)
        print(f"Summary: Processed {len(json_files)} file(s)")
        print(f"  Total questions: {total_all}")
        print(f"  Accepted: {accepted_all} (score >= {args.min_score})")
        print(f"  Rejected: {total_all - accepted_all}")
        
        if failed_files:
            print(f"\nFailed to process {len(failed_files)} file(s):")
            for f, err in failed_files:
                print(f"  - {f}: {err}")
        
    else:
        raise ValueError(f"Input path is neither a file nor directory: {in_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

