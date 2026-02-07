#!/usr/bin/env python3
"""
Balance QA dataset by difficulty (easy/medium/hard).

This script:
1. Loads all generated QA files
2. Groups by pattern and difficulty
3. Samples evenly across difficulty levels
4. Outputs balanced datasets

Usage:
    python balance_difficulty.py --mode quantitative --output balanced_quantitative.json
    python balance_difficulty.py --mode qualitative --output balanced_qualitative.json
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


# Target counts per pattern
TARGETS = {
    # 2-hop patterns
    '2hop_cross_company': 1000,
    '2hop_cross_year': 1000,
    '2hop_intra_doc': 1000,
    # 3-hop type1 patterns
    '3hop_cross_company_type1': 500,
    '3hop_cross_year_type1': 500,
    '3hop_intra_doc_type1': 500,
    # 3-hop type2 patterns (causal)
    '3hop_cross_company_type2': 1000,
    '3hop_cross_year_type2': 500,
    '3hop_intra_doc_type2': 500,
}


def load_qa_files(base_dir: Path, mode: str) -> Dict[str, List[dict]]:
    """Load all QA files for a given mode."""
    qa_by_pattern = defaultdict(list)

    mode_dir = base_dir / mode
    if not mode_dir.exists():
        print(f"Directory not found: {mode_dir}")
        return qa_by_pattern

    # Walk through all subdirectories
    for json_file in mode_dir.rglob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'qa_items' in data:
                items = data['qa_items']
            else:
                continue

            # Determine pattern from filename or path
            filename = json_file.stem
            parent = json_file.parent.name

            # Extract pattern from filename (e.g., qa_2hop_cross_company_...)
            pattern = None
            for p in TARGETS.keys():
                if p.replace('_', '') in filename.replace('_', '').replace('-', ''):
                    pattern = p
                    break

            if not pattern:
                # Try to infer from directory structure
                if '2hop' in str(json_file):
                    if 'cross-company' in str(json_file) or 'cross_company' in str(json_file):
                        pattern = '2hop_cross_company'
                    elif 'cross-year' in str(json_file) or 'cross_year' in str(json_file):
                        pattern = '2hop_cross_year'
                    elif 'intra-doc' in str(json_file) or 'intra_doc' in str(json_file):
                        pattern = '2hop_intra_doc'
                elif '3hop' in str(json_file):
                    if 'type1' in filename or 'type1' in parent:
                        if 'cross-company' in str(json_file) or 'cross_company' in str(json_file):
                            pattern = '3hop_cross_company_type1'
                        elif 'cross-year' in str(json_file) or 'cross_year' in str(json_file):
                            pattern = '3hop_cross_year_type1'
                        elif 'intra-doc' in str(json_file) or 'intra_doc' in str(json_file):
                            pattern = '3hop_intra_doc_type1'
                    elif 'type2' in filename or 'type2' in parent:
                        if 'cross-company' in str(json_file) or 'cross_company' in str(json_file):
                            pattern = '3hop_cross_company_type2'
                        elif 'cross-year' in str(json_file) or 'cross_year' in str(json_file):
                            pattern = '3hop_cross_year_type2'
                        elif 'intra-doc' in str(json_file) or 'intra_doc' in str(json_file):
                            pattern = '3hop_intra_doc_type2'

            if pattern:
                qa_by_pattern[pattern].extend(items)
                print(f"  Loaded {len(items)} items from {json_file.name} -> {pattern}")
            else:
                print(f"  Skipped {json_file.name} (unknown pattern)")

        except Exception as e:
            print(f"  Error loading {json_file}: {e}")

    return qa_by_pattern


def get_difficulty(item: dict) -> str:
    """Extract difficulty from QA item."""
    # Try different field locations
    if 'difficulty' in item:
        return item['difficulty'].lower()
    if 'metadata' in item and 'difficulty' in item['metadata']:
        return item['metadata']['difficulty'].lower()
    if 'idf_score' in item:
        # Infer from IDF score
        idf = item['idf_score']
        if idf < 3:
            return 'easy'
        elif idf < 5:
            return 'medium'
        else:
            return 'hard'
    return 'medium'  # Default


def balance_by_difficulty(
    items: List[dict],
    target_count: int,
    distribution: Dict[str, float] = None
) -> List[dict]:
    """
    Sample items to achieve balanced difficulty distribution.

    Args:
        items: All QA items for a pattern
        target_count: Target number of items
        distribution: Difficulty distribution (default: equal)

    Returns:
        Balanced list of items
    """
    if distribution is None:
        distribution = {'easy': 0.33, 'medium': 0.34, 'hard': 0.33}

    # Group by difficulty
    by_difficulty = defaultdict(list)
    for item in items:
        diff = get_difficulty(item)
        by_difficulty[diff].append(item)

    print(f"    Available: easy={len(by_difficulty['easy'])}, "
          f"medium={len(by_difficulty['medium'])}, hard={len(by_difficulty['hard'])}")

    # Calculate targets per difficulty
    balanced = []
    for diff, ratio in distribution.items():
        diff_target = int(target_count * ratio)
        available = by_difficulty[diff]

        if len(available) >= diff_target:
            sampled = random.sample(available, diff_target)
        else:
            # Use all available
            sampled = available
            print(f"    Warning: Only {len(available)} {diff} items available (needed {diff_target})")

        balanced.extend(sampled)

    # If we're short, fill with remaining items
    remaining_needed = target_count - len(balanced)
    if remaining_needed > 0:
        used_ids = {id(item) for item in balanced}
        remaining = [item for item in items if id(item) not in used_ids]
        if remaining:
            fill = random.sample(remaining, min(remaining_needed, len(remaining)))
            balanced.extend(fill)

    random.shuffle(balanced)
    return balanced[:target_count]


def main():
    parser = argparse.ArgumentParser(description='Balance QA dataset by difficulty')
    parser.add_argument('--mode', choices=['quantitative', 'qualitative'], required=True)
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    base_dir = Path(__file__).parent / 'outputs'

    print(f"\nLoading {args.mode} QA files from {base_dir}...")
    qa_by_pattern = load_qa_files(base_dir, args.mode)

    print(f"\n{'='*60}")
    print("Balancing by difficulty...")
    print('='*60)

    balanced_dataset = {}
    stats = {
        'total': 0,
        'by_pattern': {},
        'by_difficulty': defaultdict(int)
    }

    for pattern, target in TARGETS.items():
        items = qa_by_pattern.get(pattern, [])
        print(f"\n{pattern}: {len(items)} items -> target {target}")

        if not items:
            print(f"  No items found for {pattern}")
            balanced_dataset[pattern] = []
            stats['by_pattern'][pattern] = 0
            continue

        balanced = balance_by_difficulty(items, target)
        balanced_dataset[pattern] = balanced

        # Update stats
        stats['by_pattern'][pattern] = len(balanced)
        stats['total'] += len(balanced)
        for item in balanced:
            stats['by_difficulty'][get_difficulty(item)] += 1

    # Save output
    output_path = Path(args.output)
    output_data = {
        'mode': args.mode,
        'stats': {
            'total': stats['total'],
            'by_pattern': stats['by_pattern'],
            'by_difficulty': dict(stats['by_difficulty'])
        },
        'qa_items': balanced_dataset
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"Total items: {stats['total']}")
    print(f"By difficulty: {dict(stats['by_difficulty'])}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()
