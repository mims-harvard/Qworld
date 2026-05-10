"""
Command-line interface for criteria generation.

Usage:
    python -m qworld -i input.json -o output.json -m gpt-4o
    python -m qworld -i input.json -o output.json -m deepseek-chat
    python -m qworld -i input.json -o output.json --base-url http://localhost:8000/v1
"""
import argparse
import json
import os


def _normalize_input_data(data):
    if isinstance(data, str):
        return [{"id": "0", "question": data}]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise TypeError("input JSON must be a string, object, or array")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation criteria for questions")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("-m", "--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--base-url", type=str, help="API base URL (for vLLM)")
    parser.add_argument("--api-key", type=str, help="API key (uses env vars if not set)")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--n-scenario-expands", type=int, default=3)
    parser.add_argument("--n-perspective-expands", type=int, default=4)
    parser.add_argument("--n-criteria-expands", type=int, default=3)
    parser.add_argument("--max-examples", type=int, help="Limit examples")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, 'r', encoding='utf-8') as f:
        data = _normalize_input_data(json.load(f))
    
    if args.max_examples:
        data = data[:args.max_examples]
    
    # Resume: filter already processed
    existing = []
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        done_ids = {r.get("id") or r.get("prompt_id") for r in existing if "final_criteria" in r}
        data = [d for d in data if (d.get("id") or d.get("prompt_id")) not in done_ids]
        print(f"Resuming: {len(existing)} done, {len(data)} remaining")
    
    if not data:
        print("Nothing to process")
        return
    
    print(f"Processing {len(data)} items with {args.max_workers} workers")
    print(f"Using Model: {args.model}")
    
    from .client import CriteriaGenerator
    gen = CriteriaGenerator(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        n_scenario_expands=args.n_scenario_expands,
        n_perspective_expands=args.n_perspective_expands,
        n_criteria_expands=args.n_criteria_expands,
        max_workers=args.max_workers,
    )
    
    results = gen.generate(data)
    if isinstance(results, dict):
        results = [results]
    all_results = existing + results
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    success = sum(1 for r in results if 'error' not in r)
    print(f"Done: {success}/{len(results)} successful. Saved to {args.output}")


if __name__ == "__main__":
    main()
