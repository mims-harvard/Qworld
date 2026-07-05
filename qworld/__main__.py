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


def _load_input(path, input_format="auto"):
    """Load JSON or JSONL input for CLI batch generation."""
    if input_format == "auto":
        input_format = "jsonl" if path.endswith(".jsonl") else "json"

    with open(path, "r", encoding="utf-8") as f:
        if input_format == "jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if input_format == "json":
            return json.load(f)

    raise ValueError(f"Unsupported input format: {input_format}")


def _write_output(path, results, output_format="json"):
    """Write CLI results as JSON or JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if output_format == "jsonl":
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            return
        if output_format == "json":
            json.dump(results, f, ensure_ascii=False, indent=2)
            return

    raise ValueError(f"Unsupported output format: {output_format}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation criteria for questions")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("-m", "--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--input-format", choices=["auto", "json", "jsonl"], default="auto")
    parser.add_argument("--output-format", choices=["json", "jsonl"], default="json")
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
    
    data = _load_input(args.input, args.input_format)
    
    if args.max_examples:
        data = data[:args.max_examples]
    
    # Resume: filter already processed
    existing = []
    if args.resume and os.path.exists(args.output):
        existing = _load_input(args.output, args.output_format)
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
    all_results = existing + results
    
    _write_output(args.output, all_results, args.output_format)
    
    success = sum(1 for r in results if 'error' not in r)
    print(f"Done: {success}/{len(results)} successful. Saved to {args.output}")


if __name__ == "__main__":
    main()
