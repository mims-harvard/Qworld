"""
Test script for CriteriaGenerator with multiple providers.
Tests each provider with all questions (with images).

Usage:
    python -m criteria_gen.test_providers
    python -m criteria_gen.test_providers --provider openai
    python -m criteria_gen.test_providers --n-questions 4
"""
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
import dotenv

dotenv.load_dotenv()

def load_hle_with_images(n: int = 8) -> List[Dict[str, Any]]:
    """Load n HLE questions that have images."""
    from datasets import load_dataset
    
    print("Loading HLE dataset...")
    dataset = load_dataset("cais/hle", split="test")
    
    items = []
    for row in dataset:
        if row.get("image") and len(items) < n:
            question = f"Question: {row['question']}\nAnswer: {row['answer']}\nRationale: {row['rationale']}\n\nIf one or more images are provided, focus on the image content as well."
            items.append({
                "id": row["id"],
                "question": question,
                "image": row["image"],
            })
    
    print(f"Loaded {len(items)} HLE questions with images")
    return items

def load_healthbench_data(n: int = 8, path='../HealthBench_Data/1000.json') -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        data = json.load(f)
    def convert_conversation_to_string(conversation_list):
        conversation_parts = ["User-assistant conversation: \n"]
        if isinstance(conversation_list, list):
            for message in conversation_list:
                role = message.get('role', 'user')
                content = message.get('content', '')
                conversation_parts.append(f"{role}: {content}")
        else:
            conversation_parts.append(conversation_list)
        return "\n".join(conversation_parts)
    questions = []
    for item in data:
        item['question'] = convert_conversation_to_string(item['prompt'])
        del item['prompt']
    questions = data[:n]
    return questions


def load_fallback_questions(n: int = 8) -> List[Dict[str, Any]]:
    """Fallback when HealthBench/HLE not available."""
    samples = [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning.",
        "What are the main steps in training a neural network?",
    ]
    return [{"id": f"q{i}", "question": samples[i % len(samples)]} for i in range(n)]


def test_single_question(gen, question: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
    """Test a single question and return result."""
    print(f"  [{idx+1}/{total}] Question ID: {question.get('id', question.get('prompt_id', ''))}")
    
    try:
        result = gen.generate(question)
        
        if "error" in result:
            print(f"    ERROR: {result['error']}")
            if "traceback" in result:
                print(f"    TRACEBACK:\n{result['traceback']}")
            # Show partial results if available
            for key in ["partial_scenarios", "partial_perspectives", "partial_criteria"]:
                if result.get(key):
                    print(f"    {key}: {result[key]}")
            return {"id": question["id"], "success": False, "error": result["error"]}
        
        n_scenarios = len(result.get('scenarios', []))
        n_perspectives = len(result.get('reviewed_perspectives', []))
        n_criteria = len(result.get('final_criteria', []))
        print(f"    SUCCESS: {n_scenarios} scenarios, {n_perspectives} perspectives, {n_criteria} criteria")
        
        return {
            "id": question.get("id", question.get("prompt_id", "")),
            "success": True,
            "n_scenarios": n_scenarios,
            "n_perspectives": n_perspectives,
            "n_criteria": n_criteria,
            "result": result,
        }
    except Exception as e:
        print(f"    EXCEPTION: {e}")
        return {"id": question.get("id", question.get("prompt_id", "")), "success": False, "error": str(e)}


def test_provider(
    provider: str,
    model: str,
    questions: List[Dict[str, Any]],
    base_url: str = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Test a provider with all questions sequentially."""
    from qworld import CriteriaGenerator
    
    print(f"\n{'='*70}")
    print(f"Testing: {provider} ({model})")
    print(f"{'='*70}")
    
    try:
        gen = CriteriaGenerator(
            model=model,
            base_url=base_url,
            temperature=0.4,
            n_scenario_expands=0,
            n_perspective_expands=0,
            n_criteria_expands=0,
            max_retries=5,
            max_workers=1,  # Sequential
            debug=debug,  # Enable debug output to see raw LLM responses
        )
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return {
            "provider": provider,
            "model": model,
            "initialized": False,
            "error": str(e),
            "results": [],
        }
    
    results = []
    for idx, q in enumerate(questions):
        result = test_single_question(gen, q, idx, len(questions))
        results.append(result)
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\n  Summary: {success_count}/{len(results)} successful")
    
    return {
        "provider": provider,
        "model": model,
        "initialized": True,
        "success_count": success_count,
        "total_count": len(results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test CriteriaGenerator providers")
    parser.add_argument("--provider", type=str, default="all",
                        choices=["all", "openai", "claude", "gemini", "grok", "deepseek", "vllm"],
                        help="Provider to test")
    parser.add_argument("--n-questions", type=int, default=8, help="Number of questions")
    parser.add_argument("--vllm-url", type=str, help="vLLM server URL")
    parser.add_argument("--vllm-model", type=str, default="qwen3-30b")
    parser.add_argument("--output", "-o", type=str, help="Output file for results")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    # Load test data
    # questions = load_hle_with_images(args.n_questions)
    questions = load_healthbench_data(args.n_questions)
    
    # Provider configurations: (model, base_url)
    providers = {
        "openai": ('gpt-4.1', None),
        'vllm': (args.vllm_model, args.vllm_url or 'http://localhost:8000/v1'),
        "claude": ("claude-sonnet-4-5", None),
        "gemini": ("gemini-3-flash-preview", None),
        "grok": ("grok-4-1-fast-non-reasoning", None),
        "deepseek": ("deepseek-chat", None),
    }
    
    if args.provider == "all":
        test_providers = providers
    else:
        if args.provider not in providers:
            if args.provider == "vllm":
                print("Error: vLLM requires --vllm-url or VLLM_SERVER_URL env var")
            else:
                print(f"Error: Provider {args.provider} not available")
            return
        test_providers = {args.provider: providers[args.provider]}
    
    all_results = {}
    for provider, (model, base_url) in test_providers.items():
        result = test_provider(provider, model, questions, base_url, args.debug)
        all_results[provider] = result
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for provider, info in all_results.items():
        if not info.get("initialized"):
            status = f"INIT FAILED: {info.get('error', 'unknown')[:50]}"
        else:
            status = f"{info['success_count']}/{info['total_count']} passed"
        print(f"  {provider:12} ({info.get('model', 'N/A'):30}): {status}")
    
    if args.output:
        save_results = {}
        for p, info in all_results.items():
            save_info = {k: v for k, v in info.items() if k != "results"}
            save_info["question_results"] = [
                {k: v for k, v in r.items()}
                for r in info.get("results", [])
            ]
            save_results[p] = save_info
        
        with open(args.output, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
