"""
Evaluate generated criteria against expert criteria (rubrics).
Computes coverage and uniquenessmetrics.
"""
import json
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

def load_healthbench_data(path) -> List[Dict[str, Any]]:
    """Load HealthBench data from URL."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def add_expert_criteria(results: List[Dict], healthbench_data: List[Dict]) -> List[Dict]:
    """Add expert_criteria (rubrics) to results by matching prompt_id."""
    rubrics_lookup = {ex['prompt_id']: ex['rubrics'] for ex in healthbench_data if 'prompt_id' in ex and 'rubrics' in ex}
    
    for result in results:
        prompt_id = result.get('prompt_id') or result.get('id')
        if prompt_id and prompt_id in rubrics_lookup:
            result['expert_criteria'] = rubrics_lookup[prompt_id]
    
    return results


def judge_uniqueness(question: str, expert_criteria: List[Dict], model_criteria: List[Dict], call_fn) -> List[Dict]:
    """Judge alignment between model criteria and expert criteria."""
    converted_expert = [f"[{c.get('points', 1)}pts] {c.get('criterion', c.get('text', ''))}" for c in expert_criteria]
    converted_model = [f"[{c.get('points', 1)}pts] {c.get('criterion', '')}" for c in model_criteria]
    
    batch_size = 10
    all_results = []
    
    for i in range(0, len(converted_model), batch_size):
        batch = converted_model[i:i + batch_size]
        try:
            result = call_fn("ModelUniqueCriteriaJudger", {
                "question": question,
                "expert_criteria": "\n".join(converted_expert),
                "model_criteria": "\n".join(batch),
            })
            if isinstance(result, dict):
                batch_criteria = result.get('model_criteria', result.get('criteria', []))
            elif isinstance(result, list):
                batch_criteria = result
            else:
                batch_criteria = []
            all_results.extend(batch_criteria)
        except Exception as e:
            print(f"Error in judge_uniqueness: {e}")
    
    return all_results


def judge_coverage(question: str, expert_criteria: List[Dict], model_criteria: List[Dict], call_fn) -> List[Dict]:
    """Judge how well model criteria cover expert criteria."""
    converted_expert = [f"[{c.get('points', 1)}pts] {c.get('criterion', c.get('text', ''))}" for c in expert_criteria]
    converted_model = [f"[{c.get('points', 1)}pts] {c.get('criterion', '')}" for c in model_criteria]
    
    batch_size = 10
    all_results = []
    
    for i in range(0, len(converted_expert), batch_size):
        batch = converted_expert[i:i + batch_size]
        try:
            result = call_fn("CriteriaAlignmentJudger", {
                "question": question,
                "model_criteria": "\n".join(converted_model),
                "expert_criteria": "\n".join(batch),
            })
            if isinstance(result, dict):
                batch_criteria = result.get('expert_criteria', result.get('criteria', []))
            elif isinstance(result, list):
                batch_criteria = result
            else:
                batch_criteria = []
            all_results.extend(batch_criteria)
        except Exception as e:
            print(f"Error in judge_coverage: {e}")
    
    return all_results


def calculate_criteria_counts(data: Dict) -> Dict[str, int]:
    """Count scenarios, perspectives, and criteria at different stages."""
    return {
        'scenarios_count': len(data.get('scenarios', [])),
        'raw_perspectives_count': len(data.get('raw_perspectives', [])),
        'reviewed_perspectives_count': len(data.get('reviewed_perspectives', [])),
        'raw_criteria_count': len(data.get('raw_criteria', [])),
        'reviewed_criteria_count': len(data.get('reviewed_criteria', [])),
        'final_criteria_count': len(data.get('final_criteria', [])),
    }


def calculate_criteria_metrics(data: Dict) -> Dict[str, Any]:
    """Calculate criteria quality metrics from alignment results."""
    if 'error' in data or 'alignment_results' not in data:
        return None
    
    alignment = data.get('alignment_results', {})
    expert_results = alignment.get('expert_criteria', [])
    model_results = alignment.get('model_criteria', [])
    final_criteria = data.get('final_criteria', [])
    
    expert_results = [c for c in expert_results if isinstance(c, dict)]
    model_results = [c for c in model_results if isinstance(c, dict)]
    
    expert_covered = sum(1 for c in expert_results if c.get('covered', False) or c.get('is_covered') == 'yes')
    
    model_aligned = sum(1 for c in model_results if c.get('aligned', False) or c.get('is_covered') == 'yes')
    model_valuable = sum(1 for c in model_results 
                        if (not c.get('aligned', False) and c.get('is_covered') != 'yes') 
                        and (c.get('valuable', False) or c.get('is_valuable') == 'yes'))
    model_not_covered = sum(1 for c in model_results if not c.get('aligned', False) and c.get('is_covered') != 'yes')
    
    positive_model = sum(1 for c in final_criteria if c.get('points', 0) > 0)
    negative_model = sum(1 for c in final_criteria if c.get('points', 0) < 0)
    
    return {
        'total_expert_criteria': len(expert_results),
        'expert_covered_count': expert_covered,
        'total_model_criteria': len(final_criteria),
        'model_aligned_count': model_aligned,
        'model_valuable_count': model_valuable,
        'model_not_covered_count': model_not_covered,
        'positive_model_criteria': positive_model,
        'negative_model_criteria': negative_model,
    }


def analyze_criteria(results: List[Dict]):
    """Analyze criteria coverage without answer performance."""
    print("\nCRITERIA COVERAGE ANALYSIS")
    print("=" * 50)
    
    total_expert_covered = 0
    total_expert_count = 0
    total_model_valuable = 0
    total_model_aligned = 0
    total_model_count = 0
    total_positive_model = 0
    total_negative_model = 0
    
    total_scenarios = 0
    total_raw_perspectives = 0
    total_reviewed_perspectives = 0
    total_raw_criteria = 0
    total_reviewed_criteria = 0
    total_final_criteria = 0
    
    valid_count = 0
    
    for data in results:
        metrics = calculate_criteria_metrics(data)
        if metrics is None:
            continue
        
        valid_count += 1
        total_expert_covered += metrics['expert_covered_count']
        total_expert_count += metrics['total_expert_criteria']
        total_model_valuable += metrics['model_valuable_count']
        total_model_aligned += metrics['model_aligned_count']
        total_model_count += metrics['total_model_criteria']
        total_positive_model += metrics['positive_model_criteria']
        total_negative_model += metrics['negative_model_criteria']
        
        
        counts = calculate_criteria_counts(data)
        total_scenarios += counts['scenarios_count']
        total_raw_perspectives += counts['raw_perspectives_count']
        total_reviewed_perspectives += counts['reviewed_perspectives_count']
        total_raw_criteria += counts['raw_criteria_count']
        total_reviewed_criteria += counts['reviewed_criteria_count']
        total_final_criteria += counts['final_criteria_count']
    
    if valid_count == 0:
        print("No valid results to analyze.")
        return {}
    
    expert_coverage_pct = total_expert_covered / total_expert_count * 100 if total_expert_count > 0 else 0
    model_valuable_pct = total_model_valuable / total_model_count * 100 if total_model_count > 0 else 0
    model_aligned_pct = total_model_aligned / total_model_count * 100 if total_model_count > 0 else 0
    
    print(f"Questions analyzed: {valid_count}")
    print(f"Average Model Criteria per Question: {total_model_count / valid_count:.1f}")
    print(f"Average Expert Criteria per Question: {total_expert_count / valid_count:.1f}")
    
    if total_model_count > 0:
        print(f"Positive Model Criteria: {total_positive_model}/{total_model_count} ({total_positive_model / total_model_count * 100:.1f}%)")
        print(f"Negative Model Criteria: {total_negative_model}/{total_model_count} ({total_negative_model / total_model_count * 100:.1f}%)")
        print(f"Model criteria aligned: {total_model_aligned}/{total_model_count} ({model_aligned_pct:.1f}%)")
        print(f"Model criteria valuable (unique): {total_model_valuable}/{total_model_count} ({model_valuable_pct:.1f}%)")
    
    print(f"Expert criteria covered: {total_expert_covered}/{total_expert_count} ({expert_coverage_pct:.1f}%)")
    
    print("\n" + "-" * 50)
    print("GENERATION COUNTS")
    print("-" * 50)
    print(f"Average Scenarios per Question: {total_scenarios / valid_count:.1f}")
    print(f"Average Raw Perspectives per Question: {total_raw_perspectives / valid_count:.1f}")
    print(f"Average Reviewed Perspectives per Question: {total_reviewed_perspectives / valid_count:.1f}")
    print(f"Average Raw Criteria per Question: {total_raw_criteria / valid_count:.1f}")
    print(f"Average Reviewed Criteria per Question: {total_reviewed_criteria / valid_count:.1f}")
    print(f"Average Final Criteria per Question: {total_final_criteria / valid_count:.1f}")
    
    return {
        "questions_analyzed": valid_count,
        "expert_coverage_rate": expert_coverage_pct / 100,
        "model_aligned_rate": model_aligned_pct / 100,
        "model_valuable_rate": model_valuable_pct / 100,
        "avg_model_criteria": total_model_count / valid_count,
        "avg_expert_criteria": total_expert_count / valid_count,
    }


def evaluate_single(item: Dict, call_fn) -> Dict:
    """Evaluate a single result."""
    if 'error' in item or 'expert_criteria' not in item or 'final_criteria' not in item:
        return item
    
    question = item['question']
    if '[Retrieved Web Context]' in question:
        question = question.split('[Retrieved Web Context]')[0]
    
    expert = item['expert_criteria']
    model = item['final_criteria']
    
    try:
        uniqueness = judge_uniqueness(question, expert, model, call_fn)
        coverage = judge_coverage(question, expert, model, call_fn)
        
        item['alignment_results'] = {
            'model_criteria': uniqueness,
            'expert_criteria': coverage,
        }
    except Exception as e:
        item['evaluation_error'] = str(e)
    
    return item


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated criteria")
    parser.add_argument("--input", "-i", required=True, help="Input results JSON file")
    parser.add_argument("--output", "-o", help="Output file (default: input_evaluated.json)")
    parser.add_argument("--model", default="gpt-4.1", help="Model for evaluation")
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results (no LLM calls)")
    parser.add_argument("--healthbench-path", default='../HealthBench_Data/1000.json', help="HealthBench data path")
    args = parser.parse_args()
    
    print(f"Loading results from {args.input}...")
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    if isinstance(results, dict):
        if 'results' in results:
            results = results['results']
        else:
            results = list(results.values()) if all(isinstance(v, dict) for v in results.values()) else [results]
    
    print("Loading HealthBench data...")
    healthbench_data = load_healthbench_data(path=args.healthbench_path)
    results = add_expert_criteria(results, healthbench_data)
    
    with_expert = sum(1 for r in results if 'expert_criteria' in r)
    print(f"Results with expert criteria: {with_expert}/{len(results)}")
    
    if with_expert == 0:
        print("No results have expert criteria. Check prompt_id matching.")
        return
    
    if args.analyze_only:
        metrics = analyze_criteria(results)
        return
    
    from qworld import CriteriaGenerator
    gen = CriteriaGenerator(model=args.model)
    
    # Evaluate
    print("Evaluating criteria...")
    evaluated = []
    
    def process(item):
        return evaluate_single(item, gen._call_llm)
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process, item) for item in results]
        for future in tqdm(as_completed(futures), total=len(futures)):
            evaluated.append(future.result())
    
    metrics = analyze_criteria(evaluated)
    
    output_file = args.output or args.input.replace('.json', '_evaluated.json')
    with open(output_file, 'w') as f:
        json.dump({"metrics": metrics, "results": evaluated}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
