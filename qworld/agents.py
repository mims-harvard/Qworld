"""Core agent functions for criteria generation."""
import json
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def parse_json(result: Any) -> Any:
    """Parse JSON result recursively."""
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return parse_json(parsed) if isinstance(parsed, str) else parsed
        except json.JSONDecodeError:
            return result
    return result


def extract_list(parsed: Any, key: str) -> List:
    """Extract list from parsed result."""
    # If still a string, try to extract JSON from markdown
    if isinstance(parsed, str):
        import re
        md_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', parsed)
        if md_match:
            try:
                parsed = json.loads(md_match.group(1).strip())
            except:
                pass
        else:
            # Try direct JSON parse
            try:
                parsed = json.loads(parsed)
            except:
                pass
    
    if isinstance(parsed, dict):
        return parsed.get(key, [])
    return parsed if isinstance(parsed, list) else []


def _max_iteration(items: List[Dict], key: str) -> int:
    """Get max iteration value from items."""
    return max((item.get(key, -1) for item in items if isinstance(item.get(key), int)), default=-1)


# Agent function implementations using a generic call_fn
def generate_scenarios(question: str, call_fn: Callable, image=None) -> List[Dict]:
    result = call_fn("ScenarioAnalyzer", {"question": question}, image)
    return extract_list(result, 'scenarios')


def expand_scenarios(question: str, call_fn: Callable, scenarios: List[Dict], image=None) -> List[Dict]:
    inputs = [{"scenario_name": s["scenario_name"], "scenario_description": s["scenario_description"]} for s in scenarios]
    result = call_fn("ScenarioExpander", {"question": question, "scenarios": inputs}, image)
    return extract_list(result, 'scenarios')


def generate_perspectives(question: str, call_fn: Callable, scenario: Dict, image=None) -> List[Dict]:
    analysis = f"{scenario['scenario_name']}: {scenario['scenario_description']}"
    result = call_fn("PerspectiveAnalyzer", {"question": question, "scenario_analysis": analysis}, image)
    return extract_list(result, 'perspectives')


def expand_perspectives(question: str, call_fn: Callable, perspectives: List[Dict], image=None) -> List[Dict]:
    inputs = [{"perspective_name": p["perspective_name"], "perspective_description": p["perspective_description"]} for p in perspectives]
    result = call_fn("PerspectiveExpander", {"question": question, "perspectives": inputs}, image)
    return extract_list(result, 'perspectives')


def review_perspectives(question: str, call_fn: Callable, perspectives: List[Dict], image=None) -> List[Dict]:
    inputs = [{"perspective_name": p["perspective_name"], "perspective_description": p["perspective_description"]} for p in perspectives]
    result = call_fn("PerspectiveReviewer", {"question": question, "all_perspectives": inputs}, image)
    return extract_list(result, 'perspectives')


def generate_criteria(question: str, call_fn: Callable, perspective: Dict, image=None) -> List[Dict]:
    desc = f"{perspective['perspective_name']}: {perspective['perspective_description']}"
    result = call_fn("CriteriaGenerator", {"question": question, "perspective_description": desc}, image)
    return extract_list(result, 'criteria')


def expand_criteria(question: str, call_fn: Callable, criteria: List[Dict], image=None) -> List[Dict]:
    inputs = [{"criterion": c["criterion"], "points": c["points"]} for c in criteria]
    result = call_fn("CriteriaExpander", {"question": question, "all_criteria": inputs}, image)
    return extract_list(result, 'criteria')


def review_criteria(question: str, call_fn: Callable, criteria: List[Dict], image=None) -> List[Dict]:
    inputs = [{"criterion": c["criterion"], "points": c["points"]} for c in criteria]
    result = call_fn("CriteriaReviewer", {"question": question, "all_criteria": inputs}, image)
    return extract_list(result, 'criteria')


def check_polarity(question: str, call_fn: Callable, criteria: List[Dict], image=None) -> List[Dict]:
    texts = [f"{c['criterion_id']}: {c['criterion']}" for c in criteria]
    result = call_fn("NegativeCriteriaChecker", {"question": question, "all_criteria": texts}, image)
    polarities = extract_list(result, 'criteria')
    output = []
    for i, c in enumerate(criteria):
        pts = c["points"]
        if i < len(polarities):
            pts = abs(pts) if polarities[i].get("positive", True) else -abs(pts)
        output.append({"criterion": c["criterion"], "points": pts})
    return output


def assign_scores(question: str, call_fn: Callable, criteria: List[Dict], image=None) -> List[Dict]:
    result = call_fn("CriteriaScoreAssigner", {"question": question, "all_criteria": criteria}, image)
    return extract_list(result, 'criteria')


def deduplicate_by_embedding(criteria: List[Dict], embed_fn: Callable, threshold: float = 0.6) -> List[Dict]:
    """Deduplicate criteria using embeddings."""
    if len(criteria) <= 1:
        return criteria
    texts = [c["criterion"] for c in criteria]
    embeddings = embed_fn(texts)
    sim = cosine_similarity(embeddings)
    keep = []
    for i in range(len(criteria)):
        if not any(sim[i, j] >= threshold for j in keep):
            keep.append(i)
    return [criteria[i] for i in keep]


def run_pipeline(
    item: Dict[str, Any],
    call_fn: Callable,
    embed_fn: Callable,
    n_scenario_expands: int = 0,
    n_perspective_expands: int = 0,
    n_criteria_expands: int = 0,
    dedup_threshold: float = 0.6,
    use_parallel: bool = False,
    max_workers: int = 8,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full criteria generation pipeline."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    pipeline_start = time.time()
    time_mark = time.time()
    
    def log(step: str, counts: str):
        nonlocal time_mark
        if not verbose:
            return
        elapsed = time.time() - time_mark
        time_mark = time.time()
        if 'Done' in step:
            print(f"Total time: {time.time() - pipeline_start:.1f}s")
        else:
            print(f"[Pipeline] {step} | {counts} | {elapsed:.1f}s")
        
    
    question = item.get("question")
    prompt_id = item.get("prompt_id", item.get("id", ""))
    image = item.get("image")
    
    if not question:
        return {"prompt_id": prompt_id, "error": "question is required"}
    
    scenarios = list(item.get("scenarios", []))
    raw_perspectives = list(item.get("raw_perspectives", []))
    reviewed_perspectives = list(item.get("reviewed_perspectives", []))
    raw_criteria = list(item.get("raw_criteria", []))
    reviewed_criteria = list(item.get("reviewed_criteria", []))
    
    before_dedup = 0
    after_dedup = 0
    after_review = 0
    
    def parallel_map(fn, items):
        """Execute fn on items, parallel if use_parallel else sequential."""
        if not items:
            return []
        if use_parallel and len(items) > 1:
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(fn, item) for item in items]
                for future in tqdm(as_completed(futures), total=len(futures), disable=not verbose):
                    results.append(future.result())
            return results
        return [fn(item) for item in items]
    
    try:
        new_perspective_ids: List[str] = []
        
        # Helper for parallel perspective generation
        def gen_persp_for_scenario(sc):
            return (sc["scenario_id"], generate_perspectives(question, call_fn, sc, image) or [])
        
        def gen_criteria_for_persp(persp):
            return (persp["perspective_id"], generate_criteria(question, call_fn, persp, image) or [])
            
        # Step 1: Generate initial scenarios and perspectives
        if not scenarios:
            initial = generate_scenarios(question, call_fn, image) or []
            for idx, s in enumerate(initial):
                scenarios.append({
                    "scenario_id": f"s{idx}",
                    "scenario_name": s["scenario_name"],
                    "scenario_description": s["scenario_description"],
                    "expand_iteration": 0,
                })
            persp_results = parallel_map(gen_persp_for_scenario, scenarios)
            for scenario_id, generated in persp_results:
                for p in generated:
                    raw_perspectives.append({
                        "scenario_id": scenario_id,
                        "perspective_name": p["perspective_name"],
                        "perspective_description": p["perspective_description"],
                    })
            reviewed_all = review_perspectives(question, call_fn, raw_perspectives, image) or []
            for idx, p in enumerate(reviewed_all):
                reviewed_perspectives.append({
                    "perspective_id": f"p{idx}",
                    "perspective_name": p["perspective_name"],
                    "perspective_description": p["perspective_description"],
                    "expand_iteration": 0,
                })
                new_perspective_ids.append(f"p{idx}")
            log("Init scenarios and perspectives generation", f"scenarios={len(scenarios)} perspectives={len(reviewed_perspectives)}")
        
        # Step 2: Expand scenarios
        current = _max_iteration(scenarios, "expand_iteration")
        needed = max(0, n_scenario_expands - current)
        
        if needed > 0:
            sid_next = len(scenarios)
            new_scenarios = []
            for i in range(needed):
                iteration = current + 1 + i
                expanded = expand_scenarios(question, call_fn, scenarios, image) or []
                for s in expanded:
                    new_sc = {
                        "scenario_id": f"s{sid_next}",
                        "scenario_name": s["scenario_name"],
                        "scenario_description": s["scenario_description"],
                        "expand_iteration": iteration,
                    }
                    scenarios.append(new_sc)
                    new_scenarios.append(new_sc)
                    sid_next += 1
            persp_results = parallel_map(gen_persp_for_scenario, new_scenarios)
            for scenario_id, generated in persp_results:
                for p in generated:
                    raw_perspectives.append({
                        "scenario_id": scenario_id,
                        "perspective_name": p["perspective_name"],
                        "perspective_description": p["perspective_description"],
                    })
            reviewed_all = review_perspectives(question, call_fn, raw_perspectives, image) or []
            existing_names = {p["perspective_name"].strip() for p in reviewed_perspectives}
            pid_next = len(reviewed_perspectives)
            for p in reviewed_all:
                name = p["perspective_name"].strip()
                if name and name not in existing_names:
                    reviewed_perspectives.append({
                        "perspective_id": f"p{pid_next}",
                        "perspective_name": name,
                        "perspective_description": p["perspective_description"],
                        "expand_iteration": 0,
                    })
                    new_perspective_ids.append(f"p{pid_next}")
                    existing_names.add(name)
                    pid_next += 1
            log(f"Scenario expand x{needed}, generate perspectives for new scenarios", f"scenarios={len(scenarios)} perspectives={len(reviewed_perspectives)}")
        
        # Step 3: Expand perspectives
        current = _max_iteration(reviewed_perspectives, "expand_iteration")
        needed = max(0, n_perspective_expands - current)
        
        if needed > 0:
            pid_next = len(reviewed_perspectives)
            for i in range(needed):
                iteration = current + 1 + i
                expanded = expand_perspectives(question, call_fn, reviewed_perspectives, image) or []
                for p in expanded:
                    reviewed_perspectives.append({
                        "perspective_id": f"p{pid_next}",
                        "perspective_name": p["perspective_name"],
                        "perspective_description": p["perspective_description"],
                        "expand_iteration": iteration,
                    })
                    new_perspective_ids.append(f"p{pid_next}")
                    pid_next += 1
            log(f"Perspective expand x{needed}", f"perspectives={len(reviewed_perspectives)}")
        
        # Step 4: Generate criteria for new perspectives (parallel if enabled)
        if new_perspective_ids:
            new_persps = [p for p in reviewed_perspectives if p["perspective_id"] in new_perspective_ids]
            criteria_results = parallel_map(gen_criteria_for_persp, new_persps)
            for perspective_id, generated in criteria_results:
                for c in generated:
                    raw_criteria.append({
                        "perspective_id": perspective_id,
                        "criterion": c["criterion"],
                        "points": c.get("points", 1),
                    })
            before_dedup = len(raw_criteria)
            deduped = deduplicate_by_embedding(raw_criteria, embed_fn, dedup_threshold)
            after_dedup = len(deduped)
            reviewed_all = review_criteria(question, call_fn, deduped, image) or []
            reviewed_criteria = []
            for idx, c in enumerate(reviewed_all):
                reviewed_criteria.append({
                    "criterion_id": f"c{idx}",
                    "criterion": c["criterion"],
                    "points": c.get("points", 1),
                    "expand_iteration": 0,
                })
            after_review = len(reviewed_criteria)
            log("Criteria gen+dedup+review", f"raw={before_dedup} dedup={after_dedup} review={after_review}")
        
        # Step 5: Expand criteria
        current = _max_iteration(reviewed_criteria, "expand_iteration")
        needed = max(0, n_criteria_expands - current)
        
        if needed > 0:
            cid_next = len(reviewed_criteria)
            for i in range(needed):
                iteration = current + 1 + i
                expanded = expand_criteria(question, call_fn, reviewed_criteria, image) or []
                for c in expanded:
                    reviewed_criteria.append({
                        "criterion_id": f"c{cid_next}",
                        "criterion": c["criterion"],
                        "points": c.get("points", 1),
                        "expand_iteration": iteration,
                    })
                    cid_next += 1
            log(f"Criteria expand x{needed}", f"criteria={len(reviewed_criteria)}")
        
        # Step 6: Finalize
        final = check_polarity(question, call_fn, reviewed_criteria, image)
        final = assign_scores(question, call_fn, final, image)
        log("Check polarity + score assignment", f"final={len(final)} criteria")
        log("Done", "")
        return {
            "prompt_id": prompt_id,
            "question": question,
            "scenarios": scenarios,
            "raw_perspectives": raw_perspectives,
            "reviewed_perspectives": reviewed_perspectives,
            "raw_criteria": raw_criteria,
            "reviewed_criteria": reviewed_criteria,
            "final_criteria": final,
            "dedup_stats": {
                "before": before_dedup,
                "after": after_dedup,
                "after_review": after_review,
            },
        }
    except Exception as e:
        import traceback
        return {
            "prompt_id": prompt_id,
            "question": question,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
