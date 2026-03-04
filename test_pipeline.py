import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from qworld import CriteriaGenerator
import dotenv

dotenv.load_dotenv()

with open('../HealthBench_Data/1000.json', 'r') as f:
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

for item in data:
    item['question'] = convert_conversation_to_string(item['prompt'])
    del item['prompt']

gen = CriteriaGenerator(
            model='gpt-4.1',
            base_url=None,
            temperature=0.4,
            n_scenario_expands=3,
            n_perspective_expands=4,
            n_criteria_expands=3,
            max_retries=5,
            max_workers=16,
            debug=False,  # Enable debug output to see raw LLM responses
        )

results = gen.generate(data)

with open('test_gpt-4.1-1000.json', 'w') as f:
    json.dump(results, f)
