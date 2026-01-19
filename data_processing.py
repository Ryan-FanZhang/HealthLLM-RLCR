"""
Healthcare Dataset Processing for RLCR Training

This module processes PMData healthcare datasets (fatigue, readiness, sleep_quality, stress)
into the required format for RLCR training.

Required input data format (CSV):
- problem: The full problem text (includes task description + sensor data)
- answer: Numeric answer (e.g., 2, 3, 6)
- source: Dataset identifier (e.g., "PMData-fatigue")

Output format after processing:
{
    "prompt": [
        {"role": "system", "content": "<system prompt>"},
        {"role": "user", "content": "\n\nPROBLEM: <problem text>\n\n"}
    ],
    "answer": "2",  # String format
    "source": "PMData-fatigue"
}
"""

from datasets import load_dataset, Dataset, DatasetDict
from system_prompts import get_sys_prompt
import numpy as np

# ============================================================================
# Main Dataset Processing Functions
# ============================================================================

def process_dataset(dataset, script_args):
    """
    Main entry point for dataset processing.
    
    Args:
        dataset: HuggingFace dataset with 'problem', 'answer', 'source' columns
        script_args: Script arguments containing sys_prompt_name, task_spec, etc.
    
    Returns:
        Processed dataset with 'prompt' column in conversation format
    """
    sys_prompt = get_sys_prompt(script_args.sys_prompt_name)

    if script_args.task_spec == "gen":
        dataset = make_generation_dataset(dataset, sys_prompt)

    return dataset

def make_generation_dataset(dataset,sys_prompt):
    def make_generation_conversation(example):
        if 'question' in example.keys():
            user_format = (
                f"\n\nPROBLEM: {example['question']}\n\n"
                )
        else:
            user_format = (
                    f"\n\nPROBLEM: {example['problem']}\n\n"
                    )
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ],
        }
    
    dataset = dataset.map(make_generation_conversation)
    return dataset

def make_healthcare_dataset(dataset, sys_prompt, add_instruction=True):
    """
    Healthcare-specific dataset processing with optional instruction enhancement.
    
    Args:
        dataset: Input dataset
        sys_prompt: System prompt string
        add_instruction: Whether to add verification instruction to problem
    
    Returns:
        Processed dataset
    """
    def process_example(example):
        problem_text = example['problem']
        
        # Optionally add instruction for answer format
        if add_instruction:
            # Extract the task type from source to customize instruction
            source = example.get('source', 'PMData')
            if 'fatigue' in source.lower():
                instruction = " Your prediction should be a single integer from 0 to 5. Only provide the predicted value within the <answer> </answer> tags."
            elif 'readiness' in source.lower():
                instruction = " Your prediction should be a single integer from 0 to 10. Only provide the predicted value within the <answer> </answer> tags."
            elif 'sleep_quality' in source.lower():
                instruction = " Your prediction should be a single integer from 1 to 5. Only provide the predicted value within the <answer> </answer> tags."
            elif 'stress' in source.lower():
                instruction = " Your prediction should be a single integer from 0 to 5. Only provide the predicted value within the <answer> </answer> tags."
            else:
                instruction = " Only provide the predicted value within the <answer> </answer> tags."
            
            problem_text = problem_text + instruction
        
        user_format = f"\n\nPROBLEM: {problem_text}\n\n"
        
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ],
            "answer": str(example['answer']),
            "source": example.get('source', 'healthcare'),
        }
    
    dataset = dataset.map(process_example)
    return dataset







