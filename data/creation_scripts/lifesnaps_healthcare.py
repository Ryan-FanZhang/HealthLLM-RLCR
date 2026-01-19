"""
LifeSnaps Healthcare Dataset Creation Script

This script processes LifeSnaps CSV files (sleep_disorder, stress_resilience)
and creates HuggingFace datasets for RLCR training.

Usage:
    # Generate datasets for all system prompts
    python data/creation_scripts/lifesnaps_healthcare.py
    
    # Generate for a specific prompt
    python data/creation_scripts/lifesnaps_healthcare.py --sys_prompt tabc

Tasks:
    - sleep_disorder: Binary classification (0 or 1)
    - stress_resilience: Regression (0.2 to 5)
"""

import os
import sys
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from data_processing import process_dataset


# ============================================================================
# Configuration
# ============================================================================

# CSV file paths - modify these according to your environment
CSV_FILES = {
    "sleep_disorder": "/home/fzha836/HealthLLM-RLCR/data/LifeSnaps/LifeSnaps_sleep_disorder.csv",
    "stress_resilience": "/home/fzha836/HealthLLM-RLCR/data/LifeSnaps/LifeSnaps_stress_resilience.csv",
}

# All available system prompts
ALL_SYS_PROMPTS = ["gen", "tac", "tabc", "tabc_long"]

# Output directory for processed datasets
OUTPUT_DIR = "./data/lifesnaps_processed"

# Default settings
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SEED = 42


# ============================================================================
# Script Arguments
# ============================================================================

class ScriptArgs:
    """Mock script arguments for dataset processing."""
    def __init__(self, sys_prompt_name="tabc", task_spec="gen"):
        self.sys_prompt_name = sys_prompt_name
        self.task_spec = task_spec


# ============================================================================
# Main Functions
# ============================================================================

def load_single_csv(csv_path, train_ratio=0.9, seed=42):
    """Load a single CSV file and split into train/test."""
    print(f"  Loading: {csv_path}")
    
    df = pd.read_csv(csv_path)

    # Convert answer to string to avoid type conflicts when combining datasets
    # (sleep_disorder has int answers, stress_resilience has float answers)
    if 'answer' in df.columns:
        df['answer'] = df['answer'].astype(str)
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split
    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


def load_all_csv_files(csv_files, train_ratio=0.9, seed=42):
    """Load all CSV files and combine into a single DatasetDict."""
    all_train = []
    all_test = []
    
    for task_name, csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            print(f"  Warning: File not found - {csv_path}, skipping...")
            continue
        
        train_ds, test_ds = load_single_csv(csv_path, train_ratio, seed)
        all_train.append(train_ds)
        all_test.append(test_ds)
        print(f"    - {task_name}: train={len(train_ds)}, test={len(test_ds)}")
    
    if not all_train:
        print("Error: No valid CSV files found!")
        return None
    
    # Combine all datasets
    combined_train = concatenate_datasets(all_train)
    combined_test = concatenate_datasets(all_test)
    
    # Shuffle combined
    combined_train = combined_train.shuffle(seed=seed)
    combined_test = combined_test.shuffle(seed=seed)
    
    return DatasetDict({"train": combined_train, "test": combined_test})


def create_dataset_for_prompt(raw_dataset, sys_prompt_name, output_dir):
    """Create processed dataset for a specific system prompt."""
    script_args = ScriptArgs(sys_prompt_name=sys_prompt_name, task_spec="gen")
    
    # Process dataset
    processed = process_dataset(raw_dataset, script_args)
    
    # Save
    save_path = os.path.join(output_dir, f"lifesnaps_{sys_prompt_name}")
    processed.save_to_disk(save_path)
    
    return processed, save_path


def create_datasets_for_all_prompts(csv_files, output_dir, train_ratio=0.9, seed=42):
    """
    Create datasets for ALL system prompts.
    
    Output structure:
        output_dir/
        ├── lifesnaps_gen/
        ├── lifesnaps_tac/
        ├── lifesnaps_tabc/
        └── lifesnaps_tabc_long/
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Step 1: Loading all CSV files")
    print(f"{'='*70}")
    
    # Load raw data once (without processing)
    raw_dataset = load_all_csv_files(csv_files, train_ratio, seed)
    
    if raw_dataset is None:
        return
    
    print(f"\nTotal samples: train={len(raw_dataset['train'])}, test={len(raw_dataset['test'])}")
    
    print(f"\n{'='*70}")
    print("Step 2: Creating datasets for each system prompt")
    print(f"{'='*70}")
    
    results = {}
    
    for sys_prompt_name in ALL_SYS_PROMPTS:
        print(f"\n--- Processing with '{sys_prompt_name}' prompt ---")
        
        processed, save_path = create_dataset_for_prompt(
            raw_dataset, sys_prompt_name, output_dir
        )
        
        results[sys_prompt_name] = {
            "path": save_path,
            "train_size": len(processed['train']),
            "test_size": len(processed['test']),
        }
        
        print(f"  Saved to: {save_path}")
        print(f"  Train: {len(processed['train'])}, Test: {len(processed['test'])}")
    
    return results


def create_dataset_for_single_prompt(csv_files, output_dir, sys_prompt_name, train_ratio=0.9, seed=42):
    """Create dataset for a single specific system prompt."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Creating dataset with '{sys_prompt_name}' prompt")
    print(f"{'='*70}")
    
    # Load raw data
    print("\nLoading CSV files...")
    raw_dataset = load_all_csv_files(csv_files, train_ratio, seed)
    
    if raw_dataset is None:
        return None
    
    print(f"\nTotal samples: train={len(raw_dataset['train'])}, test={len(raw_dataset['test'])}")
    
    # Process
    print(f"\nProcessing with '{sys_prompt_name}' prompt...")
    processed, save_path = create_dataset_for_prompt(raw_dataset, sys_prompt_name, output_dir)
    
    print(f"\nSaved to: {save_path}")
    
    return processed


def main():
    parser = argparse.ArgumentParser(description="Process LifeSnaps healthcare CSV files")
    parser.add_argument("--sys_prompt", type=str, default="all",
                        choices=["all", "gen", "tac", "tabc", "tabc_long"],
                        help="System prompt to use. 'all' generates datasets for all prompts.")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                        help="Train/test split ratio")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for processed datasets")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push to HuggingFace Hub")
    parser.add_argument("--hub_name", type=str, default=None,
                        help="HuggingFace Hub dataset name prefix (e.g., 'username/lifesnaps')")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("LifeSnaps Healthcare Dataset Processing")
    print(f"{'='*70}")
    print(f"System prompt: {args.sys_prompt}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nTasks:")
    print(f"  - sleep_disorder: Binary classification (0 or 1)")
    print(f"  - stress_resilience: Regression (0.2 to 5)")
    
    # Process datasets
    if args.sys_prompt == "all":
        results = create_datasets_for_all_prompts(
            CSV_FILES, args.output_dir, args.train_ratio, args.seed
        )
        
        # Push to Hub if requested
        if args.push_to_hub and args.hub_name and results:
            print(f"\n{'='*70}")
            print("Pushing to HuggingFace Hub")
            print(f"{'='*70}")
            for prompt_name, info in results.items():
                from datasets import load_from_disk
                ds = load_from_disk(info['path'])
                hub_path = f"{args.hub_name}_{prompt_name}"
                print(f"  Pushing {prompt_name} to {hub_path}...")
                ds.push_to_hub(hub_path, private=True)
    else:
        processed = create_dataset_for_single_prompt(
            CSV_FILES, args.output_dir, args.sys_prompt,
            args.train_ratio, args.seed
        )
        
        # Push to Hub if requested
        if args.push_to_hub and processed and args.hub_name:
            hub_path = f"{args.hub_name}_{args.sys_prompt}"
            print(f"\nPushing to HuggingFace Hub: {hub_path}")
            processed.push_to_hub(hub_path, private=True)
    
    # Print summary
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"{'='*70}")
    
    if args.sys_prompt == "all":
        print("\nGenerated datasets:")
        for prompt_name in ALL_SYS_PROMPTS:
            print(f"  - {args.output_dir}/lifesnaps_{prompt_name}/")
        
        print("\nTo load a dataset:")
        print("  from datasets import load_from_disk")
        print(f"  dataset = load_from_disk('{args.output_dir}/lifesnaps_tabc')")
    else:
        print(f"\nTo load the dataset:")
        print("  from datasets import load_from_disk")
        print(f"  dataset = load_from_disk('{args.output_dir}/lifesnaps_{args.sys_prompt}')")


if __name__ == "__main__":
    main()

