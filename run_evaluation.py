#!/usr/bin/env python
"""Test script for Modal evaluation with MONAI instance."""

import argparse
import json
from pathlib import Path
from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from swebench.harness.modal_eval.run_evaluation_modal import run_instances_modal
from swebench.harness.test_spec import make_test_spec
from datasets import load_dataset

def test_modal_evaluation(instance_id, run_id, patch_file):
    """Test modal evaluation with a specific instance."""
    
    # Load the dataset to get the instance data
    print(f"Loading SWE-Gym dataset...")
    dataset = load_dataset("SWE-Gym/SWE-Gym", split="train")
    
    # Find the specific instance
    instance_data = None
    for item in dataset:
        if item["instance_id"] == instance_id:
            instance_data = item
            break
    
    if not instance_data:
        print(f"Instance {instance_id} not found in dataset!")
        return
    
    print(f"Found instance: {instance_id}")
    print(f"Problem statement: {instance_data['problem_statement'][:200]}...")
    
    # Read patch from file
    patch_path = Path(patch_file)
    if patch_path.exists():
        print(f"Reading patch from: {patch_path}")
        with open(patch_path, 'r') as f:
            model_patch = f.read()
    else:
        print(f"Patch file not found: {patch_path}")
        print("Using default test patch...")
        model_patch = """diff --git a/test.py b/test.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/test.py
@@ -0,0 +1 @@
+# Test patch
"""
    
    # Create prediction dict
    prediction = {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: "test_model",
        KEY_PREDICTION: model_patch,
    }
    predictions = {instance_id: prediction}
    
    # Prepare dataset
    instances = [instance_data]
    full_dataset = instances
    
    # Timeout
    timeout = 1800  # 30 minutes
    
    print(f"\nStarting Modal evaluation:")
    print(f"- Instance ID: {instance_id}")
    print(f"- Run ID: {run_id}")
    print(f"- Timeout: {timeout} seconds")
    
    try:
        # Run the evaluation
        run_instances_modal(
            predictions=predictions,
            instances=instances,
            full_dataset=full_dataset,
            run_id=run_id,
            timeout=timeout,
        )
        
        print(f"\nEvaluation completed successfully!")
        
        # Check results
        log_dir = Path("logs") / run_id / "test_model" / instance_id
        if log_dir.exists():
            print(f"\nResults saved to: {log_dir}")
            
            # Read report if it exists
            report_file = log_dir / "report.json"
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                print(f"\nEvaluation report:")
                print(json.dumps(report, indent=2))
                
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Run Modal evaluation for SWE-Gym instances')
    parser.add_argument(
        '--instance-id',
        type=str,
        default='Project-MONAI__MONAI-1095',
        help='Instance ID to evaluate (default: Project-MONAI__MONAI-1095)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run ID for the evaluation (default: modal_test_<timestamp>)'
    )
    parser.add_argument(
        '--patch-file',
        type=str,
        default='test_patch.diff',
        help='Path to the patch file (default: test_patch.diff)'
    )
    
    args = parser.parse_args()
    
    # Generate run ID if not provided
    if args.run_id is None:
        args.run_id = f"modal_test"
    
    test_modal_evaluation(args.instance_id, args.run_id, args.patch_file)

if __name__ == "__main__":
    main()
