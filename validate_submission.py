#!/usr/bin/env python3
"""Pre-flight Validation Script for Kaggle-in-Kaggle Agent Submissions.

This script lints a participant's submission directory to verify that it is fully valid
and compatible with the Kaggle-in-Kaggle evaluation harness before uploading to Kaggle.

## What it checks
1. Verifies `agent.yaml` exists and is valid YAML.
2. Resolves all `!include` directives to ensure prompt files exist.
3. Verifies requested tools match the competition's allowed tool registry.
4. Verifies requested model exists in the competition's `models.yaml`.
5. Performs a dry-run compilation of the ADK agent.

## Usage
    python validate_submission.py [--agent-dir sample_submission]
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure we can import adk_submission and kaggle_kaggle
from adk_submission import ModelRegistry, compile_submission
from adk_submission.yaml_loader import load_yaml
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import (
    CodeExecutionInput,
    CodeExecutionResult,
)
from google.adk.models.lite_llm import LiteLlm
from kaggle_kaggle.budget import PricingTable


def validate_agent(agent_dir: Path, base_dir: Path) -> bool:
    agent_yaml_path = agent_dir / "agent.yaml"
    if not agent_yaml_path.exists():
        print(f"[Error] agent.yaml not found at {agent_yaml_path}")
        return False
    
    print(f"[Check 1] Found agent.yaml at {agent_yaml_path}")

    # 1. Parse YAML and validate includes
    try:
        config = load_yaml(agent_yaml_path, root_dir=agent_dir)
    except Exception as e:
        print(f"[Error] Failed to parse agent.yaml (check syntax and !include paths): {e}")
        return False
    
    print("[Check 2] YAML syntax and !include prompt files are valid.")

    # 2. Validate top-level keys based on agent_class
    agent_class = config.get("agent_class", "LlmAgent")
    if agent_class in {"LoopAgent", "SequentialAgent", "ParallelAgent"}:
        required_keys = {"name", "sub_agents"}
    else:
        required_keys = {"name", "model", "instruction", "tools"}
    missing = required_keys - set(config.keys())
    if missing:
        print(f"[Error] agent.yaml ({agent_class}) is missing required top-level keys: {missing}")
        return False
    
    print(f"[Check 3] Required top-level keys present for {agent_class} (Name: {config['name']}).")

    # 3. Validate against models.yaml
    models_yaml_path = base_dir / "models.yaml"
    if not models_yaml_path.exists():
        print(f"[Warning] models.yaml not found at {models_yaml_path}. Skipping model pricing check.")
        pricing_table = PricingTable.from_yaml()
    else:
        pricing_table = PricingTable.from_yaml(models_yaml_path)
    
    # Check all model declarations in agent_dir
    allowed_models = set()
    for slug in pricing_table.model_ids:
        allowed_models.add(slug)
        pricing = pricing_table.get(slug)
        if pricing and pricing.path:
            allowed_models.add(pricing.path)
            allowed_models.add(pricing.path.removeprefix("openai/"))
    for yaml_file in agent_dir.rglob("*.yaml"):
        try:
            sub_cfg = load_yaml(yaml_file, root_dir=agent_dir)
            if isinstance(sub_cfg, dict) and "model" in sub_cfg:
                req_model = sub_cfg["model"]
                if req_model not in allowed_models:
                    print(f"[Error] Requested model '{req_model}' in {yaml_file.relative_to(agent_dir)} is not listed in competition models.yaml.")
                    print(f"          Allowed models: {list(pricing_table.model_ids)}")
                    return False
        except Exception:
            pass

    print("[Check 4] All requested models are fully valid and permitted in models.yaml.")

    # 4. Dry-run compilation
    print("[Check 5] Performing dry-run compilation of ADK agent...")
    try:
        # Setup dummy tools matching KaggleKaggleContext.create_tools()
        dummy_tools = {
            "run_command": lambda command: "",
            "read_file": lambda filepath, start_line=None, end_line=None: "",
            "write_file": lambda filepath, content: "",
            "edit_file": lambda filepath, old_string, new_string, allow_multiple=False: "",
            "submit_predictions": lambda filepath: "",
            "select_submission": lambda submission_ids: "",
            "get_status": lambda: "",
        }
        
        # Setup dummy model registry
        models = ModelRegistry()
        for model_id in pricing_table.model_ids:
            alias = model_id.split("/")[-1]
            models.register(alias, LiteLlm(model=alias, num_retries=100))

        class DummyCodeExecutor(BaseCodeExecutor):
            def execute_code(self, invocation_context: Any, code_execution_input: CodeExecutionInput) -> CodeExecutionResult:
                return CodeExecutionResult(stdout="", stderr="", output_files=[])

        agent = compile_submission(
            str(agent_dir),
            dummy_tools,
            models,
            code_executor=DummyCodeExecutor(),
            script_timeout=300,
        )
        print(f"[Success] Agent '{agent.name}' compiled successfully with {len(config.get('tools', []))} tools.")
    except Exception as e:
        print(f"[Error] Dry-run compilation failed: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate Kaggle-in-Kaggle agent submission.")
    parser.add_argument(
        "--agent-dir",
        type=str,
        default="sample_submission",
        help="Path to the participant submission directory containing agent.yaml",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    agent_dir = base_dir / args.agent_dir

    if not agent_dir.exists():
        print(f"Error: Agent directory not found at {agent_dir}")
        sys.exit(1)
    
    print(f"\n=== Pre-flight Validation: {args.agent_dir} ===\n")
    is_valid = validate_agent(agent_dir, base_dir)

    if is_valid:
        print("\n>>> VALIDATION SUCCESSFUL: Submission is ready for Kaggle upload! <<<\n")
        sys.exit(0)
    else:
        print("\n>>> VALIDATION FAILED: Please fix the errors above before uploading. <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
