# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_guidelines() -> Dict[str, str]:
    """
    Load the JSON guideline files and return them as formatted strings.

    Returns:
        Dict: Dictionary containing the formatted guideline content
    """
    # Import logger locally to avoid circular import
    from .logging_config import get_logger

    logger = get_logger(__name__)
    # Use the same relative path structure as the prompt.yaml loading
    guidelines_dir = Path("auditor_agent/guidelines")
    writing_guidelines_file = guidelines_dir / "writing_guidelines.json"
    regulatory_guidelines_file = guidelines_dir / "compliance_guidelines.json"

    guidelines = {}

    try:
        # Load writing guidelines
        with open(writing_guidelines_file, "r", encoding="utf-8") as file:
            writing_data = json.load(file)
            guidelines["writing_guidelines"] = json.dumps(writing_data, ensure_ascii=False, indent=2)
            logger.info(f"Successfully loaded writing guidelines from {writing_guidelines_file}")

        # Load regulatory guidelines
        with open(regulatory_guidelines_file, "r", encoding="utf-8") as file:
            regulatory_data = json.load(file)
            guidelines["regulatory_guidelines"] = json.dumps(regulatory_data, ensure_ascii=False, indent=2)
            logger.info(f"Successfully loaded regulatory guidelines from {regulatory_guidelines_file}")

        return guidelines

    except FileNotFoundError as e:
        logger.error(f"Guidelines file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise


def load_prompts_config(prompts_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load prompts configuration from YAML file and insert guideline content.

    Args:
        prompts_file: Path to the prompts YAML file (can be string or Path object)

    Returns:
        Dict: The prompts configuration dictionary with guidelines inserted

    Raises:
        FileNotFoundError: If the prompts file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    # Import logger locally to avoid circular import
    from .logging_config import get_logger

    logger = get_logger(__name__)

    prompts_path = Path(prompts_file)

    try:
        # Load the YAML configuration
        with open(prompts_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logger.info(f"Successfully loaded prompts configuration from {prompts_path}")

        # Load the guidelines
        guidelines = load_guidelines()

        # Insert the guidelines into the prompt template
        if "prompt" in config and isinstance(config["prompt"], str):
            config["prompt"] = config["prompt"].format(
                writing_guidelines=guidelines["writing_guidelines"],
                regulatory_guidelines=guidelines["regulatory_guidelines"],
            )
            logger.info("Successfully inserted guidelines into prompt template")

        return config

    except FileNotFoundError:
        logger.error(f"Prompts file not found: {prompts_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {prompts_path}: {e}")
        raise