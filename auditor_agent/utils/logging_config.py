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

"""
Simple logging configuration.
"""

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """
    Set up basic logging configuration for the entire project.

    Args:
        level: Logging level (defaults to INFO, or from LOG_LEVEL env var)
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure basic logging once for the entire project
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)