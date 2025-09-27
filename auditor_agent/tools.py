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

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def json_to_markdown_table(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """
    Intercepts a model's JSON response and replaces it with a
    Markdown table.
    """
    if not llm_response.content or not llm_response.content.parts:
        return llm_response

    # Handle function call responses (AFC mode)
    parts = llm_response.content.parts
    if parts is None:
        return llm_response

    json_data = None

    for part in parts:
        if hasattr(part, "text") and part.text is not None:
            try:
                parsed_json = json.loads(str(part.text))
                if isinstance(parsed_json, dict) and "corrections" in parsed_json:
                    json_data = parsed_json["corrections"]
                elif isinstance(parsed_json, list):
                    json_data = parsed_json
                break
            except json.JSONDecodeError:
                continue

    if json_data is None:
        return llm_response

    if not isinstance(json_data, list):
        return llm_response

    # Filter out findings where before == after and renumber sequentially
    filtered_data = []
    filtered_count = 0

    for item in json_data:
        if isinstance(item, dict):
            text_before = str(item.get("text_before_revision", "")).strip()
            text_after = str(item.get("text_after_revision", "")).strip()

            # Filter out if before and after are identical
            if text_before == text_after:
                filtered_count += 1
                continue

            filtered_data.append(item)

    # Renumber the remaining corrections sequentially
    for i, item in enumerate(filtered_data, 1):
        item["correction_number"] = i

    json_data = filtered_data

    # Get all unique keys from all dictionaries to handle any missing fields
    all_keys = set()
    for item in json_data:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    # Define the desired column order
    desired_order = [
        "correction_number",
        "severity",
        "violation_category",
        "specific_location",
        "text_before_revision",
        "text_after_revision",
        "reason_for_revision",
        "rule_id",
    ]

    # Filter headers to include only keys present in the data, maintaining the desired order
    headers = [key for key in desired_order if key in all_keys]

    # Add any extra keys from the data that are not in the desired order to the end
    extra_keys = sorted([key for key in all_keys if key not in headers])
    headers.extend(extra_keys)

    # Create the header row
    header_row = "| " + " | ".join(headers) + " |"

    # Create the separator row
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    # Create data rows
    data_rows = []
    for item in json_data:
        row_values = []
        for header in headers:
            value = item.get(header, "")
            # Handle None values and convert to string
            if value is None:
                value = ""
            else:
                value = str(value)
            # Escape pipe characters in the data to prevent table formatting issues
            value = value.replace("|", "\\|")
            # Replace newlines with <br> for better table formatting
            value = value.replace("\n", "<br>")
            row_values.append(value)

        data_row = "| " + " | ".join(row_values) + " |"
        data_rows.append(data_row)

    # Combine all parts with a prefix sentence
    prefix = "\n## Document Audit Results\n\nThe following table shows the audit findings:\n\n"
    markdown_table = prefix + "\n".join([header_row, separator_row] + data_rows)
    new_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=markdown_table)],
        )
    )
    return new_response