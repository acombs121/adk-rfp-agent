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

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentCorrection(BaseModel):
    """Individual correction for a document violation."""

    correction_number: int = Field(..., description="Sequential number for this correction")
    specific_location: str = Field(
        ..., description="Exact location in document (e.g., 'Section 4.1.2', 'Appendix B, Item 3', 'Paragraph 5')"
    )
    text_before_revision: str = Field(..., description="The original incorrect text found in the document")
    text_after_revision: str = Field(..., description="The corrected text that should replace the original")
    reason_for_revision: str = Field(
        ...,
        description="Detailed explanation of why this correction is needed, including the specific rule ID and guideline reference when applicable",
    )
    violation_category: str = Field(
        ...,
        description="Category of violation: 'Content Accuracy', 'Formatting', 'Writing Standards', or 'Regulatory Compliance'",
    )
    rule_id: Optional[str] = Field(
        None,
        description="The specific rule ID from the guidelines that was violated (null for general errors not in guidelines)",
    )
    severity: str = Field(
        ...,
        description="Severity level: 'High' for content errors and regulatory violations, 'Medium' for terminology and structural issues, 'Low' for minor style issues",
    )


class DocumentAuditResult(BaseModel):
    """Complete audit result for a document."""

    corrections: List[DocumentCorrection] = Field(..., description="List of all corrections needed")