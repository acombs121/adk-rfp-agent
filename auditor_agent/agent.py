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


from auditor_agent.model import DocumentAuditResult
from auditor_agent.tools import json_to_markdown_table
from auditor_agent.utils.load_prompt import load_prompts_config

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import load_artifacts
from google.genai.types import GenerateContentConfig

# Gemini model configuration
GEMINI_MODEL = "gemini-2.5-pro"

# Load prompt configurations for all agents
file_retrieval_prompts_config = load_prompts_config("auditor_agent/prompts/retrieval_agent_prompt.yaml")
spelling_grammar_prompts_config = load_prompts_config("auditor_agent/prompts/spelling_grammar_agent_prompt.yaml")
guidelines_compliance_prompts_config = load_prompts_config(
    "auditor_agent/prompts/guidelines_compliance_agent_prompt.yaml"
)
aggregator_prompts_config = load_prompts_config("auditor_agent/prompts/aggregator_agent_prompt.yaml")

# --- Agent 1: File Retrieval Agent ---
# Retrieves uploaded files using load_artifacts and extracts their text content
file_retrieval_agent: LlmAgent = LlmAgent(
    name="file_retrieval_agent",
    model=GEMINI_MODEL,
    instruction=file_retrieval_prompts_config["prompt"],
    description="Get the content of uploaded files using load_artifacts",
    output_key="file_content",
    tools=[load_artifacts],
    generate_content_config=GenerateContentConfig(temperature=0, seed=5),
)

# --- Agent 2: Spelling & Grammar Auditor Agent (First Pass) ---
# Specialized agent for spelling, grammar, punctuation, and basic accuracy
spelling_grammar_auditor_agent: LlmAgent = LlmAgent(
    name="spelling_grammar_auditor_agent",
    model=GEMINI_MODEL,
    instruction=spelling_grammar_prompts_config["prompt"] + "\n\n**Document Content:**\n{file_content}",
    description="First-pass auditor specializing in spelling, grammar, punctuation, and basic document accuracy for RRP procurement documents",
    output_key="spelling_grammar_audit_result",  # Stores output for next agent
    generate_content_config=GenerateContentConfig(temperature=0, seed=5),
)

# --- Agent 3: Guidelines Compliance Auditor Agent (Second Pass) ---
# Specialized agent for guidelines compliance and regulatory requirements
guidelines_compliance_auditor_agent: LlmAgent = LlmAgent(
    name="guidelines_compliance_auditor_agent",
    model=GEMINI_MODEL,
    instruction=guidelines_compliance_prompts_config["prompt"]
    + "\n\n**Document Content:**\n{file_content}\n\n**Spelling/Grammar Audit Results:**\n{spelling_grammar_audit_result}",
    description="Second-pass auditor specializing in guidelines compliance and regulatory requirements for RFP procurement documents",
    output_key="guidelines_compliance_audit_result",  # Stores output for aggregator
    generate_content_config=GenerateContentConfig(temperature=0, seed=5),
)

# --- Agent 4: Audit Results Aggregator Agent (Final Pass) ---
# Consolidates results from both audits, resolves duplicates and contradictions
audit_results_aggregator_agent: LlmAgent = LlmAgent(
    name="audit_results_aggregator_agent",
    model=GEMINI_MODEL,
    instruction=aggregator_prompts_config["prompt"]
    + "\n\n**Document Content:**\n{file_content}\n\n**Spelling/Grammar Audit Results:**\n{spelling_grammar_audit_result}\n\n**Guidelines Compliance Audit Results:**\n{guidelines_compliance_audit_result}",
    description="Final-pass aggregator that consolidates audit results, resolves duplicates and contradictions, and ensures document grounding",
    generate_content_config=GenerateContentConfig(temperature=0, seed=5),
    output_schema=DocumentAuditResult,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    after_model_callback=json_to_markdown_table,
)

# --- Document Audit Pipeline ---
# Orchestrates the complete three-stage document auditing workflow
audit_pipeline: SequentialAgent = SequentialAgent(
    name="RFPAuditPipeline",
    sub_agents=[
        file_retrieval_agent,  # Stage 0: Retrieve document
        spelling_grammar_auditor_agent,  # Stage 1: Spelling & Grammar
        guidelines_compliance_auditor_agent,  # Stage 2: Guidelines Compliance
        audit_results_aggregator_agent,  # Stage 3: Results Aggregation
    ],
    description="RFP procurement document auditing pipeline: (1) File retrieval, (2) Spelling/Grammar check, (3) Guidelines compliance review, (4) Results aggregation and conflict resolution for RFP procurement documents.",
    # The agents will run in sequence: File Retrieval -> Spelling/Grammar -> Guidelines Compliance -> Results Aggregation
)

# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent: SequentialAgent = audit_pipeline