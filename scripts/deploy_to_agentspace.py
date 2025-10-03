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
Deploy to AgentSpace

This script deploys ADK agents to Vertex AI Agent Engine and then to AgentSpace.
It handles the complete deployment pipeline from local agent to production AgentSpace.

Usage:
    python deploy_to_agentspace.py --agentspace-id <agentspace_id>
    python deploy_to_agentspace.py -a <agentspace_id>

The script will:
1. Initialize Vertex AI with project configuration
2. Create and test the ADK application locally
3. Deploy the application to Vertex AI Agent Engine
4. Register the agent in the specified AgentSpace
"""

import argparse
import json
import os

from typing import Optional

from auditor_agent.agent import root_agent
from auditor_agent.utils.logging_config import get_logger, setup_logging

import vertexai

from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import AuthorizedSession
from vertexai import agent_engines
from vertexai.preview import reasoning_engines

# Agent Configuration Constants
AGENT_FOLDER = "auditor_agent"
AGENT_DESCRIPTION = """Audits RFP documents for Cymbal Corp., ensuring compliance with established writing guidelines,
                       regulatory requirements and general document accuracy"""
TOOL_DESCRIPTION = """The agent audits RFP documents for Cymbal Corp., ensuring compliance with established writing guidelines,
                    regulatory requirements and general document accuracy"""

load_dotenv(f"./{AGENT_FOLDER}/.env")

# Deployment Configuration Constants
AGENT_ENGINE_REQUIREMENTS = [
    "google-adk==1.15.1",
    "google-genai==1.39.1",
    "google-cloud-aiplatform[agent_engines]==1.109.0",
]
AGENT_ENGINE_EXTRA_PACKAGES = [f"{AGENT_FOLDER}"]


class AgentSpaceDeployer:
    """
    A class to deploy ADK agents to Vertex AI Agent Engine and AgentSpace.
    """

    def __init__(
        self,
        agent_engine_project_id: Optional[str] = None,
        agent_engine_project_number: Optional[str] = None,
        agent_engine_location: Optional[str] = None,
        agent_engine_display_name: Optional[str] = None,
        agentspace_id: Optional[str] = None,
        agentspace_project_id: Optional[str] = None,
        agentspace_location: Optional[str] = None,
        agentspace_agent_display_name: Optional[str] = None,
        staging_bucket: Optional[str] = None,
    ):
        """
        Initialize the AgentSpaceDeployer.

        Args:
            agent_engine_project_id: Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT_ID from environment)
            agent_engine_project_number: Google Cloud project number (defaults to GOOGLE_PROJECT_NUMBER from environment)
            agent_engine_location: Vertex AI location (defaults to GOOGLE_CLOUD_LOCATION from environment)
            agent_engine_display_name: Display name for the Agent Engine.
            agentspace_id: Id of the Agentspace instance from AI Applications page
            agentspace_project_id: Google Cloud project number of agentspace, typically the same as agent_engine_project_id
            agentspace_location: AgentSpace multi-region ('global', 'us', or 'eu').
            agentspace_agent_display_name: Display name for the agent in AgentSpace.
            staging_bucket: GCS staging bucket for Vertex AI
        """
        self.logger = get_logger(__name__)

        # Use environment variables if parameters not provided
        if agent_engine_project_id is None:
            agent_engine_project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
            if agent_engine_project_id is None:
                raise ValueError("GOOGLE_CLOUD_PROJECT_ID environment variable not set and no google_cloud_project_id provided")

        if agent_engine_project_number is None:
            agent_engine_project_number = os.getenv("GOOGLE_CLOUD_PROJECT_NUMBER")
            if agent_engine_project_number is None:
                raise ValueError("GOOGLE_CLOUD_PROJECT_NUMBER environment variable not set and no agent_engine_project_number provided")

        if agent_engine_location is None:
            agent_engine_location = os.getenv("GOOGLE_CLOUD_LOCATION")
            if agent_engine_location is None:
                raise ValueError("GOOGLE_CLOUD_LOCATION environment variable not set and no agent_engine_location provided")

        if agent_engine_display_name is None:
            agent_engine_display_name = os.getenv("AGENT_ENGINE_AGENT_DISPLAY_NAME")
            if agent_engine_display_name is None:
                raise ValueError("AGENT_ENGINE_AGENT_DISPLAY_NAME environment variable not set and no agent_engine_display_name provided")

        if agentspace_id is None:
            agentspace_id = os.getenv("AGENTSPACE_ID")
            if agentspace_id is None:
                raise ValueError("AGENTSPACE_ID environment variable not set and no agentspace_id provided")

        if agentspace_project_id is None:
            agentspace_project_id = os.getenv("AGENTSPACE_PROJECT_ID")
            if agentspace_project_id is None:
                agentspace_project_id = agent_engine_project_id
                self.logger.warning("AGENTSPACE_PROJECT_ID environment variable not found. Using GOOGLE_CLOUD_PROJECT_ID as fallback.")
        
        if agentspace_location is None:
            agentspace_location = os.getenv("AGENTSPACE_LOCATION")
            if agentspace_location is None:
                raise ValueError("AGENTSPACE_LOCATION environment variable not set and no agentspace_location provided")
        
        if agentspace_agent_display_name is None:
            agentspace_agent_display_name = os.getenv("AGENTSPACE_AGENT_DISPLAY_NAME")
            if agentspace_agent_display_name is None:
                raise ValueError("AGENTSPACE_AGENT_DISPLAY_NAME environment variable not set and no agentspace_agent_display_name provided")
        
        if agentspace_location not in ["global", "us", "eu"]:
            raise ValueError(f"Invalid AGENTSPACE_LOCATION: '{agentspace_location}'. Must be one of 'global', 'us', or 'eu'.")

        if staging_bucket is None:
            staging_bucket = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")
            if staging_bucket is None:
                # Default staging bucket pattern as fallback
                staging_bucket = f"gs://{agent_engine_project_id}-staging"

        # Ensure staging bucket starts with gs://
        if staging_bucket and not staging_bucket.startswith("gs://"):
            staging_bucket = f"gs://{staging_bucket}"

        self.agent_engine_project_id = agent_engine_project_id
        self.agent_engine_project_number = agent_engine_project_number
        self.agent_engine_location = agent_engine_location
        self.agent_engine_display_name = agent_engine_display_name
        self.agentspace_id = agentspace_id
        self.agentspace_project_id = agentspace_project_id
        self.agentspace_location = agentspace_location
        self.agentspace_agent_display_name = agentspace_agent_display_name
        self.staging_bucket = staging_bucket

        # Initialize Vertex AI
        vertexai.init(
            project=self.agent_engine_project_id,
            location=self.agent_engine_location,
            staging_bucket=self.staging_bucket,
        )

        # Setup authentication
        creds, _ = default()
        self.authed_session = AuthorizedSession(creds)
        self.header = {"X-Goog-User-Project": self.agent_engine_project_id, "Content-Type": "application/json"}

        self.logger.info(f"Initialized AgentSpaceDeployer with project: {agent_engine_project_id}")

    def _get_discovery_engine_endpoint(self, agentspace_id: str) -> str:
        """
        Determines the correct Discovery Engine regional endpoint based on the explicit agentspace_location.

        Args:
            agentspace_id: The AgentSpace ID to build the URL for.

        Returns:
            The full, regionalized Discovery Engine API endpoint URL for the agents collection.
        """
        hostname_map = {
            "us": "us-discoveryengine.googleapis.com",
            "eu": "eu-discoveryengine.googleapis.com",
            "global": "discoveryengine.googleapis.com",
        }
        
        hostname = hostname_map[self.agentspace_location]
        
        return (
            f"https://{hostname}/v1alpha/projects/{self.agentspace_project_id}"
            f"/locations/{self.agentspace_location}/collections/default_collection/engines/"
            f"{agentspace_id}/assistants/default_assistant/agents"
        )

    def create_adk_app(self) -> reasoning_engines.AdkApp:
        """
        Create and initialize the ADK application.

        Returns:
            The initialized ADK application
        """
        try:
            self.logger.info("Creating ADK application")

            app = reasoning_engines.AdkApp(
                agent=root_agent,
                enable_tracing=True,
            )

            self.logger.info("Successfully created ADK application")
            return app

        except Exception as e:
            self.logger.error(f"Error creating ADK application: {str(e)}")
            raise

    def deploy_to_agent_engine(self, app: reasoning_engines.AdkApp):
        """
        Deploy the ADK application to Vertex AI Agent Engine.

        Args:
            app: The ADK application to deploy

        Returns:
            The deployed agent engine
        """
        try:
            self.logger.info("Deploying to Vertex AI Agent Engine")

            remote_app = agent_engines.create(
                agent_engine=app,
                display_name=self.agent_engine_display_name,
                requirements=AGENT_ENGINE_REQUIREMENTS,
                extra_packages=AGENT_ENGINE_EXTRA_PACKAGES,
            )

            self.logger.info(f"Successfully deployed to Agent Engine: {remote_app.resource_name}")
            return remote_app

        except Exception as e:
            self.logger.error(f"Error deploying to Agent Engine: {str(e)}")
            raise

    def deploy_to_agentspace(self, agent_engine, agentspace_id: str) -> dict:
        """
        Deploy the agent to AgentSpace.

        Args:
            agent_engine: The deployed agent engine
            agentspace_id: The AgentSpace ID to deploy to

        Returns:
            The response from the AgentSpace deployment
        """
        try:
            self.logger.info(f"Deploying to AgentSpace: {agentspace_id}")

            agent_config = {
                    "displayName": self.agentspace_agent_display_name,
                    "description": AGENT_DESCRIPTION,
                    "adk_agent_definition": {
                        "tool_settings": {"tool_description": TOOL_DESCRIPTION},
                        "provisioned_reasoning_engine": {"reasoning_engine": agent_engine.resource_name},
                    },
                }
            
            discovery_engine_url = self._get_discovery_engine_endpoint(agentspace_id)

            response = self.authed_session.post(
                discovery_engine_url, headers=self.header, data=json.dumps(agent_config)
            )

            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Successfully deployed to AgentSpace: {result}")
                return result
            else:
                self.logger.error(f"Failed to deploy to AgentSpace. Status: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                raise Exception(f"AgentSpace deployment failed: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Error deploying to AgentSpace: {str(e)}")
            raise

    def list_agentspace_agents(self, agentspace_id: str) -> dict:
        """
        List all agents in the specified AgentSpace.

        Args:
            agentspace_id: The AgentSpace ID to query

        Returns:
            The list of agents in the AgentSpace
        """
        try:
            self.logger.info(f"Listing agents in AgentSpace: {agentspace_id}")

            discovery_engine_url = f"{self._get_discovery_engine_endpoint(agentspace_id)}/"

            response = self.authed_session.get(discovery_engine_url, headers=self.header)

            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Found {len(result.get('agents', []))} agents in AgentSpace")
                return result
            else:
                self.logger.error(f"Failed to list AgentSpace agents. Status: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                raise Exception(f"Failed to list AgentSpace agents: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Error listing AgentSpace agents: {str(e)}")
            raise

    def deploy_complete_pipeline(self) -> dict:
        """
        Execute the complete deployment pipeline.

        Returns:
            Dictionary with deployment results
        """
        try:
            self.logger.info(f"Starting complete deployment pipeline, deploying to {self.agentspace_id}")

            app = self.create_adk_app()
            remote_app = self.deploy_to_agent_engine(app)
            agentspace_result = self.deploy_to_agentspace(remote_app, self.agentspace_id)
            agents_list = self.list_agentspace_agents(self.agentspace_id)

            results = {
                "reasoning_engine": remote_app.resource_name,
                "agentspace_deployment": agentspace_result,
                "agents_in_space": agents_list,
            }

            self.logger.info("Complete deployment pipeline finished successfully")
            return results

        except Exception as e:
            self.logger.error(f"Error in complete deployment pipeline: {str(e)}")
            raise


def main():
    """
    Main function to deploy agent to AgentSpace.
    """
    parser = argparse.ArgumentParser(description="Deploy ADK agent to AgentSpace")
    parser.add_argument("--project-id", "-p", type=str, help="Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT_ID)")
    parser.add_argument("--project-number", "-n", type=str, help="Google Cloud project number (defaults to GOOGLE_PROJECT_NUMBER)")
    parser.add_argument("--agent-engine-location", "-l", type=str, help="Vertex AI location for Agent Engine (e.g. us-central1)")
    parser.add_argument("--agent-engine-display-name", type=str, help="Display name for Agent Engine")
    parser.add_argument("--agentspace-id", "-a", type=str, help="AgentSpace ID to deploy the agent to")
    parser.add_argument("--agentspace-project", "-ap", type=str, help="AgentSpace project to deploy the agent to")
    parser.add_argument("--agentspace-location", "-al", type=str, help="AgentSpace multi-region ('global', 'us', or 'eu')")
    parser.add_argument("--agentspace-agent-display-name", type=str, help="Display name for the agent in AgentSpace")
    parser.add_argument("--staging-bucket", "-s", type=str, help="GCS staging bucket for Vertex AI")

    args = parser.parse_args()
    setup_logging()
    logger = get_logger(__name__)

    try:
        deployer = AgentSpaceDeployer(
            agent_engine_project_id=args.project_id,
            agent_engine_project_number=args.project_number,
            agent_engine_location=args.agent_engine_location,
            agent_engine_display_name=args.agent_engine_display_name,
            agentspace_id=args.agentspace_id,
            agentspace_project_id=args.agentspace_project,
            agentspace_location=args.agentspace_location,
            agentspace_agent_display_name=args.agentspace_agent_display_name,
            staging_bucket=args.staging_bucket,
        )
        results = deployer.deploy_complete_pipeline()

        logger.info("Deployment completed successfully!")
        logger.info(f"Reasoning Engine: {results['reasoning_engine']}")
        logger.info(f"AgentSpace Deployment: {results['agentspace_deployment']}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()