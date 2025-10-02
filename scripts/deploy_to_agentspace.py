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

# Load environment variables
load_dotenv("./auditor_agent/.env")


# Agent Configuration Constants
AGENT_DISPLAY_NAME = os.getenv("AGENT_DISPLAY_NAME")
AGENT_ENGINE_DISPLAY_NAME = os.getenv("AGENT_DISPLAY_NAME")
AGENTSPACE_ID = os.getenv("AGENTSPACE_ID")
AGENT_DESCRIPTION = "Audits RFP documents for Cymbal Corp., ensuring compliance with established writing guidelines, regulatory requirements and general document accuracy"
TOOL_DESCRIPTION = "The agent audits RFP documents for Cymbal Corp., ensuring compliance with established writing guidelines, regulatory requirements and general document accuracy"

# Deployment Configuration Constants
AGENT_ENGINE_REQUIREMENTS = [
    "google-adk==1.15.1",
    "google-genai==1.39.1",
    "google-cloud-aiplatform[agent_engines]==1.109.0",
]
AGENT_ENGINE_EXTRA_PACKAGES = ["auditor_agent"]


class AgentSpaceDeployer:
    """
    A class to deploy ADK agents to Vertex AI Agent Engine and AgentSpace.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        project_number: Optional[str] = None,
        location: Optional[str] = None,
        staging_bucket: Optional[str] = None,
    ):
        """
        Initialize the AgentSpaceDeployer.

        Args:
            project_id: Google Cloud project ID (defaults to GCP_PROJECT_ID from environment)
            project_number: Google Cloud project number (defaults to GCP_PROJECT_NUMBER from environment)
            location: Vertex AI location (defaults to GCP_LOCATION from environment)
            staging_bucket: GCS staging bucket for Vertex AI
        """
        # Use environment variables if parameters not provided
        if project_id is None:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if project_id is None:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set and no project_id provided")

        if project_number is None:
            project_number = os.getenv("GCP_PROJECT_NUMBER")
            if project_number is None:
                raise ValueError("GCP_PROJECT_NUMBER environment variable not set and no project_number provided")

        if location is None:
            location = os.getenv("GOOGLE_CLOUD_LOCATION")
            if location is None:
                raise ValueError("GOOGLE_CLOUD_LOCATION environment variable not set and no location provided")

        if staging_bucket is None:
            staging_bucket = os.getenv("GCP_BUCKET_NAME")
            if staging_bucket is None:
                # Default staging bucket pattern as fallback
                staging_bucket = f"gs://{project_id}-staging"

        # Ensure staging bucket starts with gs://
        if staging_bucket and not staging_bucket.startswith("gs://"):
            staging_bucket = f"gs://{staging_bucket}"

        self.project_id = project_id
        self.project_number = project_number
        self.location = location
        self.staging_bucket = staging_bucket
        self.logger = get_logger(__name__)

        # ----------------------------------------------------------------------
        # NEW: Get the AGENTSPACE_PROJECT ID
        # ----------------------------------------------------------------------
        self.agentspace_project_id = os.getenv("AGENTSPACE_PROJECT")
        if self.agentspace_project_id is None:
            # If AGENTSPACE_PROJECT is not explicitly set, fall back to the main project_id.
            self.agentspace_project_id = self.project_id
            self.logger.warning("AGENTSPACE_PROJECT environment variable not found. Using GOOGLE_CLOUD_PROJECT/project_id as fallback.")
        # ----------------------------------------------------------------------
        
        # Initialize Vertex AI
        vertexai.init(
            project=self.project_id,
            location=self.location,
            staging_bucket=self.staging_bucket,
        )

        # Setup authentication
        creds, _ = default()
        self.authed_session = AuthorizedSession(creds)
        self.header = {"X-Goog-User-Project": self.project_id, "Content-Type": "application/json"}

        self.logger.info(f"Initialized AgentSpaceDeployer with project: {project_id}")

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

    def test_app_locally(self, app: reasoning_engines.AdkApp) -> None:
        """
        Test the ADK application locally.

        Args:
            app: The ADK application to test
        """
        try:
            self.logger.info("Testing ADK application locally")

            # Create a test session
            session = app.create_session(user_id="test_user")
            self.logger.info(f"Created test session: {session.id}")

            # Test with a simple query
            agent_context = {
                "message": {"role": "user", "parts": [{"text": "How were you built?"}]},
                "events": [
                    {
                        "content": {"role": "user", "parts": [{"text": "how were you built ?"}]},
                        "author": "AgentSpace_root_agent",
                    }
                ],
            }

            self.logger.info("Running test query")
            for response in app.streaming_agent_run_with_events(json.dumps(agent_context)):
                for event in response.get("events", []):
                    self.logger.debug(f"Event: {event}")

            self.logger.info("Local testing completed successfully")

        except Exception as e:
            self.logger.error(f"Error testing app locally: {str(e)}")
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
                display_name=AGENT_ENGINE_DISPLAY_NAME,
                requirements=AGENT_ENGINE_REQUIREMENTS,
                extra_packages=AGENT_ENGINE_EXTRA_PACKAGES,
            )

            self.logger.info(f"Successfully deployed to Agent Engine: {remote_app.resource_name}")

            # Test the remote deployment
            self.logger.info("Testing remote deployment")
            for event in remote_app.stream_query(
                user_id="test_user",
                message="Test query for remote deployment",
            ):
                self.logger.debug(f"Remote event: {event}")

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

            # Agent configuration
            agent_config = {
                "displayName": AGENT_DISPLAY_NAME,
                "description": AGENT_DESCRIPTION,
                "adk_agent_definition": {
                    "tool_settings": {"tool_description": TOOL_DESCRIPTION},
                    "provisioned_reasoning_engine": {"reasoning_engine": agent_engine.resource_name},
                },
            }

            # ----------------------------------------------------------------------
            # MODIFIED: Using self.agentspace_project_id instead of self.project_id
            # ----------------------------------------------------------------------
            discovery_engine_url = f"https://discoveryengine.googleapis.com/v1alpha/projects/{self.agentspace_project_id}/locations/global/collections/default_collection/engines/{agentspace_id}/assistants/default_assistant/agents"

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

            # ----------------------------------------------------------------------
            # MODIFIED: Using self.agentspace_project_id instead of self.project_id
            # ----------------------------------------------------------------------
            discovery_engine_url = f"https://discoveryengine.googleapis.com/v1alpha/projects/{self.agentspace_project_id}/locations/global/collections/default_collection/engines/{agentspace_id}/assistants/default_assistant/agents/"

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

    def deploy_complete_pipeline(self, agentspace_id: str) -> dict:
        """
        Execute the complete deployment pipeline.

        Args:
            agentspace_id: The AgentSpace ID to deploy to

        Returns:
            Dictionary with deployment results
        """
        try:
            self.logger.info("Starting complete deployment pipeline")

            # Step 1: Create ADK app
            app = self.create_adk_app()

            # Step 2: Test locally
            #self.test_app_locally(app)

            # Step 3: Deploy to Agent Engine
            remote_app = self.deploy_to_agent_engine(app)

            # Step 4: Deploy to AgentSpace
            agentspace_result = self.deploy_to_agentspace(remote_app, agentspace_id)

            # Step 5: List agents to confirm deployment
            agents_list = self.list_agentspace_agents(agentspace_id)

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
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Deploy ADK agent to AgentSpace")
    parser.add_argument(
        "--agentspace-id",
        "-a",
        type=str,
        help="AgentSpace ID to deploy the agent to (defaults to AGENTSPACE_ID from environment)",
    )
    parser.add_argument(
        "--project-id",
        "-p",
        type=str,
        help="Google Cloud project ID (defaults to GCP_PROJECT_ID from environment)",
    )
    parser.add_argument(
        "--project-number",
        "-n",
        type=str,
        help="Google Cloud project number (defaults to GCP_PROJECT_NUMBER from environment)",
    )
    parser.add_argument(
        "--location",
        "-l",
        type=str,
        help="Vertex AI location (defaults to GCP_LOCATION from environment)",
    )
    parser.add_argument(
        "--staging-bucket",
        "-s",
        type=str,
        help="GCS staging bucket for Vertex AI (defaults to GCP_BUCKET_NAME from environment)",
    )

    args = parser.parse_args()

    # Setup logging configuration
    setup_logging()

    # Get logger for main function
    logger = get_logger(__name__)

    try:
        # Get agentspace ID from args or environment
        agentspace_id = args.agentspace_id
        if agentspace_id is None:
            agentspace_id = os.getenv("AGENTSPACE_ID")
            if agentspace_id is None:
                raise ValueError(
                    "AgentSpace ID must be provided via --agentspace-id argument or AGENTSPACE_ID environment variable"
                )

        # Initialize the deployer
        deployer = AgentSpaceDeployer(
            project_id=args.project_id,
            project_number=args.project_number,
            location=args.location,
            staging_bucket=args.staging_bucket,
        )

        # Execute deployment pipeline
        results = deployer.deploy_complete_pipeline(agentspace_id)

        logger.info("Deployment completed successfully!")
        logger.info(f"Reasoning Engine: {results['reasoning_engine']}")
        logger.info(f"AgentSpace Deployment: {results['agentspace_deployment']}")

        # Print important notes
        logger.info("\n" + "=" * 80)
        logger.info("IMPORTANT DEPLOYMENT NOTES:")
        logger.info("=" * 80)
        logger.info("1. Ensure that Vertex AI API and Discovery Engine API are enabled in your project")
        logger.info("1. Get your project allowlisted to be able to attach ADK agent to agentspace")
        logger.info("2. Verify that Discovery Engine service account has 'Vertex AI User' permission")
        logger.info("3. Check 'Include Google-provided role grants' in IAM if you cannot find the service account")
        logger.info("4. Open AgentSpace app in GCP console -> Integration -> Use provided URL")
        logger.info("5. Agent display names must not contain spaces for proper functionality")
        logger.info(
            "6. Grant necessary permissions to service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
        )
        logger.info("7. Use Trace Explorer for debugging any issues with agent responses")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
