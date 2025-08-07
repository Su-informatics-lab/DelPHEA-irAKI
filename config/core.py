"""
Core configuration classes for DelPHEA-irAKI system.

This module contains all configuration dataclasses including infrastructure,
runtime, and Delphi methodology parameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InfrastructureConfig:
    """Infrastructure-specific configuration for different deployments."""

    # endpoint types
    ENDPOINT_AWS = "aws"
    ENDPOINT_TEMPEST = "tempest"
    ENDPOINT_LOCAL = "local"

    endpoint_type: str = ENDPOINT_AWS

    # aws configuration
    aws_endpoint: str = "http://172.31.11.192:8000"

    # tempest hpc configuration
    tempest_host: str = "tempest-gpu021"
    tempest_port: int = 8000
    tempest_login_node: str = "tempest-login.msu.montana.edu"
    tempest_username: Optional[str] = None

    # local development
    local_endpoint: str = "http://localhost:8000"

    # ssh tunnel configuration (for JetStream2 -> Tempest)
    use_ssh_tunnel: bool = False
    ssh_tunnel_local_port: int = 8001

    def get_endpoint(self) -> str:
        """Get the appropriate endpoint based on configuration.

        Returns:
            str: The vLLM endpoint URL

        Raises:
            ValueError: If endpoint type is invalid
        """
        if self.endpoint_type == self.ENDPOINT_AWS:
            return self.aws_endpoint
        elif self.endpoint_type == self.ENDPOINT_TEMPEST:
            if self.use_ssh_tunnel:
                # use local forwarded port from JetStream2
                return f"http://localhost:{self.ssh_tunnel_local_port}"
            else:
                # direct connection within Tempest network
                return f"http://{self.tempest_host}:{self.tempest_port}"
        elif self.endpoint_type == self.ENDPOINT_LOCAL:
            return self.local_endpoint
        else:
            raise ValueError(
                f"Invalid endpoint type: {self.endpoint_type}. "
                f"Must be one of: {self.ENDPOINT_AWS}, {self.ENDPOINT_TEMPEST}, {self.ENDPOINT_LOCAL}"
            )

    def get_ssh_tunnel_command(self) -> str:
        """Generate SSH tunnel command for Tempest access from JetStream2.

        Returns:
            str: SSH command for port forwarding
        """
        if not self.tempest_username:
            raise ValueError("Tempest username required for SSH tunnel")

        return (
            f"ssh -N -L {self.ssh_tunnel_local_port}:"
            f"{self.tempest_host}:{self.tempest_port} "
            f"{self.tempest_username}@{self.tempest_login_node}"
        )


@dataclass
class RuntimeConfig:
    """Enhanced runtime configuration with infrastructure support."""

    # infrastructure selection
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)

    # model configuration
    model_name: str = "openai/gpt-oss-120b"
    api_key: Optional[str] = None

    # inference parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 3072
    timeout: float = 120.0

    # configuration file paths
    expert_panel_config: str = "config/panel.json"
    questionnaire_config: str = "config/questionnaire.json"
    prompts_dir: str = "prompts"

    # data configuration
    data_dir: str = "irAKI_data"
    cache_dir: Optional[str] = None
    use_real_data: bool = True  # toggle between real and dummy data

    def get_vllm_endpoint(self) -> str:
        """Get the configured vLLM endpoint."""
        return self.infrastructure.get_endpoint()


@dataclass
class DelphiConfig:
    """Delphi methodology parameters."""

    conflict_threshold: int = 3
    max_debate_rounds: int = 6
    max_debate_participants: int = 8
    expert_count: int = 8

    # consensus configuration
    consensus_threshold: float = 0.8
    confidence_threshold: float = 0.7

    # timeouts (seconds)
    round1_timeout: int = 900
    round3_timeout: int = 600
    debate_timeout: int = 240

    # export configuration
    export_full_transcripts: bool = True
    include_reasoning_chains: bool = True
