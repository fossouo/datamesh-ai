"""
Agent contract loader and validator.

This module provides utilities for loading and validating agent contracts
defined in YAML format.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from jsonschema import Draft7Validator, ValidationError
from pydantic import BaseModel, Field

from datamesh_ai_core.models import AgentCapability, AgentMetadata


# JSON Schema for validating agent.yaml contracts
AGENT_CONTRACT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["agent_id", "name", "version"],
    "properties": {
        "agent_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 255,
            "pattern": "^[a-zA-Z][a-zA-Z0-9_-]*$",
            "description": "Unique agent identifier",
        },
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 255,
            "description": "Human-readable agent name",
        },
        "version": {
            "type": "string",
            "pattern": "^\\d+\\.\\d+\\.\\d+",
            "description": "Semantic version",
        },
        "description": {
            "type": "string",
            "description": "Agent description",
        },
        "owner": {
            "type": "string",
            "description": "Team or individual responsible",
        },
        "endpoint": {
            "type": "string",
            "format": "uri",
            "description": "HTTP endpoint for A2A communication",
        },
        "capabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 255,
                    },
                    "version": {
                        "type": "string",
                        "pattern": "^\\d+\\.\\d+\\.\\d+",
                    },
                    "description": {
                        "type": "string",
                    },
                    "input_schema": {
                        "type": "object",
                    },
                    "output_schema": {
                        "type": "object",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "metadata": {
            "type": "object",
            "additionalProperties": True,
        },
        "dependencies": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["agent_id"],
                "properties": {
                    "agent_id": {"type": "string"},
                    "capability": {"type": "string"},
                    "required": {"type": "boolean", "default": True},
                },
            },
        },
        "config": {
            "type": "object",
            "additionalProperties": True,
            "description": "Agent-specific configuration",
        },
    },
}


class ContractValidationError(Exception):
    """Raised when contract validation fails."""

    def __init__(self, message: str, errors: List[str]):
        """
        Initialize validation error.

        Args:
            message: Error summary
            errors: List of specific validation errors
        """
        super().__init__(message)
        self.errors = errors


class AgentDependency(BaseModel):
    """Represents a dependency on another agent."""

    agent_id: str = Field(..., description="ID of the required agent")
    capability: Optional[str] = Field(
        default=None,
        description="Specific capability required",
    )
    required: bool = Field(
        default=True,
        description="Whether this dependency is required",
    )


class AgentContract(BaseModel):
    """
    Full agent contract loaded from YAML.

    This represents the complete contract definition for an agent,
    including capabilities, dependencies, and configuration.
    """

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Semantic version")
    description: Optional[str] = Field(default=None)
    owner: Optional[str] = Field(default=None)
    endpoint: Optional[str] = Field(default=None)
    capabilities: List[AgentCapability] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[AgentDependency] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)

    def to_metadata(self) -> AgentMetadata:
        """
        Convert contract to AgentMetadata for registry.

        Returns:
            AgentMetadata instance
        """
        return AgentMetadata(
            agent_id=self.agent_id,
            name=self.name,
            version=self.version,
            description=self.description,
            owner=self.owner,
            capabilities=self.capabilities,
            endpoint=self.endpoint,
            tags=self.tags,
            metadata=self.metadata,
        )


class ContractValidator:
    """
    Validates agent contracts against the JSON Schema.

    This class provides schema validation for agent.yaml contracts
    to ensure they conform to the expected structure.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the validator.

        Args:
            schema: Custom JSON Schema (defaults to AGENT_CONTRACT_SCHEMA)
        """
        self.schema = schema or AGENT_CONTRACT_SCHEMA
        self._validator = Draft7Validator(self.schema)

    def validate(self, contract_data: Dict[str, Any]) -> List[str]:
        """
        Validate contract data against the schema.

        Args:
            contract_data: Contract data dictionary

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        for error in self._validator.iter_errors(contract_data):
            path = ".".join(str(p) for p in error.path) or "root"
            errors.append(f"{path}: {error.message}")
        return errors

    def is_valid(self, contract_data: Dict[str, Any]) -> bool:
        """
        Check if contract data is valid.

        Args:
            contract_data: Contract data dictionary

        Returns:
            True if valid, False otherwise
        """
        return self._validator.is_valid(contract_data)

    def validate_or_raise(self, contract_data: Dict[str, Any]) -> None:
        """
        Validate contract data, raising on errors.

        Args:
            contract_data: Contract data dictionary

        Raises:
            ContractValidationError: If validation fails
        """
        errors = self.validate(contract_data)
        if errors:
            raise ContractValidationError(
                f"Contract validation failed with {len(errors)} error(s)",
                errors,
            )


class ContractLoader:
    """
    Loads agent contracts from YAML files.

    This class provides utilities for loading, parsing, and validating
    agent contracts defined in agent.yaml files.
    """

    def __init__(
        self,
        validator: Optional[ContractValidator] = None,
        validate_on_load: bool = True,
    ):
        """
        Initialize the contract loader.

        Args:
            validator: Custom validator (creates default if None)
            validate_on_load: Whether to validate contracts on load
        """
        self.validator = validator or ContractValidator()
        self.validate_on_load = validate_on_load

    def load_from_file(self, file_path: Union[str, Path]) -> AgentContract:
        """
        Load a contract from a YAML file.

        Args:
            file_path: Path to the agent.yaml file

        Returns:
            Parsed AgentContract

        Raises:
            FileNotFoundError: If file doesn't exist
            ContractValidationError: If validation fails
            yaml.YAMLError: If YAML parsing fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Contract file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self.load_from_dict(data)

    def load_from_string(self, yaml_content: str) -> AgentContract:
        """
        Load a contract from a YAML string.

        Args:
            yaml_content: YAML content as string

        Returns:
            Parsed AgentContract

        Raises:
            ContractValidationError: If validation fails
            yaml.YAMLError: If YAML parsing fails
        """
        data = yaml.safe_load(yaml_content)
        return self.load_from_dict(data)

    def load_from_dict(self, data: Dict[str, Any]) -> AgentContract:
        """
        Load a contract from a dictionary.

        Args:
            data: Contract data dictionary

        Returns:
            Parsed AgentContract

        Raises:
            ContractValidationError: If validation fails
        """
        if self.validate_on_load:
            self.validator.validate_or_raise(data)

        # Parse capabilities
        capabilities = []
        for cap_data in data.get("capabilities", []):
            capabilities.append(AgentCapability(**cap_data))

        # Parse dependencies
        dependencies = []
        for dep_data in data.get("dependencies", []):
            dependencies.append(AgentDependency(**dep_data))

        return AgentContract(
            agent_id=data["agent_id"],
            name=data["name"],
            version=data["version"],
            description=data.get("description"),
            owner=data.get("owner"),
            endpoint=data.get("endpoint"),
            capabilities=capabilities,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            dependencies=dependencies,
            config=data.get("config", {}),
        )

    def discover_contracts(
        self,
        directory: Union[str, Path],
        filename: str = "agent.yaml",
        recursive: bool = True,
    ) -> List[AgentContract]:
        """
        Discover and load all contracts in a directory.

        Args:
            directory: Directory to search
            filename: Contract filename to look for
            recursive: Whether to search subdirectories

        Returns:
            List of loaded AgentContracts
        """
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        contracts = []
        pattern = f"**/{filename}" if recursive else filename

        for contract_file in path.glob(pattern):
            try:
                contract = self.load_from_file(contract_file)
                contracts.append(contract)
            except (ContractValidationError, yaml.YAMLError) as e:
                # Log error but continue loading other contracts
                import logging
                logging.warning(f"Failed to load contract {contract_file}: {e}")

        return contracts
