# DataMesh-AI Governance Abstraction Layer

## Overview

DataMesh-AI provides a **unified governance interface** that abstracts the underlying cloud provider's governance solution. When a user configures their cloud credentials, the system automatically discovers and connects to the appropriate governance service.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATAMESH-AI GOVERNANCE LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User: "What tables contain PII that I can access?"                         │
│                            │                                                │
│                            ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              UNIFIED GOVERNANCE INTERFACE                            │   │
│  │                                                                      │   │
│  │  • get_accessible_resources(user_context)                           │   │
│  │  • get_classifications(resource)                                    │   │
│  │  • check_permission(user, resource, action)                         │   │
│  │  • get_data_policies(resource)                                      │   │
│  │  • get_masking_rules(resource, user_context)                        │   │
│  │  • audit_access(user, resource, action, result)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                            │                                                │
│         ┌──────────────────┼──────────────────┬──────────────────┐         │
│         ▼                  ▼                  ▼                  ▼         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │    AWS      │    │   Azure     │    │    GCP      │    │ Databricks  │ │
│  │   Adapter   │    │   Adapter   │    │   Adapter   │    │   Adapter   │ │
│  │             │    │             │    │             │    │             │ │
│  │ Lake Form.  │    │  Purview    │    │  Dataplex   │    │Unity Catalog│ │
│  │ IAM/Glue    │    │  IAM/RBAC   │    │  IAM/DLP    │    │  + ACLs     │ │
│  │ Macie       │    │  Defender   │    │  SCC        │    │             │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Basic Setup

When configuring DataMesh-AI, the governance layer auto-discovers available services based on credentials:

```yaml
# datamesh.config.yaml
apiVersion: datamesh.ai/v1
kind: Config

cloud:
  provider: aws  # aws | azure | gcp | databricks | multi-cloud

  # Credentials (use environment variables or secret manager)
  credentials:
    # Option 1: Environment variables (recommended)
    type: environment

    # Option 2: AWS Profile
    # type: aws-profile
    # profile: my-profile

    # Option 3: Service Account (GCP)
    # type: service-account
    # keyFile: ${GOOGLE_APPLICATION_CREDENTIALS}

governance:
  # Auto-discover governance services based on credentials
  autoDiscover: true

  # Or explicitly configure
  # services:
  #   - type: lake-formation
  #   - type: glue-catalog
  #   - type: macie
```

### Provider-Specific Configuration

#### AWS

```yaml
cloud:
  provider: aws
  region: eu-west-1

governance:
  services:
    - type: lake-formation
      enabled: true
      # User's permissions are automatically derived from their IAM role

    - type: glue-catalog
      enabled: true
      databases:
        - finance
        - marketing

    - type: macie
      enabled: true
      # For PII/sensitive data discovery

    - type: iam
      enabled: true
      # For permission checks
```

#### Azure

```yaml
cloud:
  provider: azure
  subscription: ${AZURE_SUBSCRIPTION_ID}

governance:
  services:
    - type: purview
      enabled: true
      accountName: my-purview-account

    - type: defender-for-cloud
      enabled: true
      # For data classification

    - type: azure-rbac
      enabled: true
```

#### GCP

```yaml
cloud:
  provider: gcp
  project: ${GCP_PROJECT_ID}

governance:
  services:
    - type: dataplex
      enabled: true
      location: us-central1

    - type: dlp
      enabled: true
      # For sensitive data discovery

    - type: data-catalog
      enabled: true

    - type: iam
      enabled: true
```

#### Databricks

```yaml
cloud:
  provider: databricks
  workspace: ${DATABRICKS_HOST}

governance:
  services:
    - type: unity-catalog
      enabled: true
      metastore: my-metastore

    - type: workspace-acls
      enabled: true
```

#### Multi-Cloud

```yaml
cloud:
  provider: multi-cloud

governance:
  services:
    # AWS resources
    - type: lake-formation
      cloudContext: aws
      region: eu-west-1

    # Azure resources
    - type: purview
      cloudContext: azure
      subscription: ${AZURE_SUBSCRIPTION_ID}

    # Unified view across clouds
  unifiedCatalog:
    enabled: true
    conflictResolution: most-restrictive  # Apply strictest policy
```

---

## Unified Governance Interface

### GovernanceProvider Interface

```python
# packages/core/governance/interface.py

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class UserContext:
    """User identity and attributes from cloud provider."""
    user_id: str
    provider: str  # aws, azure, gcp, databricks

    # Provider-specific identity
    aws_arn: Optional[str] = None
    azure_oid: Optional[str] = None
    gcp_email: Optional[str] = None
    databricks_user: Optional[str] = None

    # Resolved attributes
    groups: List[str] = None
    roles: List[str] = None
    tags: dict = None

@dataclass
class Resource:
    """A data resource (table, view, file, etc.)."""
    uri: str  # catalog://database.schema.table
    provider: str
    type: str  # table, view, file, dataset

    # Provider-specific identifiers
    aws_arn: Optional[str] = None
    azure_resource_id: Optional[str] = None
    gcp_resource_name: Optional[str] = None

@dataclass
class Permission:
    """Permission check result."""
    allowed: bool
    resource: Resource
    action: str

    # Details
    reason: Optional[str] = None
    granted_by: Optional[str] = None  # Policy/role that granted access
    conditions: Optional[dict] = None  # Row filters, column masks, etc.
    expires_at: Optional[str] = None

@dataclass
class Classification:
    """Data classification tag."""
    name: str  # PII, SENSITIVE, CONFIDENTIAL, PUBLIC
    confidence: float  # 0.0 - 1.0
    source: str  # macie, purview, dlp, manual

    # Scope
    applies_to: str  # table, column, row
    column_name: Optional[str] = None

    # Metadata
    detected_at: Optional[str] = None
    verified: bool = False

@dataclass
class MaskingRule:
    """Column masking rule."""
    column: str
    method: str  # full, partial, hash, tokenize, null

    # Conditions
    applies_when: Optional[str] = None  # Expression
    exempt_roles: List[str] = None

@dataclass
class RowFilter:
    """Row-level security filter."""
    expression: str  # SQL WHERE clause
    reason: str


class GovernanceProvider(ABC):
    """Abstract interface for governance providers."""

    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """Initialize the provider with configuration."""
        pass

    @abstractmethod
    async def get_user_context(self, credentials: dict) -> UserContext:
        """
        Get user context from cloud credentials.

        This extracts the user's identity, groups, roles, and attributes
        from the provided credentials.
        """
        pass

    @abstractmethod
    async def get_accessible_resources(
        self,
        user: UserContext,
        resource_type: str = None,
        catalog: str = None,
    ) -> List[Resource]:
        """
        Get all resources the user can access.

        Returns only resources where the user has at least read permission.
        """
        pass

    @abstractmethod
    async def check_permission(
        self,
        user: UserContext,
        resource: Resource,
        action: str,  # read, write, delete, admin
    ) -> Permission:
        """
        Check if user has permission for an action on a resource.

        Returns detailed permission info including conditions.
        """
        pass

    @abstractmethod
    async def get_classifications(
        self,
        resource: Resource,
        user: UserContext = None,
    ) -> List[Classification]:
        """
        Get data classifications for a resource.

        If user is provided, only returns classifications visible to them.
        """
        pass

    @abstractmethod
    async def get_masking_rules(
        self,
        resource: Resource,
        user: UserContext,
    ) -> List[MaskingRule]:
        """
        Get column masking rules that apply for this user.
        """
        pass

    @abstractmethod
    async def get_row_filters(
        self,
        resource: Resource,
        user: UserContext,
    ) -> List[RowFilter]:
        """
        Get row-level security filters that apply for this user.
        """
        pass

    @abstractmethod
    async def audit_access(
        self,
        user: UserContext,
        resource: Resource,
        action: str,
        result: str,  # allowed, denied, masked
        details: dict = None,
    ) -> None:
        """
        Log an access event for audit purposes.
        """
        pass
```

---

## Provider Implementations

### AWS Governance Adapter

```python
# packages/connectors/aws-governance/adapter.py

import boto3
from datamesh_ai_core.governance import GovernanceProvider, UserContext, Resource

class AWSGovernanceAdapter(GovernanceProvider):
    """AWS governance using Lake Formation, IAM, Glue, and Macie."""

    def __init__(self):
        self.lf_client = None
        self.glue_client = None
        self.iam_client = None
        self.sts_client = None
        self.macie_client = None

    async def initialize(self, config: dict) -> None:
        session = boto3.Session(
            region_name=config.get('region', 'us-east-1'),
            profile_name=config.get('profile'),
        )

        self.lf_client = session.client('lakeformation')
        self.glue_client = session.client('glue')
        self.iam_client = session.client('iam')
        self.sts_client = session.client('sts')

        if config.get('macie_enabled', True):
            self.macie_client = session.client('macie2')

    async def get_user_context(self, credentials: dict) -> UserContext:
        # Get caller identity
        identity = self.sts_client.get_caller_identity()

        # Extract user/role info from ARN
        arn = identity['Arn']

        # Get IAM groups/roles
        groups = []
        roles = []
        tags = {}

        if ':user/' in arn:
            user_name = arn.split(':user/')[-1]
            user_groups = self.iam_client.list_groups_for_user(UserName=user_name)
            groups = [g['GroupName'] for g in user_groups['Groups']]

            user_tags = self.iam_client.list_user_tags(UserName=user_name)
            tags = {t['Key']: t['Value'] for t in user_tags['Tags']}

        elif ':assumed-role/' in arn:
            role_name = arn.split(':assumed-role/')[-1].split('/')[0]
            roles = [role_name]

            role_tags = self.iam_client.list_role_tags(RoleName=role_name)
            tags = {t['Key']: t['Value'] for t in role_tags['Tags']}

        return UserContext(
            user_id=identity['UserId'],
            provider='aws',
            aws_arn=arn,
            groups=groups,
            roles=roles,
            tags=tags,
        )

    async def get_accessible_resources(
        self,
        user: UserContext,
        resource_type: str = None,
        catalog: str = None,
    ) -> List[Resource]:
        resources = []

        # Get Lake Formation permissions for this principal
        paginator = self.lf_client.get_paginator('list_permissions')

        for page in paginator.paginate(Principal={'DataLakePrincipalIdentifier': user.aws_arn}):
            for perm in page['PrincipalResourcePermissions']:
                resource = perm.get('Resource', {})

                # Table permission
                if 'Table' in resource:
                    table = resource['Table']
                    resources.append(Resource(
                        uri=f"catalog://{table['DatabaseName']}.{table['Name']}",
                        provider='aws',
                        type='table',
                        aws_arn=f"arn:aws:glue:{self.region}:{self.account}:table/{table['DatabaseName']}/{table['Name']}",
                    ))

                # Database permission (all tables)
                elif 'Database' in resource:
                    db = resource['Database']
                    tables = self.glue_client.get_tables(DatabaseName=db['Name'])
                    for t in tables['TableList']:
                        resources.append(Resource(
                            uri=f"catalog://{db['Name']}.{t['Name']}",
                            provider='aws',
                            type='table',
                            aws_arn=f"arn:aws:glue:{self.region}:{self.account}:table/{db['Name']}/{t['Name']}",
                        ))

        return resources

    async def check_permission(
        self,
        user: UserContext,
        resource: Resource,
        action: str,
    ) -> Permission:
        # Map action to Lake Formation permission
        lf_action_map = {
            'read': 'SELECT',
            'write': 'INSERT',
            'delete': 'DELETE',
            'admin': 'ALL',
        }
        lf_action = lf_action_map.get(action, 'SELECT')

        # Parse resource URI
        parts = resource.uri.replace('catalog://', '').split('.')
        database = parts[0]
        table = parts[1] if len(parts) > 1 else None

        # Check Lake Formation permissions
        try:
            result = self.lf_client.get_effective_permissions_for_path(
                CatalogId=self.account,
                ResourceArn=resource.aws_arn,
            )

            for perm in result['Permissions']:
                if user.aws_arn in perm['Principal']['DataLakePrincipalIdentifier']:
                    if lf_action in perm['Permissions']:
                        return Permission(
                            allowed=True,
                            resource=resource,
                            action=action,
                            granted_by=f"Lake Formation: {perm.get('PermissionsWithGrantOption', [])}",
                        )

            return Permission(
                allowed=False,
                resource=resource,
                action=action,
                reason="No Lake Formation permission found",
            )

        except Exception as e:
            return Permission(
                allowed=False,
                resource=resource,
                action=action,
                reason=str(e),
            )

    async def get_classifications(
        self,
        resource: Resource,
        user: UserContext = None,
    ) -> List[Classification]:
        classifications = []

        # Get Glue table with column classifications
        parts = resource.uri.replace('catalog://', '').split('.')
        database, table = parts[0], parts[1]

        glue_table = self.glue_client.get_table(DatabaseName=database, Name=table)

        # Check column parameters for classifications
        for col in glue_table['Table'].get('StorageDescriptor', {}).get('Columns', []):
            params = col.get('Parameters', {})
            if 'classification' in params:
                classifications.append(Classification(
                    name=params['classification'],
                    confidence=1.0,
                    source='glue-catalog',
                    applies_to='column',
                    column_name=col['Name'],
                    verified=True,
                ))

        # Get Macie findings if available
        if self.macie_client:
            try:
                findings = self.macie_client.list_findings(
                    findingCriteria={
                        'criterion': {
                            'resourcesAffected.s3Bucket.name': {
                                'eq': [self._get_s3_location(glue_table)]
                            }
                        }
                    }
                )

                for finding_id in findings.get('findingIds', []):
                    finding = self.macie_client.get_findings(findingIds=[finding_id])
                    for f in finding['findings']:
                        classifications.append(Classification(
                            name=f['type'],  # e.g., SensitiveData:S3Object/Personal
                            confidence=f.get('severity', {}).get('score', 0) / 100,
                            source='macie',
                            applies_to='table',
                            detected_at=f.get('createdAt'),
                        ))
            except Exception:
                pass  # Macie not available or no findings

        return classifications

    async def get_masking_rules(
        self,
        resource: Resource,
        user: UserContext,
    ) -> List[MaskingRule]:
        rules = []

        # Check Lake Formation column-level permissions
        # If user doesn't have access to a column, mask it
        parts = resource.uri.replace('catalog://', '').split('.')
        database, table = parts[0], parts[1]

        try:
            col_perms = self.lf_client.get_effective_permissions_for_path(
                CatalogId=self.account,
                ResourceArn=resource.aws_arn,
            )

            # Get all columns
            glue_table = self.glue_client.get_table(DatabaseName=database, Name=table)
            all_columns = [c['Name'] for c in glue_table['Table']['StorageDescriptor']['Columns']]

            # Find columns user can access
            accessible_columns = set()
            for perm in col_perms['Permissions']:
                if 'ColumnWildcard' in perm.get('Resource', {}).get('TableWithColumns', {}):
                    accessible_columns = set(all_columns)
                    break
                columns = perm.get('Resource', {}).get('TableWithColumns', {}).get('ColumnNames', [])
                accessible_columns.update(columns)

            # Mask inaccessible columns
            for col in all_columns:
                if col not in accessible_columns:
                    rules.append(MaskingRule(
                        column=col,
                        method='null',
                    ))
        except Exception:
            pass

        return rules

    async def get_row_filters(
        self,
        resource: Resource,
        user: UserContext,
    ) -> List[RowFilter]:
        filters = []

        # Check Lake Formation row-level security
        parts = resource.uri.replace('catalog://', '').split('.')
        database, table = parts[0], parts[1]

        try:
            data_cells_filter = self.lf_client.list_data_cells_filter(
                Table={
                    'CatalogId': self.account,
                    'DatabaseName': database,
                    'Name': table,
                }
            )

            for dcf in data_cells_filter.get('DataCellsFilters', []):
                # Check if this filter applies to user
                if self._filter_applies_to_user(dcf, user):
                    if dcf.get('RowFilter', {}).get('FilterExpression'):
                        filters.append(RowFilter(
                            expression=dcf['RowFilter']['FilterExpression'],
                            reason=f"Lake Formation Data Cell Filter: {dcf['Name']}",
                        ))
        except Exception:
            pass

        return filters

    async def audit_access(
        self,
        user: UserContext,
        resource: Resource,
        action: str,
        result: str,
        details: dict = None,
    ) -> None:
        # Log to CloudTrail via Lake Formation
        # This happens automatically for Lake Formation operations
        # Additional custom logging can be added here
        pass
```

---

## Usage in Agents

### Governance-Aware SQL Agent

```python
# Example: SQL Agent using governance layer

class SQLAgent:
    def __init__(self, governance: GovernanceProvider):
        self.governance = governance

    async def generate_query(
        self,
        request: str,
        user: UserContext,
    ) -> QueryResult:
        # 1. Get accessible resources
        resources = await self.governance.get_accessible_resources(user)

        # 2. Generate SQL based on accessible tables only
        accessible_tables = [r.uri for r in resources]
        sql = await self.llm.generate_sql(request, accessible_tables)

        # 3. Check specific permissions for tables in query
        tables_in_query = self.extract_tables(sql)
        for table in tables_in_query:
            resource = Resource(uri=f"catalog://{table}", provider=user.provider)
            perm = await self.governance.check_permission(user, resource, 'read')

            if not perm.allowed:
                raise PermissionDeniedError(f"No access to {table}: {perm.reason}")

        # 4. Apply masking rules
        for table in tables_in_query:
            resource = Resource(uri=f"catalog://{table}", provider=user.provider)
            masks = await self.governance.get_masking_rules(resource, user)
            sql = self.apply_masks(sql, masks)

        # 5. Apply row filters
        for table in tables_in_query:
            resource = Resource(uri=f"catalog://{table}", provider=user.provider)
            filters = await self.governance.get_row_filters(resource, user)
            sql = self.apply_filters(sql, table, filters)

        # 6. Execute and audit
        result = await self.execute(sql)

        for table in tables_in_query:
            resource = Resource(uri=f"catalog://{table}", provider=user.provider)
            await self.governance.audit_access(
                user, resource, 'read', 'allowed',
                details={'query': sql, 'rows_returned': result.row_count}
            )

        return result
```

---

## Classification Taxonomy

DataMesh-AI uses a unified classification taxonomy that maps to provider-specific classifications:

| DataMesh-AI | AWS Macie | Azure Purview | GCP DLP |
|-------------|-----------|---------------|---------|
| `PII` | SensitiveData:Personal | Microsoft.Personal | PERSON_NAME, EMAIL |
| `PII.NAME` | SensitiveData:Personal/Name | Microsoft.Personal.Name | PERSON_NAME |
| `PII.EMAIL` | SensitiveData:Personal/Email | Microsoft.Personal.Email | EMAIL_ADDRESS |
| `PII.PHONE` | SensitiveData:Personal/Phone | Microsoft.Personal.Phone | PHONE_NUMBER |
| `PII.SSN` | SensitiveData:Personal/SSN | Microsoft.Government.USSocialSecurityNumber | US_SOCIAL_SECURITY_NUMBER |
| `FINANCIAL` | SensitiveData:Financial | Microsoft.Financial | CREDIT_CARD_NUMBER |
| `HEALTH` | SensitiveData:Medical | Microsoft.Medical | MEDICAL_RECORD_NUMBER |
| `CONFIDENTIAL` | (custom tag) | Microsoft.Confidential | (custom tag) |
| `SECRET` | (custom tag) | Microsoft.Secret | (custom tag) |
| `PUBLIC` | (custom tag) | Microsoft.Public | (custom tag) |

---

## Next Steps

1. **Implement Azure Adapter** — Purview, Defender for Cloud, Azure RBAC
2. **Implement GCP Adapter** — Dataplex, DLP, Data Catalog, IAM
3. **Implement Databricks Adapter** — Unity Catalog, Workspace ACLs
4. **Add OpenMetadata Adapter** — For on-premise/vendor-neutral governance
5. **Build Policy Sync** — Sync policies between providers in multi-cloud

---

*Specification v1.0 — DataMesh-AI Governance Abstraction*
