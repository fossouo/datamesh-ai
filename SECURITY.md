# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email us at: security@datamesh-ai.io (or open a private security advisory on GitHub)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

### Disclosure Policy

- We will coordinate disclosure with you
- Credit will be given to reporters (unless anonymity is requested)
- We follow responsible disclosure practices

## Security Best Practices

When using DataMesh-AI:

1. **Never commit credentials** - Use environment variables or secret managers
2. **Use least privilege** - Configure minimal data access in agent contracts
3. **Enable audit logging** - Track all agent actions
4. **Review agent contracts** - Understand what each agent can access
5. **Keep dependencies updated** - Run `pip audit` regularly

## Known Security Considerations

### Data Access

- Agents can only access data explicitly allowed in their contracts
- Default policy is `deny` - access must be explicitly granted
- PII and sensitive data require additional governance controls

### LLM Considerations

- User inputs are sent to LLM providers (if using cloud LLMs)
- Consider on-premise LLM deployment for sensitive environments
- Agent outputs should be validated before execution

### Network Security

- Use TLS for all agent-to-agent communication
- Authenticate agents using mTLS in production
- Isolate agent networks where possible
