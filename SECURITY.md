# Security Policy

## Supported Versions

Only the latest release is supported with security updates.

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities.

2. **Use GitHub's private vulnerability reporting**:
   - Go to the [Security Advisories](https://github.com/llm-works/llm-infer/security/advisories/new) page
   - Click "Report a vulnerability"
   - Include: description, steps to reproduce, potential impact, and any suggested fixes

3. **Expected response time**: We aim to acknowledge reports within 48 hours and provide a more
   detailed response within 7 days.

4. **Disclosure timeline**: We request a 90-day disclosure window to address vulnerabilities before
   public disclosure.

## Security Considerations

### API Security

- The inference server binds to `0.0.0.0` by default. In production, use a reverse proxy with
  authentication.
- No built-in authentication or rate limiting. Implement these at the infrastructure level.

### Model Security

- Only load models from trusted sources.
- Model weights can contain arbitrary code in some formats. Use `safetensors` format when possible.

### Input Validation

- Prompts are validated for length limits.
- API inputs are validated via Pydantic schemas.

## Security Best Practices for Deployment

1. Run behind a reverse proxy (nginx, Caddy, etc.) with TLS
2. Implement authentication at the proxy level
3. Set appropriate resource limits (memory, GPU)
4. Monitor for unusual usage patterns
5. Keep dependencies updated
