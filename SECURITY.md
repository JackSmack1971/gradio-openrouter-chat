# Security Policy

We follow GitHub's security advisory process to coordinate fixes, track mitigation status, and communicate resolution timelines. This policy describes how to report vulnerabilities, what versions receive fixes, and the guardrails already implemented in the chat application.

## Reporting a Vulnerability
- **Email:** security@openrouter.chat (monitored 24/7 by the core maintainers).
- **Preferred channel:** Submit a draft advisory via GitHub Security Advisories (`Security` tab → `Report a vulnerability`).
- **Do _not_ file public issues** for suspected vulnerabilities until we publish an advisory.
- Please include:
  - A clear description of the impact and root cause.
  - Steps to reproduce (proof-of-concept script or HTTP trace).
  - Affected configuration (model selection, environment, rate limit settings, etc.).
  - Any mitigation or workaround ideas.

### Coordinated Disclosure Timeline
| Phase | Target timeline |
| --- | --- |
| Acknowledge receipt | within 2 business days |
| Initial triage + CVSS assessment | within 5 business days |
| Mitigation or patch available | within 30 calendar days for high/critical issues (90 days otherwise) |
| Public advisory publication | within 7 days of deploying the fix or agreeing on a disclosure date with the reporter |

We will provide status updates at least weekly for critical issues and bi-weekly for others. If we cannot meet these timelines, we will notify the reporter with revised expectations.

## Supported Versions
| Version | Supported? |
| --- | --- |
| `main` branch | ✅ – receives security fixes and dependency updates |
| Tagged releases within the last 6 months | ✅ – fixes backported when feasible |
| Older releases | ❌ – please upgrade to the latest supported release |

## Vulnerability Handling & Remediation
1. Verify the report, reproduce the issue, and score it using CVSS.
2. Develop a patch on a private branch with targeted tests.
3. Run dependency and static security scans (Dependabot + `pip-audit`) before releasing.
4. Coordinate disclosure timing with the reporter and GitHub Security Advisories.
5. Publish a new release, notify impacted users, and document mitigations.

## Threat Model Summary
- **Rate limiting:** The `RateLimiter` enforces a per-IP token bucket to prevent brute-force or resource exhaustion attacks before requests reach model backends.【F:utils.py†L215-L238】【F:main.py†L163-L173】
- **Input sanitization:** Incoming chat messages are trimmed and control characters stripped via `sanitize_text`, ensuring untrusted text does not poison logs or downstream renderers.【F:utils.py†L200-L203】【F:main.py†L133-L159】
- **Persistence boundaries:** Conversation exports and usage analytics are stored locally via `export_conversation`, `log_usage`, and the Gradio handlers, keeping sensitive chat data scoped to controlled storage rather than remote services.【F:utils.py†L241-L264】【F:main.py†L248-L333】

These controls assume the application runs behind HTTPS termination and that infrastructure-level protections (WAF, reverse proxy) enforce additional throttling and authentication as needed.

## Secret Management & Dependency Security
- **Secrets:** API keys and operational toggles are sourced exclusively from environment variables loaded via `config.Settings`, preventing hard-coded credentials and supporting platform secret stores (.env files, GitHub Actions secrets, cloud vaults).【F:config.py†L12-L111】
- **Dependency hygiene:** The project uses `requirements.txt` for pinned dependencies and integrates with GitHub Dependabot / security advisories for timely updates. Before release, run `pip install -r requirements.txt` followed by `pip-audit` (or GitHub Advanced Security) to flag vulnerable packages.【F:requirements.txt†L1-L8】

## Additional Recommendations
- Rotate API keys regularly and restrict them to the minimum OpenRouter scopes required.
- Configure infrastructure monitoring to alert on repeated rate-limit hits, unexpected IP ranges, or export activity spikes.
- Engage in periodic third-party penetration testing, especially after major feature launches.

Thank you for helping keep the OpenRouter + Gradio Chat community secure!
