# AGENTS.md — AI Collaboration Guide

**Project**: OpenRouter + Gradio Chat  
**Purpose**: A production-ready streaming conversational interface over OpenRouter's unified API with smart routing, fallback logic, and cost optimization.

---

## 1. Build and Test Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\Activate.ps1     # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (REQUIRED)
pre-commit install

# Configure environment
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=your_key_here
```

### Development Commands
```bash
# Run application
python main.py                   # Serves on http://127.0.0.1:7860

# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py

# Run with coverage report
pytest --cov=. --cov-report=term-missing

# Run pre-commit checks (lint, format, type-check)
pre-commit run --all-files
```

### File-Scoped Commands (Fast Feedback)
```bash
# Type check single file
mypy config.py

# Lint single file
flake8 main.py

# Format single file
black utils.py
isort utils.py
```

### Verification Commands
```bash
# Verify configuration loads
python -c "from config import settings; print('Config OK')"

# Verify utilities work
python -c "from utils import RateLimiter; print('Utils OK')"

# Test health endpoint
curl http://127.0.0.1:7860/health
```

### Docker Commands
```bash
# Build image
docker build -t openrouter-chat .

# Run container
docker run -p 7860:7860 --env-file .env openrouter-chat
```

---

## 2. Project Architecture

**Core Stack**: Python 3.12 + Gradio v5 (Blocks UI) + OpenAI SDK + OpenRouter API + structlog + FastAPI

**Key Modules**:
- `main.py` — Gradio Blocks app with sidebar, chat orchestration, FastAPI health endpoint
- `config.py` — Environment-based settings with validation and structured logging setup
- `utils.py` — Rate limiting (token bucket), input sanitization, correlation IDs, persistence
- `conversations.json` — Persistent conversation storage (auto-saved)
- `.data/` — Runtime analytics (usage.csv) and conversation exports

**Architecture Patterns**:
- Dependency injection for settings/clients
- Streaming via Python generators (`yield`)
- Token-bucket rate limiting per IP
- Correlation ID propagation for request tracing
- Conversation persistence with UUID-based identifiers

**Data Flow**:
```
User → Gradio UI → Rate Limiter → OpenAI SDK → OpenRouter API → Model
         ↓                                                           ↓
   Conversation Manager ← ← ← ← ← ← Streaming Response ← ← ← ← ← ←
         ↓
   conversations.json + usage.csv
```

---

## 3. Code Style and Conventions

### Python Style
- **PEP 8 compliant** with modern type hints
- **Line length**: 88 characters (Black formatter)
- **Import style**: `from __future__ import annotations` at top of every module
- **Type hints**: Use modern syntax (`str | None`, `list[dict]`, not `Optional` or `List`)
- **Dataclasses**: Use `@dataclass(frozen=True)` for immutable settings/config
- **Context managers**: Always use `with` for locks, file I/O, correlation contexts

### Naming Conventions
- **Variables/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **File names**: `snake_case.py`

### Architectural Rules

**DO**:
- Use `sanitize_text()` for all user input before processing
- Wrap external calls (API, HTTP) in `try/except` with specific exceptions
- Use correlation IDs for request tracing (`ensure_correlation_id()`)
- Log structured events with `logger.info()`, not `print()`
- Test using fixtures from `tests/fixtures/` and `conftest.py`
- Use `gr.Blocks()` for all Gradio UIs (never `gr.Interface()`)
- Handle streaming with generators and `yield`

**DON'T**:
- Hardcode API keys or secrets (use environment variables only)
- Disable SSL verification (`verify=False`)
- Use `print()` statements (use logging)
- Add heavy dependencies without approval
- Modify core OpenRouter integration logic without context
- Skip input sanitization
- Use blocking operations in event loops

### Good Examples to Follow
- `tests/test_correlation_ids.py` — Proper correlation ID testing
- `tests/conftest.py` — Fixture patterns and mocking
- `utils.py:RateLimiter` — Thread-safe token bucket implementation
- `config.py:Settings` — Dataclass configuration with validation

### Legacy Patterns to Avoid
- Direct `print()` usage (replaced with structured logging)
- Ungated external calls (must have timeout and error handling)

---

## 4. Git Workflow and PR Instructions

### Branch and Commit Guidelines
```bash
# Branch naming
feature/<description>    # New features
fix/<description>        # Bug fixes
refactor/<description>   # Code refactoring
docs/<description>       # Documentation updates

# Commit message format (Conventional Commits)
<type>(<scope>): <short description>

# Examples:
feat(chat): add streaming token counter
fix(rate-limiter): correct token replenishment calculation
refactor(utils): extract correlation ID helpers
docs(readme): update installation instructions
test(integration): add conversation persistence tests
```

### Pre-Commit Requirements
**Before every commit**, pre-commit hooks automatically run:
1. **Black** — Code formatting (88 char line length)
2. **isort** — Import sorting
3. **flake8** — Linting with docstring checks
4. **mypy** — Type checking (strict on select modules)
5. **check-requirements-pinned** — Ensures exact version pins

**Manual override** (if needed): `git commit --no-verify` (use sparingly)

### Pull Request Checklist

A PR is **ready** when:
- [ ] All pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] All tests pass (`pytest`)
- [ ] Coverage remains ≥80% (`pytest --cov`)
- [ ] New features include tests (unit + integration)
- [ ] Breaking changes documented in PR description
- [ ] No hardcoded secrets or API keys
- [ ] Changes verified manually (include screenshots/logs if relevant)

### PR Description Template
```markdown
## What changed?
Brief description of the change.

## Why?
Context and motivation.

## How to verify?
Steps to test the change manually.

## Test evidence
- Unit tests: [link to test file]
- Manual verification: [screenshot/log output]
```

---

## 5. Security and Environment

### Required Environment Variables
```bash
# Required (application will not start without this)
OPENROUTER_API_KEY=your-key-here  # Get from https://openrouter.ai/keys

# Optional (with defaults)
DEFAULT_MODEL=openai/gpt-4o
TEMPERATURE=0.3
MAX_INPUT_CHARS=8000
MAX_HISTORY_MESSAGES=40
RATE_LIMIT_REQUESTS_PER_MIN=60
ENABLE_ANALYTICS=true
HOST=0.0.0.0
PORT=7860
LOG_LEVEL=INFO
LOG_JSON=true
HEALTHCHECK_ENABLED=true
HEALTHCHECK_TIMEOUT=5.0
```

### Security Rules (CRITICAL)

**NEVER**:
- Commit `.env` files with real secrets
- Hardcode API keys in source code
- Disable SSL verification in production
- Skip input sanitization
- Log raw API keys or sensitive data

**ALWAYS**:
- Use `sanitize_text()` for user input
- Wrap API calls with timeouts
- Use correlation IDs for request tracing
- Rate limit per IP before API calls
- Validate environment variables at startup

### Built-in Security Features
- **Rate limiting**: Token-bucket algorithm per IP (`RateLimiter`)
- **Input sanitization**: Control character stripping (`sanitize_text()`)
- **Correlation IDs**: Request tracing and structured logging
- **Timeouts**: All HTTP calls have explicit timeouts
- **Validation**: Settings validated at startup (`config.py:Settings.__post_init__`)

---

## 6. Testing Strategy

### Test Structure
```
tests/
├── conftest.py                    # Shared fixtures and mocks
├── fixtures/                      # Test data (JSON, YAML)
├── test_*.py                      # Unit tests
├── integration/                   # Integration tests
│   └── test_*.py
└── e2e/                          # End-to-end tests
    └── test_*.py
```

### Testing Patterns

**Use fixtures for**:
- `conversation_state` — Pre-loaded sanitized chat history
- `fake_openai_stream` — Mock streaming responses (no network calls)
- `mock_openai_client` — Patchable OpenAI client for testing
- `app_storage` — Isolated tmp directory for file I/O tests
- `quiet_main_logger` — Suppress logging in tests

**Mocking external calls**:
```python
# Example: Mock OpenAI streaming
def test_chat_streaming(mock_openai_client, fake_openai_stream):
    stream = fake_openai_stream(["Hello", " world"])
    mock_openai_client.queue_stream(stream)
    # ... test code
    assert mock_openai_client.calls  # Verify API was called
```

**Test markers**:
```bash
pytest -m integration  # Run integration tests only
pytest -m e2e          # Run end-to-end tests only
```

### Coverage Requirements
- **Minimum**: 80% coverage (enforced in CI)
- **Target**: ≥85% for new code
- **Check locally**: `pytest --cov=. --cov-report=term-missing`

---

## 7. Troubleshooting and Common Pitfalls

### Common Issues

**"No models in dropdown"**
- OpenRouter `/models` endpoint may be unavailable
- App uses fallback curated list
- Check API key validity

**"Rate limit exceeded"**
- Adjust `RATE_LIMIT_REQUESTS_PER_MIN` in `.env`
- Rate limiter uses per-IP token bucket
- Check with `curl http://localhost:7860/health`

**Streaming responses stall**
- Check network timeouts in OpenAI client config (see `main.py:35`)
- Verify OpenRouter API status
- Inspect logs for correlation ID trace

**Tests fail with "OPENROUTER_API_KEY not set"**
- Tests use `test-key` by default (see `conftest.py`)
- Override with environment variable if needed

**Pre-commit hooks fail**
- Run `pre-commit install` after cloning
- Run `pre-commit run --all-files` to see specific failures
- Fix issues or use `--no-verify` (sparingly)

**Import errors in tests**
- Ensure test environment installed: `pip install -r requirements.txt`
- Check for circular imports
- Verify `conftest.py` fixtures are not conflicting

### Debug Workflow
1. **Check logs**: `tail -f logs/app.log` (JSON format)
2. **Use correlation ID**: Track request flow through logs
3. **Test health endpoint**: `curl http://localhost:7860/health`
4. **Run single test**: `pytest tests/test_config.py::test_settings_initialization_with_env -v`
5. **Use debugger**: Add `import pdb; pdb.set_trace()` in code

---

## 8. Agent Workflow Template

When generating or modifying code, follow this workflow:

### 1. Understand Context
- Read this `AGENTS.md` file first
- Review relevant modules (`main.py`, `config.py`, `utils.py`)
- Check existing tests for patterns
- Identify related fixtures in `conftest.py`

### 2. Clarify Requirements
If the request is ambiguous:
- Ask specific questions about scope, behavior, or edge cases
- Propose a plan and wait for confirmation
- Don't guess — clarity prevents rework

### 3. Generate Code
- Follow coding conventions (PEP 8, type hints, docstrings)
- Use existing patterns (dataclasses, context managers, generators)
- Add logging with correlation IDs
- Handle errors explicitly (no bare `except`)
- Sanitize user input

### 4. Write Tests
- Add unit tests in `tests/test_*.py`
- Use fixtures from `conftest.py`
- Mock external dependencies (OpenAI client, file I/O)
- Aim for ≥80% coverage
- Test happy path + error cases + edge cases

### 5. Verify Changes
```bash
# Run pre-commit checks
pre-commit run --all-files

# Run all tests
pytest

# Run specific test
pytest tests/test_config.py -v

# Check coverage
pytest --cov=. --cov-report=term-missing

# Verify app starts
python main.py
# Open browser: http://127.0.0.1:7860
```

### 6. Document Changes
- Update docstrings for new functions/classes
- Add comments for complex logic
- Update `README.md` if user-facing changes
- Include PR description with test evidence

### 7. If Errors Occur
- Read error messages carefully (often self-explanatory)
- Check correlation ID in logs for request tracing
- Run single failing test with `-v` for details
- Use `pdb` to debug interactively
- If still stuck, **ask for clarification** instead of guessing

---

## 9. Critical Guardrails

These are **hard stops** — agent must refuse or seek approval:

| ❌ Forbidden | ✅ Alternative |
|-------------|---------------|
| Hardcode API keys | Use environment variables |
| Skip input sanitization | Always use `sanitize_text()` |
| Disable SSL verification | Use proper certificates |
| Add heavy dependencies | Justify and get approval first |
| Use `gr.Interface()` | Use `gr.Blocks()` |
| Modify core OpenRouter logic without context | Ask for clarification |
| Skip tests | Write tests for all changes |
| Use bare `except` | Catch specific exceptions |
| Use blocking I/O in event loop | Use async/await or generators |

---

## 10. Dependency Management

### Adding New Dependencies
1. **Install**: `pip install <package>==<version>`
2. **Pin exact version**: Add to `requirements.txt` with `==`
3. **Audit**: `pip-audit -r requirements.txt`
4. **Test**: `pytest` to ensure no conflicts
5. **Commit**: Include `requirements.txt` in PR

### Security Auditing
```bash
# Run before every release
pip install pip-audit
pip-audit -r requirements.txt --format json > pip-audit.json

# Check for vulnerabilities
pip-audit -r requirements.txt
```

### Pinning Enforcement
- Pre-commit hook `check-requirements-pinned` enforces exact pins (`==`)
- No `>=`, `<=`, `~=`, or unpinned versions allowed
- Prevents supply-chain surprises

---

## 11. Performance Considerations

- **Streaming**: Use generators (`yield`) for token-by-token responses
- **Rate limiting**: Token-bucket algorithm prevents API abuse
- **Concurrency**: Gradio queue with `max_size=128, default_concurrency_limit=8`
- **Memory**: Monitor with `/health` endpoint (psutil metrics)
- **Timeouts**: All HTTP calls have explicit timeouts (see `main.py:35`)

---

## 12. References and Resources

### Documentation
- **OpenRouter API**: https://openrouter.ai/docs
- **Gradio v5**: https://gradio.app/docs
- **structlog**: https://www.structlog.org/en/stable/
- **Project README**: `README.md` (detailed setup and usage)
- **Security Policy**: `SECURITY.md` (vulnerability reporting)

### Key Files
- `main.py` — Application entrypoint and UI
- `config.py` — Configuration and logging setup
- `utils.py` — Core utilities and rate limiting
- `tests/conftest.py` — Test fixtures and mocks
- `requirements.txt` — Pinned dependencies
- `.env.example` — Environment template

### CI/CD
- `.github/workflows/ci.yml` — Main pipeline (test, lint, build, scan)
- `.github/workflows/security.yml` — Weekly dependency audits
- `.github/workflows/deploy.yml` — Automated deployment

---

## 13. When You're Stuck

If you encounter uncertainty:

1. **Ask clarifying questions** instead of guessing
   - "Should this validation be strict or permissive?"
   - "What's the expected behavior when X fails?"
   - "Where should this logic live — main.py or utils.py?"

2. **Propose a plan** and wait for confirmation
   - "I'll add rate limiting to the export endpoint using the existing RateLimiter. Sound good?"
   - "This will require changes to config.py, main.py, and 3 new tests. Proceed?"

3. **Point to examples** for guidance
   - "I'm modeling this after the pattern in `tests/test_correlation_ids.py`"
   - "Using the same error handling approach as `chat_fn`"

4. **Admit limitations**
   - "I need more context about the expected OpenRouter response format"
   - "I'm not sure if this should be async or sync — please advise"

**Remember**: Asking > Guessing. Clarity prevents rework.

---

## 14. Quick Wins for New Contributors

Start here to build familiarity:

1. **Read the README**: `README.md` (architecture, setup, usage)
2. **Run the app**: `python main.py` and explore the UI
3. **Inspect logs**: `tail -f logs/app.log` (structured JSON)
4. **Test the health endpoint**: `curl http://localhost:7860/health`
5. **Run a single test**: `pytest tests/test_config.py -v`
6. **Make a small change**: Add a log statement, run tests, commit
7. **Review a test**: Study `tests/test_chat_integration.py` for patterns

---

## Summary

This project is a production-ready Gradio v5 application with OpenRouter integration. Key principles:

- **Type safety**: Modern Python type hints everywhere
- **Testing**: ≥80% coverage with unit, integration, and e2e tests
- **Security**: Input sanitization, rate limiting, correlation IDs, no hardcoded secrets
- **Structured logging**: JSON logs with correlation IDs for request tracing
- **Dependency hygiene**: Exact version pins, regular audits, pre-commit hooks
- **Clear patterns**: Dataclasses, context managers, generators, fixtures

**Most important rules**:
1. Run `pre-commit run --all-files` before every commit
2. Write tests for all changes (aim for ≥80% coverage)
3. Never hardcode secrets or skip input sanitization
4. Use correlation IDs for request tracing
5. Ask when uncertain — don't guess

---

## Maintenance

This file is **living documentation**. Update it when:

- **Architecture changes**: New modules, refactored patterns
- **Conventions evolve**: Style guides, testing patterns
- **Commands change**: Build tools, test runners, CI/CD
- **Security rules update**: New threats, best practices
- **Common pitfalls emerge**: Recurring issues, gotchas

**Rule of thumb**: If you explain something 3+ times, document it here.

---

**Last updated**: 2025-01-27  
**Version**: 1.0.0  
**Maintainers**: See `LICENSE` and `SECURITY.md` for contact info

---

*This AGENTS.md file prioritizes actionable guidance for AI coding agents. For user-facing documentation, see README.md.*
