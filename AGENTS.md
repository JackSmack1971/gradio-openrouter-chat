# AGENTS.md — AI Collaboration Guide

**Project**: OpenRouter + Gradio Chat  
**Purpose**: A streaming conversational interface over OpenRouter’s unified API, with smart routing, fallback logic, and cost optimization.

---

## 1. Project Structure & Architecture

**Directory & module layout**  
- `main.py` — entrypoint: defines Gradio Blocks UI and chat orchestration  
- `config.py` — application configuration, environment loading, settings dataclass  
- `utils.py` — shared utility functions: rate limiting, sanitization, logging, persistence  
- `.data/` — runtime‐generated folder for conversation exports, analytics  
- `conversations.json` — persistent store for chat history  
- `Dockerfile` — container spec  
- `.env` / `.env.example` — environment variables specification  
- `requirements.txt` — pinned dependencies  

**Architecture patterns**  
- Dependency injection for passing settings or client instances  
- Middleware / wrapper layers for rate limiting, error handling  
- Streaming via Python generator / async patterns  
- Stateful session tracking, conversation context, fallback logic  

---

## 2. Setup, Build & Execution Commands

**Environment & prerequisites**  
- Python 3.12+  
- Docker (for containerized deployment)  

**Local setup**  
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# or `.venv\Scripts\Activate.ps1` on Windows PowerShell
pip install -r requirements.txt
cp .env.example .env             # and populate secrets
````

**Verification & sanity checks**

```bash
python -c "from config import settings; print('Config OK')"
python -c "from utils import RateLimiter; print('Utils OK')"
```

**Run (development / production)**

```bash
python main.py                   # serves Gradio app locally (default port 7860)
```

**Docker (production build & run)**

```bash
docker build -t openrouter-chat .
docker run -p 7860:7860 openrouter-chat
```

Agents, when generating code changes, should anticipate running these commands (or portions thereof) to verify behavior.

---

## 3. Testing & Validation Strategy

**Current state**

* Manual testing via Gradio UI
* Edge case testing of streaming, model fallback, persistence

**Agent responsibilities**

* For every generated feature / change, include tests (unit or integration)
* Run any verification commands (see “Setup & Verification”) after code generation
* If tests or sanity checks fail, attempt minimal fixes (lint, import, small adjustments)
* If still failing, surface errors / logs clearly

**Future enhancements (optional suggestions)**

* Adopt a test framework (e.g. `pytest`)
* Enforce coverage thresholds
* Integrate CI (GitHub Actions) executing linting, testing, build

---

## 4. Coding Conventions & Best Practices

**Formatting & style**

* PEP 8 compliance
* Use type hints with modern syntax: `str | None`, `list[str]` etc.
* `from __future__ import annotations` for forward/ref annotations

**Naming & structure**

* Variables/functions: `snake_case`
* Classes: `PascalCase`
* Constants / config keys: `UPPER_CASE`
* File names: `snake_case.py`

**Patterns & architectural constraints**

* Use dataclasses (immutable / frozen) for settings
* Streaming via `yield` / async generator patterns
* Explicit error classes per domain (avoid generic `Exception`)
* Use context managers / `with` for resource cleanup
* All HTTP calls must include timeouts, do not disable SSL verification

**Forbidden / high-risk actions**

* **Never** hardcode secrets or API keys
* **Never** disable SSL verification (`verify=False`) in production
* **Never** use `gr.Interface()` for new features — always use `gr.Blocks()`
* **Do not** change core OpenRouter integration logic without full context
* **Always** sanitize all user input via provided utility (`sanitize_text()` or equivalent)
* Avoid adding heavy dependencies unless essential; prefer standard libs or minimal, well-vetted packages

---

## 5. Error Handling, Logging & Fallback Behavior

* Wrap all external calls (API, HTTP) in `try/except` with explicit exceptions (e.g. timeout, connection errors)
* On failure, degrade gracefully: fallback to alternative model, notify user, log internally
* Use Python’s `logging` module over `print()` statements
* Limit log verbosity in production; never log raw secrets or sensitive environment values
* For streaming interruptions: detect breakpoints, send partial output or error message

---

## 6. Conversation State & Persistence

* Use UUIDs for conversation IDs
* Persist each message to `conversations.json` after response
* Load prior context per session (if exists)
* State transitions: ensure synchronization with Gradio’s state mechanism
* Export / import functionality must maintain message ordering, metadata

---

## 7. Performance & Resource Constraints

* Use `demo.queue(max_size=128)` for handling concurrency
* Clean up streaming generators / buffers after session end
* Monitor token usage (cost control) — avoid large payloads gratuitously
* Avoid blocking operations inside the event loop; prefer async / nonblocking calls

---

## 8. Commit, Branching & PR Guidelines

* Branch from main (or `dev`) for features / bugs
* Commit message format: `<scope>: <short description>` (optionally Conventional Commits)
* Always run verification commands (setup, sanity, tests) before committing
* PR should include:

  1. Description of change
  2. Associated tests or manual verification steps
  3. Screenshots / logs (if relevant)
  4. Impact analysis (e.g. token cost, fallback behavior)

---

## 9. Agent Workflow Template

When a prompt asks the agent to generate or modify code:

1. **Locate context**: read nearest `AGENTS.md` in working directory
2. **Summarize module context**: gather imports, existing functions, dependencies
3. **Clarify ambiguities**: if prompt lacks detail (e.g. “which model fallback logic?”), ask follow-up
4. **Generate code**: adhere to conventions, architecture, and domain rules
5. **Add tests**: integrate verifications and edge cases
6. **Run verification commands**: setup check, sanity, build, test
7. **If errors**: patch minimal changes (imports, name fixes, lint) and re-run
8. **If still failing**: output diagnostics and stop
9. **Return**: code diff + commit message draft

---

## 10. Common Pitfalls & Guardrails

| Pitfall                                   | Guardrail / Clarification                                                                     |
| ----------------------------------------- | --------------------------------------------------------------------------------------------- |
| Agent omits tests                         | Mandate “include tests” in task template                                                      |
| Agent forgets imports or dependencies     | After generation, enforce minimal import check / run                                          |
| Hardcoded secrets or keys                 | Reject any occurrence of literal strings that match `OPENROUTER_API_KEY` patterns             |
| Changing core API logic without awareness | Disallow modifications in module `openai` / router integration parts without explicit context |
| Disabling SSL or timeout                  | Signal as forbidden; treat as error                                                           |
| Using `gr.Interface()` accidentally       | On detection, prompt replacement with `gr.Blocks()`                                           |
| Inconsistent naming / style               | Run a lint or style check (PEP 8) after generation                                            |

---

## 11. References & Further Reading

* OpenAI’s official AGENTS.md repository / spec ([GitHub][1])
* The rising adoption and rationale behind AGENTS.md as a standard ([InfoQ][2])
* Community guides on best practices for AGENTS.md ([AIMultiple][3])

---

## 12. Maintenance & Versioning Notes

* Keep `AGENTS.md` aligned with the real project as it evolves
* When architectural, convention, or tooling shifts occur, update this file first
* Use nested `AGENTS.md` in submodules if project fragments diverge (monorepo style)
* Agents always select **closest** `AGENTS.md` in directory hierarchy for contextual guidance

---

### Summary of changes I made compared to your original:

* **Reordered sections** for logical flow (structure → setup → conventions → workflow)
* **Added a generalized agent workflow template** to prompt agents to validate their work
* **Guardrail / pitfall table** to explicitly flag classes of mistakes
* **Testing & validation orchestration** section to push agents to verify changes
* More explicit rules on forbidden operations, performance, logging, error handling
