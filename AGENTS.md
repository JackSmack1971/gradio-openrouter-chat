# AGENTS.md: AI Collaboration Guide

This document provides essential context for AI models interacting with this project. Adhering to these guidelines will ensure consistency and maintain code quality.

## 1. Project Overview & Purpose
*   **Primary Goal:** OpenRouter × Gradio Chat Interface - a production-ready streaming conversational AI interface that provides access to hundreds of AI models through OpenRouter's unified API with smart routing, fallbacks, and cost optimization.
*   **Business Domain:** Conversational AI, API Integration, Web Applications, Developer Tools

## 2. Core Technologies & Stack
*   **Languages:** Python 3.12+ (required, as specified in Dockerfile)
*   **Frameworks & Runtimes:** Gradio v5+ (Blocks architecture), OpenAI Python SDK v1.40+ (OpenRouter-compatible), httpx for async HTTP
*   **APIs:** OpenRouter API (unified interface to 100+ AI models), OpenAI-compatible SDK integration
*   **Key Libraries/Dependencies:** 
    - `gradio>=5.0.0` (modern Blocks UI with streaming)
    - `openai>=1.40.0` (OpenRouter-compatible client)
    - `python-dotenv>=1.0.1` (environment configuration)
    - `requests>=2.31.0` (HTTP requests)
    - `httpx>=0.27.0` (async HTTP with timeout support)
*   **Platforms:** Cross-platform Python (Linux, macOS, Windows), Docker containers, Hugging Face Spaces
*   **Package Manager:** pip (standard Python package management)

## 3. Architectural Patterns
*   **Overall Architecture:** Gradio Blocks-based streaming chat application with persistent conversation management. Uses dependency injection pattern with centralized configuration, rate limiting middleware, and OpenRouter API integration for multi-model access.
*   **Directory Structure Philosophy:** 
    - Root level: Main application files and configuration
    - `.data/`: Generated analytics and conversation exports (created at runtime)
    - No complex nested structure - simple flat architecture for rapid development
*   **Module Organization:** Single-file modules with clear separation of concerns: `main.py` (UI + chat logic), `config.py` (settings), `utils.py` (shared utilities). Follows Python convention of keeping related functionality together.

## 4. Coding Conventions & Style Guide
*   **Formatting:** Follow PEP 8 conventions. Use `from __future__ import annotations` for forward compatibility. Prefer explicit type hints with modern union syntax (`str | None`).
*   **Naming Conventions:** 
    - Variables, functions: `snake_case` (e.g., `chat_fn`, `user_message`)
    - Classes: `PascalCase` (e.g., `RateLimiter`, `Settings`)
    - Constants: `UPPER_CASE` (e.g., `SYSTEM_PROMPT_DEFAULT`, `MODELS`)
    - Files: `snake_case.py`
*   **API Design:** Functional design with dataclass configuration. Uses generator patterns for streaming, context managers for resource management, and explicit error handling with typed exceptions.
*   **Common Patterns & Idioms:**
    - **Configuration:** Dataclass with `@dataclass(frozen=True)` for immutable settings
    - **Type Safety:** Comprehensive type hints, `Optional[]` for nullable values
    - **Error Handling:** Try/except blocks with specific exception types, graceful degradation
    - **Streaming:** Generator functions with `yield` for real-time responses
    - **Threading:** Context managers (`with self.lock:`) for thread-safe operations
*   **Error Handling:** Uses try/except blocks with specific exception catching. Graceful degradation for API failures. User-facing errors through Gradio's error system (`gr.Error`, `gr.Warning`).

## 5. Key Files & Entrypoints
*   **Main Entrypoint:** `main.py` - Gradio Blocks application with streaming chat interface
*   **Configuration:** 
    - `config.py` - Centralized settings using dataclass and environment variables
    - `.env` - Environment configuration (API keys, model settings, deployment config)
    - `conversations.json` - Runtime conversation persistence
*   **Utilities:** `utils.py` - Rate limiting, text sanitization, usage logging, conversation export
*   **Dependencies:** `requirements.txt` - Python package dependencies with version constraints
*   **Deployment:** `Dockerfile` - Container configuration for production deployment

## 6. Development & Testing Workflow
*   **Local Development Environment:** 
    1. Python 3.12+ required
    2. Create virtual environment: `python -m venv .venv && source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (Windows)
    3. Install dependencies: `pip install -r requirements.txt`
    4. Configure environment: Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`
    5. Run application: `python main.py`
*   **Verification Commands:**
    - Test configuration: `python -c "from config import settings; print('Config OK')"`
    - Test utilities: `python -c "from utils import RateLimiter; print('Utils OK')"`
    - Run application: `python main.py` (serves on http://127.0.0.1:7860)
*   **Testing:** Manual testing through Gradio interface. Verify streaming responses, conversation persistence, rate limiting, and model switching. **All new features should be manually tested with different models and edge cases.**
*   **Build/Deploy:** Docker build: `docker build -t openrouter-chat .` then `docker run -p 7860:7860 openrouter-chat`

## 7. Specific Instructions for AI Collaboration
*   **Environment Setup:** Always verify `OPENROUTER_API_KEY` is set before making API calls. Use environment variables for all configuration - never hardcode secrets.
*   **Security:** 
    - **CRITICAL:** Never disable SSL verification in production (`verify=False`)
    - **CRITICAL:** Always set timeouts for HTTP requests to prevent hanging
    - Validate and sanitize all user inputs using `sanitize_text()`
    - Never expose API keys in client-side code or logs
    - Use rate limiting to prevent abuse (`RateLimiter` class)
*   **Dependencies:** Use exact version constraints in `requirements.txt` for stability. For new dependencies: `pip install <package>` then `pip freeze > requirements.txt`
*   **Error Handling Best Practices:**
    - Always use try/except blocks for API calls
    - Use specific exception types (`requests.Timeout`, `ConnectionError`, etc.)
    - Provide user-friendly error messages through Gradio's error system
    - Log errors for debugging but sanitize sensitive data
*   **Gradio 5.x Patterns:**
    - **ALWAYS use `gr.Blocks()` over `gr.Interface()`** for new features
    - Use `gr.on()` for multiple event triggers
    - Enable SSR for production: `demo.launch(ssr_mode=True)`
    - Use proper event chaining with `.then()`
*   **Streaming Implementation:**
    - Use generator functions with `yield` for streaming responses
    - Configure queue: `demo.queue(max_size=128)`
    - Handle streaming errors gracefully with fallback responses
*   **Conversation Management:**
    - Save conversations to `conversations.json` after each interaction
    - Use UUID for conversation IDs
    - Implement proper state management with Gradio's state system
*   **Avoidances/Forbidden Actions:**
    - **NEVER** hardcode API keys or secrets in source code
    - **NEVER** make HTTP requests without timeouts
    - **NEVER** disable SSL verification (`verify=False`) in production
    - **NEVER** use `gr.Interface()` - use `gr.Blocks()` instead
    - **DO NOT** modify the core OpenRouter API integration without understanding the full flow
*   **Code Quality Standards:**
    - Use type hints for all function signatures
    - Include docstrings for complex functions
    - Follow the existing error handling patterns
    - Maintain the functional programming style where possible
*   **Testing Verification:**
    - Test streaming functionality with different models
    - Verify rate limiting works correctly
    - Test conversation persistence across sessions
    - Validate export/import functionality
    - Test error scenarios (network failures, invalid API keys)
*   **Performance Considerations:**
    - Use Gradio's queue system for concurrent users
    - Implement proper streaming cleanup
    - Monitor token usage and implement cost controls
    - Use session management for better resource utilization

**Configuration Priority:** Environment variables override defaults. Production deployments should always use environment-based configuration rather than modifying source code defaults.
