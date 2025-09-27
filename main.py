from __future__ import annotations
import os, time, json
import gradio as gr
from openai import OpenAI
import httpx

from config import settings
from utils import (
    sanitize_text,
    trim_history,
    RateLimiter,
    log_usage,
    export_conversation,
    ensure_correlation_id,
    get_logger,
)

logger = get_logger(__name__)

# --- OpenRouter (OpenAI SDK) client ---
# OpenRouter supports OpenAI-compatible SDKs with base_url and optional headers. 
# Source: Quickstart + API Reference. 
# https://openrouter.ai/docs/quickstart
# https://openrouter.ai/docs/api-reference/overview
client = OpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key,
    timeout=httpx.Timeout(12.0, read=30.0, write=30.0, connect=5.0),  # HTTPX timeouts
)

EXTRA_HEADERS = {}
if settings.referer:
    EXTRA_HEADERS["HTTP-Referer"] = settings.referer   # optional leaderboard metadata
if settings.x_title:
    EXTRA_HEADERS["X-Title"] = settings.x_title        # optional leaderboard metadata

# --- Model Catalog (dynamic fetch with fallback) ---
def fetch_models() -> list[str]:
    # OpenRouter provides /api/v1/models to list models
    # https://openrouter.ai/docs/api-reference/list-available-models
    try:
        r = httpx.get(f"{settings.base_url}/models", headers={"Authorization": f"Bearer {settings.api_key}"}, timeout=10.0)
        r.raise_for_status()
        data = r.json().get("data", [])
        ids = []
        for m in data:
            mid = m.get("id")
            if not mid:
                continue
            # Keep common chat models only (heuristic)
            ids.append(mid)
        # prefer a small curated shortlist at top
        curated = [
            "openai/gpt-4o", "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-70b-instruct",
            "google/gemini-1.5-pro", "deepseek/deepseek-v2.5"
        ]
        # keep order: curated first then rest uniques
        seen = set()
        out = []
        for x in curated + ids:
            if x not in seen:
                out.append(x); seen.add(x)
        return out[:200]
    except Exception:
        # Fallback if models endpoint unavailable
        return [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.1-70b-instruct",
            "google/gemini-1.5-pro",
        ]

MODELS = fetch_models()

# --- Rate Limiter ---
limiter = RateLimiter(settings.rate_limit_per_min)

SYSTEM_PROMPT_DEFAULT = (
    "You are a concise, accurate assistant. Avoid speculation. "
    "Use markdown where helpful."
)

def format_messages(history: list[dict], user_message: str, system_prompt: str) -> list[dict]:
    messages: list[dict] = []
    sys = next((m for m in history if m.get("role") == "system"), None)
    if sys is None:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append(sys)
    for m in history:
        if m.get("role") in ("user","assistant"):
            messages.append({"role": m["role"], "content": m.get("content","")})
    messages.append({"role": "user", "content": user_message})
    return trim_history(messages, settings.max_history_messages)

# --- Core chat handler (streaming generator) ---
# FIXED: Function signature now matches Gradio ChatInterface expectations
from typing import Optional


def chat_fn(
    message: str,
    history: list[dict],
    model: str,
    temperature: float,
    system_prompt: str,
    request: Optional[gr.Request] = None,
):
    correlation_id = ensure_correlation_id(request)
    history = history or []

    # Basic input validation / sanitization
    user_msg = sanitize_text(message, settings.max_input_chars)
    if not user_msg:
        logger.warning(
            "chat_input_invalid",
            reason="empty_message",
            history_length=len(history),
        )
        yield "[Input Error] Empty message."
        return

    # FIXED: Proper request handling - Gradio provides this automatically when in function signature
    ip = getattr(getattr(request, "client", None), "host", "unknown") if request else "unknown"
    selected_model = model or settings.default_model

    logger.info(
        "chat_request_received",
        ip=ip,
        selected_model=selected_model,
        history_length=len(history),
        system_prompt_length=len(system_prompt or SYSTEM_PROMPT_DEFAULT),
        message_chars=len(user_msg),
        message_preview=user_msg[:200],  # SECURITY: sanitized preview only
        correlation_id=correlation_id,
    )

    # Per-IP rate limit
    if not limiter.check(ip):
        logger.warning(
            "chat_rate_limited",
            ip=ip,
            selected_model=selected_model,
            history_length=len(history),
            correlation_id=correlation_id,
        )
        yield "Rate limit exceeded. Please slow down and try again."
        return

    # Prepare messages in OpenAI format
    msgs = format_messages(history, user_msg, system_prompt or SYSTEM_PROMPT_DEFAULT)

    start = time.time()
    first_token_time = None
    partial: list[str] = []
    usage = None
    request_payload = {
        "model": selected_model,
        "temperature": temperature,
        "message_count": len(msgs),
        "has_system": any(m.get("role") == "system" for m in msgs),
    }
    logger.info(
        "openrouter_request_started",
        request_params=request_payload,
        correlation_id=correlation_id,
    )

    try:
        stream = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": m["role"], "content": m["content"]} for m in msgs],
            stream=True,  # streaming per OpenAI schema
            temperature=temperature,
            extra_headers=EXTRA_HEADERS or None,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                if first_token_time is None:
                    first_token_time = time.time()
                partial.append(token)
                yield "".join(partial)

        # FIXED: Improved usage tracking - capture usage from final chunk if available
        usage = getattr(chunk, "usage", None) if "chunk" in locals() else None

    except Exception as e:
        logger.exception(
            "openrouter_request_failed",
            request_params=request_payload,
            correlation_id=correlation_id,
        )
        yield f"[API Error] {type(e).__name__}: {e}"
        return
    else:
        total_duration_ms = int((time.time() - start) * 1000)
        first_token_latency_ms = (
            int((first_token_time - start) * 1000)
            if first_token_time is not None
            else None
        )
        logger.info(
            "openrouter_request_completed",
            metrics={
                "total_duration_ms": total_duration_ms,
                "first_token_latency_ms": first_token_latency_ms,
            },
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            },
            correlation_id=correlation_id,
        )
    finally:
        latency_ms = int(((first_token_time or time.time()) - start) * 1000)
        # Usage logging best-effort (no strict dependency on token counts)
        try:
            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None
            cost_estimate = None  # Could be calculated based on model pricing
            log_usage(
                {
                    "ts": int(time.time()),
                    "ip": ip,
                    "model": selected_model,
                    "input_tokens": input_tokens or "",
                    "output_tokens": output_tokens or "",
                    "latency_ms": latency_ms,
                    "cost_estimate": cost_estimate or "",
                }
            )
        except Exception:
            logger.exception(
                "usage_logging_failed",
                correlation_id=correlation_id,
            )

# --- Export / Import helpers for history ---
def export_handler(history: list[dict]):
    correlation_id = ensure_correlation_id()
    logger.info(
        "conversation_export_started",
        message_count=len(history or []),
        correlation_id=correlation_id,
    )
    try:
        path = export_conversation(history or [])
    except Exception:
        logger.exception(
            "conversation_export_failed",
            correlation_id=correlation_id,
        )
        raise
    logger.info(
        "conversation_export_completed",
        export_path=path,
        correlation_id=correlation_id,
    )
    return path


def import_handler(file: str, _history: list[dict]):
    correlation_id = ensure_correlation_id()
    logger.info(
        "conversation_import_started",
        file=file,
        correlation_id=correlation_id,
    )
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Basic schema sanity
            cleaned = []
            for m in data:
                if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant"):
                    cleaned.append({"role": m["role"], "content": str(m.get("content", ""))})
            logger.info(
                "conversation_import_completed",
                imported_messages=len(cleaned),
                correlation_id=correlation_id,
            )
            return cleaned
    except Exception as e:
        logger.exception(
            "conversation_import_failed",
            correlation_id=correlation_id,
        )
        return gr.Warning(f"Import failed: {e}")
    logger.warning(
        "conversation_import_invalid_format",
        correlation_id=correlation_id,
    )
    return gr.Warning("Invalid file; expected a JSON array of messages.")

# --- Conversation management helpers ---
import uuid
import json
import os

CONVERSATIONS_FILE = "conversations.json"

def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return []

def save_conversations(conversations):
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

def create_new_conversation(conversations):
    convo_id = str(uuid.uuid4())
    title = f"Conversation {len(conversations) + 1}"
    return {"id": convo_id, "title": title, "history": []}

# --- Gradio UI (Blocks with responsive layout) ---
with gr.Blocks(title=settings.app_title, fill_height=True, theme="soft") as demo:
    conversations_state = gr.State([])
    current_convo_id = gr.State(None)

    # Load conversations on app start
    demo.load(load_conversations, outputs=[conversations_state])

    with gr.Row():
        # Sidebar
        with gr.Accordion("Conversations", open=True) as sidebar:
            gr.Markdown("## Conversations")
            convo_list = gr.Dropdown(choices=[], label="Select Conversation", interactive=True)
            with gr.Row():
                new_convo_btn = gr.Button("New", variant="secondary")
                delete_convo_btn = gr.Button("Delete", variant="stop")
            export_btn = gr.Button("Export")
            import_file = gr.File(file_types=[".json"], label="Import JSON")

        # Main chat area
        with gr.Column(scale=3) as main:
            gr.Markdown(f"### {settings.app_title}\nOpenRouter-powered chat (streaming).")

            with gr.Row():
                model_dd = gr.Dropdown(choices=MODELS, value=settings.default_model, label="Model", interactive=True)
                temp = gr.Slider(0.0, 1.2, value=settings.temperature, step=0.05, label="Temperature", interactive=True)
            sys_prompt = gr.Textbox(value=SYSTEM_PROMPT_DEFAULT, lines=2, label="System prompt")

            chatbot = gr.Chatbot(type="messages", height=400, show_label=False)
            msg = gr.Textbox(placeholder="Type your message here...", label="Message", show_label=False)
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

    export_out = gr.File(label="Download", interactive=False)

    # Save on changes

    # --- Event Handlers ---

    def update_convo_list(conversations, current_id=None):
        conversations = conversations or []
        choices = [(c["title"], c["id"]) for c in conversations]
        valid_ids = {c["id"] for c in conversations}
        value = current_id if current_id in valid_ids else (conversations[0]["id"] if conversations else None)
        return gr.update(choices=choices, value=value)

    def new_conversation(conversations, current_id):
        correlation_id = ensure_correlation_id()
        conversations = conversations or []
        logger.info(
            "conversation_create_requested",
            existing_count=len(conversations),
            correlation_id=correlation_id,
        )
        if current_id and conversations:
            # Save current before creating new
            pass  # Will implement with persistence
        new_convo = create_new_conversation(conversations)
        updated_conversations = conversations + [new_convo]
        logger.info(
            "conversation_created",
            conversation_id=new_convo["id"],
            total_count=len(updated_conversations),
            correlation_id=correlation_id,
        )
        return updated_conversations, new_convo["id"], [], update_convo_list(updated_conversations, new_convo["id"])

    def select_conversation(convo_id, conversations):
        convo = next((c for c in conversations if c["id"] == convo_id), None)
        if convo:
            return convo["history"], convo_id
        return [], convo_id  # keep id even if not found?

    def delete_conversation(convo_id, conversations, current_id):
        correlation_id = ensure_correlation_id()
        conversations = conversations or []
        original_count = len(conversations)
        logger.info(
            "conversation_delete_requested",
            conversation_id=convo_id,
            existing_count=original_count,
            correlation_id=correlation_id,
        )
        conversations = [c for c in conversations if c["id"] != convo_id]
        if convo_id == current_id:
            if conversations:
                new_current = conversations[0]["id"]
                logger.info(
                    "conversation_deleted",
                    conversation_id=convo_id,
                    remaining_count=len(conversations),
                    new_current=new_current,
                    correlation_id=correlation_id,
                )
                return conversations, new_current, conversations[0]["history"], update_convo_list(conversations, new_current)
            else:
                logger.info(
                    "conversation_deleted",
                    conversation_id=convo_id,
                    remaining_count=len(conversations),
                    new_current=None,
                    correlation_id=correlation_id,
                )
                return conversations, None, [], update_convo_list(conversations, None)
        logger.info(
            "conversation_deleted",
            conversation_id=convo_id,
            remaining_count=len(conversations),
            new_current=current_id,
            correlation_id=correlation_id,
        )
        return conversations, current_id, gr.skip(), update_convo_list(conversations, current_id)

    def send_message(
        message,
        history,
        model,
        temp,
        sys_prompt,
        conversations,
        current_id,
        request: gr.Request,
    ):
        if not current_id:
            yield gr.skip(), gr.skip(), gr.skip()
            return
        if not message.strip():
            yield gr.skip(), gr.skip(), gr.skip()
            return
        # Call chat_fn for response
        response = ""
        history_with_user = history + [{"role": "user", "content": message}]
        try:
            for partial in chat_fn(message, history, model, temp, sys_prompt, request=request):
                if isinstance(partial, str):
                    response = partial
                    current_history = history_with_user + [{"role": "assistant", "content": response}]
                    yield current_history, "", gr.skip()
                else:
                    # Error
                    yield history_with_user + [{"role": "assistant", "content": partial}], "", gr.skip()
                    break
        except Exception as e:
            yield history_with_user + [{"role": "assistant", "content": f"Error: {e}"}], "", gr.skip()
            return
        # Final update
        final_history = history_with_user + [{"role": "assistant", "content": response}]
        # Update conversations
        for c in conversations:
            if c["id"] == current_id:
                c["history"] = final_history
                break
        yield final_history, "", conversations

    def clear_chat(conversations, current_id):
        if not current_id:
            return gr.skip()
        for c in conversations:
            if c["id"] == current_id:
                c["history"] = []
                break
        return [], conversations

    # Button clicks
    new_convo_btn.click(
        new_conversation,
        inputs=[conversations_state, current_convo_id],
        outputs=[conversations_state, current_convo_id, chatbot, convo_list]
    )

    convo_list.change(
        select_conversation,
        inputs=[convo_list, conversations_state],
        outputs=[chatbot, current_convo_id]
    )

    delete_convo_btn.click(
        delete_conversation,
        inputs=[convo_list, conversations_state, current_convo_id],
        outputs=[conversations_state, current_convo_id, chatbot, convo_list]
    )

    submit_btn.click(
        send_message,
        inputs=[msg, chatbot, model_dd, temp, sys_prompt, conversations_state, current_convo_id],
        outputs=[chatbot, msg, conversations_state]
    ).then(lambda: gr.Info("Message sent."), outputs=[])

    clear_btn.click(
        clear_chat,
        inputs=[conversations_state, current_convo_id],
        outputs=[chatbot, conversations_state]
    )

    export_btn.click(
        export_handler,
        inputs=[chatbot],
        outputs=[export_out]
    )

    import_file.change(
        import_handler,
        inputs=[import_file, chatbot],
        outputs=[chatbot]
    )

    # Queue: set app-wide default concurrency + backpressure
    demo.queue(max_size=128, default_concurrency_limit=8)

if __name__ == "__main__":
    # For reverse proxies, configure trusted IPs so request.client.host reflects real client:
    demo.launch(server_name=settings.host, server_port=settings.port, show_error=True)
