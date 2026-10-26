"""
Plan-and-Execute agentic loop and HTTP endpoint.

Takes a user prompt, asks the LLM to produce a plan (a list of steps without
executing them), then iterates through each step one at a time, feeding the LLM
the step instruction together with the accumulated output from all previous steps.

Supports tool calling: when the LLM invokes a tool the server cannot execute,
the plan state is saved and the tool call is returned to the client. The client
executes the tool and sends the result back, resuming the plan from where it
left off.

Includes an OpenAI-compatible endpoint that can be registered with the Quart app.
"""

import asyncio
import base64
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

import quart

from . import app, config, utils, worker
from .common import message_text, read_chat_request
from .openai import _format_tool_calls

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """You are an assistant that breaks down tasks into clear, sequential steps.
Given a user request, produce a numbered list of steps needed to complete it.
DO NOT execute any of the steps — only list them.
Each step should be concise and actionable.

You MUST return your answer using EXACTLY this format:

1. First step description
2. Second step description
3. Third step description

Rules for the format:
- Each step must start on a new line with a number, a period, and a space (e.g., "1. ").
- Do NOT use any other format like "Step 1:", bullet points, or paragraphs.
- Do NOT add any introductory text, conclusions, or commentary.
- Return ONLY the numbered list, nothing else."""


EXECUTE_SYSTEM_PROMPT = """You are an assistant executing a single step of a larger plan.
You will be given:
1. The original user request (for context).
2. The outputs of all previously completed steps.
3. The specific step you need to execute right now.

If tools are available, you may use them to gather information or perform actions.
Produce the result for this step only. Be thorough and self-contained."""


# ---------------------------------------------------------------------------
# Session state management
# ---------------------------------------------------------------------------

@dataclass
class AgenticSession:
    """Holds the state of an in-progress Plan-and-Execute session.

    When the LLM produces a tool call that the server cannot execute, the
    current plan state is saved here so the session can resume after the
    client executes the tool and sends the result back.
    """
    session_id: str
    prompt: str
    model_name: str
    plan_system_prompt: str
    execute_system_prompt: str
    max_steps: int
    tools: Optional[list[dict[str, Any]]]
    # The full list of steps from the planning phase
    steps: list[str]
    # Index of the current step being executed (1-based)
    current_step_index: int
    # Outputs collected so far from completed steps
    step_outputs: list[str]
    # Accumulated formatted result string
    accumulated: str
    # Message history for the current execution step (includes system prompt,
    # user message with context, and any assistant/tool messages from tool
    # calling round-trips within this step)
    execute_messages: list[dict[str, Any]]
    # The pending tool calls that the client needs to execute
    pending_tool_calls: list[dict[str, Any]]
    max_tokens: int = 4096
    thinking: bool = True
    created_at: float = field(default_factory=time.time)


# In-memory store for active sessions. In production this could be backed by
# Redis or a database for persistence across restarts.
_active_sessions: dict[str, AgenticSession] = {}

# Session TTL: remove sessions older than 30 minutes
SESSION_TTL_SECONDS = 1800


def _cleanup_expired_sessions() -> None:
    """Remove sessions that have exceeded the TTL."""
    now = time.time()
    expired = [
        sid for sid, session in _active_sessions.items()
        if now - session.created_at > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _remove_session(sid)
        logger.info("Cleaned up expired session %s", sid)
    if expired:
        logger.info("Cleaned up %d expired sessions", len(expired))


def _save_session(session: AgenticSession) -> None:
    """Persist an agentic session so it can be resumed later."""
    _active_sessions[session.session_id] = session
    logger.info("Saved agentic session %s at step %d", session.session_id, session.current_step_index)


def _get_session(session_id: str) -> Optional[AgenticSession]:
    """Retrieve an active session by ID."""
    return _active_sessions.get(session_id)


def _remove_session(session_id: str) -> None:
    """Remove a completed session."""
    _active_sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def plan_and_execute(
    prompt: str,
    *,
    model_name: str = config.MAIN_MODEL,
    plan_system_prompt: str = PLAN_SYSTEM_PROMPT,
    execute_system_prompt: str = EXECUTE_SYSTEM_PROMPT,
    max_steps: int = 20,
    max_tokens: int = 4096,
    tools: Optional[list[dict[str, Any]]] = None,
    thinking: bool = True,
    on_step: Optional[Callable[[int, str, str], Coroutine[Any, Any, None]]] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run a Plan-and-Execute loop for *prompt*.

    Args:
        prompt: The user's original request.
        model_name: Key into ``config.models`` (default: ``config.MAIN_MODEL``).
        plan_system_prompt: System prompt used when asking the LLM to create a plan.
        execute_system_prompt: System prompt used when executing each step.
        max_steps: Safety cap on the number of steps executed.
        max_tokens: Maximum tokens to generate per step (default: 4096).
        tools: Optional list of OpenAI-compatible tool definitions available to each step.
               The LLM may invoke these tools during execution. Tool calls for tools
               not found in LOCAL_TOOLS or MCP will be returned in the result so the
               caller can forward them to the client for execution.
        thinking: Enable reasoning/thinking mode (default: True for agentic tasks).
        on_step: Optional async callback invoked after each step completes.
                 Signature: ``async def on_step(step_index: int, instruction: str, output: str)``.
        session_id: Optional session ID for resuming a paused session. If provided
                   and a matching session exists, resumes from the saved state instead
                   of starting a new plan.

    Returns:
        A dict with ``"content"`` holding the accumulated result of all steps.
        If the LLM requested tool calls that the server could not execute, the dict
        also contains ``"tool_calls"`` (list), ``"done_reason"`` set to
        ``"tool_call"``, and ``"session_id"`` so the caller can resume later.
    """
    model_info = utils.loaded_models.get(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found in loaded models")
    options: dict[str, Any] = {"max_tokens": max_tokens}
    logger.info("Thinking mode: %s", thinking)

    # Check if we're resuming a paused session
    if session_id is not None:
        session = _get_session(session_id)
        if session is None:
            logger.warning("Session %s not found; starting fresh.", session_id)
        else:
            return await _resume_session(session, model_info, options, on_step)

    # ---- Phase 1: Plan -------------------------------------------------
    plan_messages = [
        {"role": "system", "content": plan_system_prompt},
        {"role": "user", "content": prompt},
    ]

    logger.info("Planning steps for prompt: %s", prompt[:80])
    plan_result = await _collect_completion(plan_messages, model_info, options, tools=tools, think=thinking)
    plan_text = plan_result.get("content", "")
    logger.info("Plan:\n%s", plan_text)

    steps = _parse_steps(plan_text)
    if not steps:
        logger.warning("No steps extracted from plan; returning raw plan output.")
        return {"content": plan_text}

    # Cap to max_steps
    if len(steps) > max_steps:
        logger.warning("Plan has %d steps; capping at %d.", len(steps), max_steps)
        steps = steps[:max_steps]

    # ---- Phase 2: Execute ----------------------------------------------
    step_outputs: list[str] = []
    accumulated = ""

    for idx, instruction in enumerate(steps, start=1):
        previous_context = "\n\n".join(
            f"Step {i} output:\n{out}" for i, out in enumerate(step_outputs, start=1)
        )

        execute_user_content = (
            f"Original request:\n{prompt}\n\n"
            f"{'---' if previous_context else ''}"
            f"{previous_context}"
            f"\n\n"
            f"---\n"
            f"Execute this step:\n**Step {idx}: {instruction}**"
        )

        execute_messages = [
            {"role": "system", "content": execute_system_prompt},
            {"role": "user", "content": execute_user_content},
        ]

        logger.info("Executing step %d/%d: %s", idx, len(steps), instruction[:80])
        step_result = await _collect_completion(execute_messages, model_info, options, tools=tools, think=thinking)
        step_content = step_result.get("content", "")
        logger.info("Step %d result (truncated): %s", idx, step_content[:120])

        # If the LLM called a tool the server cannot execute, save state and return
        if step_result.get("tool_calls"):
            # Generate a session ID for resumption
            sid = session_id or f"agentic-{uuid.uuid4().hex[:12]}"
            session = AgenticSession(
                session_id=sid,
                prompt=prompt,
                model_name=model_name,
                plan_system_prompt=plan_system_prompt,
                execute_system_prompt=execute_system_prompt,
                max_steps=max_steps,
                max_tokens=max_tokens,
                tools=tools,
                thinking=thinking,
                steps=steps,
                current_step_index=idx,
                step_outputs=step_outputs.copy(),
                accumulated=accumulated,
                execute_messages=execute_messages,
                pending_tool_calls=step_result["tool_calls"],
            )
            _save_session(session)

            result: dict[str, Any] = {
                "content": accumulated.strip() if accumulated else "",
                "tool_calls": step_result["tool_calls"],
                "done_reason": "tool_call",
                "session_id": sid,
            }
            return result

        step_outputs.append(step_content)
        accumulated += f"\n\n--- Step {idx}: {instruction} ---\n{step_content}"

        if on_step is not None:
            await on_step(idx, instruction, step_content)

    return {"content": accumulated.strip()}


async def _resume_session(
    session: AgenticSession,
    model_info: dict[str, Any],
    options: dict[str, Any],
    on_step: Optional[Callable[[int, str, str], Coroutine[Any, Any, None]]] = None,
) -> dict[str, Any]:
    """Resume a paused agentic session after the client executed tool calls.

    The client is expected to send back the full message history including
    tool result messages. This function reconstructs the execute_messages
    from the client-provided history, continues the LLM generation, and
    proceeds with the remaining steps.

    Args:
        session: The saved session state.
        model_info: Model info dict.
        options: Generation options.
        on_step: Optional callback for step completion.

    Returns:
        Same format as ``plan_and_execute()``.
    """
    prompt = session.prompt
    steps = session.steps
    step_outputs = session.step_outputs
    accumulated = session.accumulated
    tools = session.tools
    max_steps = session.max_steps

    # The execute_messages stored in the session contain the system prompt
    # and user message for the current step, plus any assistant/tool messages
    # from tool calling round-trips. We use this as the message history to
    # continue generation.
    execute_messages = session.execute_messages

    idx = session.current_step_index
    instruction = steps[idx - 1]  # steps is 0-indexed, idx is 1-based

    logger.info("Resuming session %s at step %d/%d: %s",
                session.session_id, idx, len(steps), instruction[:80])

    # Continue the current step by calling _collect_completion with the
    # existing message history (which now includes tool results)
    step_result = await _collect_completion(execute_messages, model_info, options, tools=tools, think=session.thinking)
    step_content = step_result.get("content", "")
    logger.info("Step %d resumed result (truncated): %s", idx, step_content[:120])

    # If the LLM called another tool the server cannot execute, update state
    if step_result.get("tool_calls"):
        session.execute_messages = execute_messages
        session.pending_tool_calls = step_result["tool_calls"]
        _save_session(session)

        return {
            "content": accumulated.strip() if accumulated else "",
            "tool_calls": step_result["tool_calls"],
            "done_reason": "tool_call",
            "session_id": session.session_id,
        }

    # Step completed successfully
    step_outputs.append(step_content)
    accumulated += f"\n\n--- Step {idx}: {instruction} ---\n{step_content}"

    if on_step is not None:
        await on_step(idx, instruction, step_content)

    # Continue with remaining steps
    for next_idx, next_instruction in enumerate(steps[idx:], start=idx + 1):
        if next_idx > max_steps:
            logger.warning("Reached max_steps cap at %d.", max_steps)
            break

        previous_context = "\n\n".join(
            f"Step {i} output:\n{out}" for i, out in enumerate(step_outputs, start=1)
        )

        execute_user_content = (
            f"Original request:\n{prompt}\n\n"
            f"{'---' if previous_context else ''}"
            f"{previous_context}"
            f"\n\n"
            f"---\n"
            f"Execute this step:\n**Step {next_idx}: {next_instruction}**"
        )

        next_execute_messages = [
            {"role": "system", "content": session.execute_system_prompt},
            {"role": "user", "content": execute_user_content},
        ]

        logger.info("Executing step %d/%d: %s", next_idx, len(steps), next_instruction[:80])
        next_result = await _collect_completion(next_execute_messages, model_info, options, tools=tools, think=session.thinking)
        next_content = next_result.get("content", "")
        logger.info("Step %d result (truncated): %s", next_idx, next_content[:120])

        if next_result.get("tool_calls"):
            session.current_step_index = next_idx
            session.step_outputs = step_outputs.copy()
            session.accumulated = accumulated
            session.execute_messages = next_execute_messages
            session.pending_tool_calls = next_result["tool_calls"]
            _save_session(session)

            return {
                "content": accumulated.strip() if accumulated else "",
                "tool_calls": next_result["tool_calls"],
                "done_reason": "tool_call",
                "session_id": session.session_id,
            }

        step_outputs.append(next_content)
        accumulated += f"\n\n--- Step {next_idx}: {next_instruction} ---\n{next_content}"

        if on_step is not None:
            await on_step(next_idx, next_instruction, next_content)

    # All steps complete — clean up session
    _remove_session(session.session_id)
    logger.info("Session %s completed successfully.", session.session_id)

    return {"content": accumulated.strip()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect_completion(
    messages: list[dict[str, str]],
    model_info: dict[str, Any],
    options: dict[str, Any],
    tools: Optional[list[dict[str, Any]]] = None,
    think: bool = True,
) -> dict[str, Any]:
    """Stream a single LLM completion and return the result.

    If *tools* are provided, the LLM may invoke them during generation.
    Tools found in LOCAL_TOOLS or MCP are executed server-side by
    ``worker.generate_stream()``. Tools not found locally are returned in
    the result so the caller can forward them to the client for execution.

    If *think* is True, the LLM will use reasoning/thinking mode.

    Returns a dict with at least ``"content"`` (str). If the LLM requested
    tool calls that could not be executed server-side, the dict also contains
    ``"tool_calls"`` (list) and ``"done_reason"`` set to ``"tool_call"``.
    """
    parts: list[str] = []
    state: dict[str, Any] = {}
    final_chunk: dict[str, Any] = {}

    async for chunk in worker.generate_stream(
        stream=False,
        model_info=model_info,
        msg_history=messages,
        options=options,
        tools=tools,
        think=think,
        state=state,
    ):
        # generate_stream always yields JSON strings
        if isinstance(chunk, str):
            chunk = json.loads(chunk.strip())
        final_chunk = chunk
        msg = chunk.get("message", {})
        content = msg.get("content", "")
        if content:
            parts.append(content)

    result: dict[str, Any] = {"content": "".join(parts)}

    # If the LLM called tools that the server could not execute,
    # forward them to the caller so they can be sent to the client.
    msg = final_chunk.get("message", {})
    if msg.get("tool_calls"):
        result["tool_calls"] = msg["tool_calls"]
        result["done_reason"] = "tool_call"
    elif final_chunk.get("done_reason") == "tool_call":
        result["done_reason"] = "tool_call"

    return result


def _parse_steps(plan_text: str) -> list[str]:
    """Extract numbered steps from LLM plan output.

    Handles formats like:
        1. Do something
        2. Do another thing
    or
        Step 1: Do something
        Step 2: Do another thing
    """
    import re

    steps: list[str] = []

    # Pattern: "1. text" or "1) text"
    for match in re.finditer(r"^[\s]*\d+[.)]\s+(.+)$", plan_text, re.MULTILINE):
        steps.append(match.group(1).strip())

    # Fallback: "Step N: text"
    if not steps:
        for match in re.finditer(r"^[\s]*Step\s+\d+[\s:]+\s*(.+)$", plan_text, re.MULTILINE):
            steps.append(match.group(1).strip())

    # Last resort: split on blank lines and treat each paragraph as a step
    if not steps:
        paragraphs = [p.strip() for p in plan_text.split("\n\n") if p.strip()]
        steps = paragraphs

    return steps


# ---------------------------------------------------------------------------
# HTTP Endpoint (OpenAI-compatible format)
# ---------------------------------------------------------------------------

def _has_tool_results(messages: list[dict[str, Any]]) -> bool:
    """Check if the message history contains tool result messages.

    When the client sends back tool execution results, the messages will
    include entries with ``role: "tool"``. This indicates a continuation
    request rather than a new agentic request.
    """
    return any(msg.get("role") == "tool" for msg in messages)


def _extract_session_id(messages: list[dict[str, Any]], data: dict[str, Any]) -> Optional[str]:
    """Extract the session ID from the request.

    The session ID can be provided explicitly in the request metadata or
    extracted from a system message added by the client.
    """
    # Check for explicit session ID in request metadata
    if data.get("session_id"):
        return data["session_id"]

    # Check for session ID in metadata field
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("agentic_session_id"):
        return metadata["agentic_session_id"]

    return None


def _inject_tool_results_into_session(
    session: AgenticSession,
    messages: list[dict[str, Any]],
) -> None:
    """Inject client-provided tool results into the session's execute messages.

    The client sends back the full conversation history including tool results.
    We need to append the assistant's tool call message and the tool result
    messages to the session's execute_messages so the LLM can continue.
    """
    # Find tool result messages in the client-provided history
    tool_results = [msg for msg in messages if msg.get("role") == "tool"]

    if not tool_results:
        logger.warning("No tool result messages found in continuation request")
        return

    # Append the assistant message with tool calls to execute_messages
    # (The LLM already produced these, but we need them in the history for
    # the continuation)
    if session.pending_tool_calls:
        session.execute_messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": session.pending_tool_calls,
        })

    # Append each tool result message
    for tool_msg in tool_results:
        session.execute_messages.append({
            "role": "tool",
            "content": message_text(tool_msg.get("content", "")),
            "tool_call_id": tool_msg.get("tool_call_id"),
        })

    # Clear pending_tool_calls to prevent re-injection on subsequent resumes
    session.pending_tool_calls = []

    logger.info("Injected %d tool result(s) into session %s",
                len(tool_results), session.session_id)


@app.route("/v1/chat/agentic-completions", methods=["POST"])
async def agentic_completions():
    """OpenAI-compatible endpoint that runs the Plan-and-Execute agentic loop.

    Accepts the same payload format as /v1/chat/completions, but instead of a
    single LLM call, it first asks the model to produce a plan, then executes
    each step one at a time, streaming results back as SSE chunks.

    Supports tool calling round-trips: when the LLM invokes a tool the server
    cannot execute, the plan state is saved and the tool call is returned to
    the client with a ``session_id``. The client executes the tool and sends
    the result back in a new request with the full message history (including
    tool result messages), and the plan resumes from where it left off.

    In Open WebUI, configure a separate model entry pointing to this URL to
    offer users an explicit "agentic" mode.
    """
    # Periodically clean up expired sessions
    _cleanup_expired_sessions()

    try:
        data, uploaded_images = await read_chat_request()
        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            return quart.jsonify({
                "error": {"message": "'messages' must be a non-empty array", "type": "invalid_request_error"}
            }), 400
        messages = raw_messages
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return quart.jsonify({"error": {"message": str(exc), "type": "invalid_request_error"}}), 400

    # Reject if the inference queue is full
    max_depth = 4 if uploaded_images else 8
    depth = utils.inference_worker.pending_count()
    if depth >= max_depth:
        wait = 30 + int(random.uniform(0, 30))
        logging.warning(
            f"Rejecting agentic request: inference queue full ({depth}/{max_depth}), retry-after {wait}s"
        )
        response = quart.jsonify({
            "error": {"message": f"Queue full ({depth}/{max_depth}); retry in {wait}s.", "type": "server_error"}
        })
        response.headers["Retry-After"] = str(wait)
        return response, 503

    model_name = data.get("model", config.QUICK_MODEL)
    model_info = utils.loaded_models.get(model_name)
    if model_info is None:
        return quart.jsonify({
            "error": {"message": f"model '{model_name}' not found", "type": "model_not_found"}
        }), 404

    stream = data.get("stream", True)
    request_id = f"chatcmpl-{base64.urlsafe_b64encode(str(time.time_ns()).encode()).decode().rstrip('=')}"
    created = int(time.time())

    max_steps = data.get("max_steps", 20)
    max_tokens = data.get("max_tokens", 4096)
    thinking = data.get("think", True)
    tools = data.get("tools")

    # Determine if this is a continuation request (client sending back tool results)
    is_continuation = _has_tool_results(messages)
    session_id = _extract_session_id(messages, data)

    if is_continuation:
        # Continuation: restore session and inject tool results
        if session_id is None:
            return quart.jsonify({
                "error": {"message": "Continuation request requires session_id", "type": "invalid_request_error"}
            }), 400

        session = _get_session(session_id)
        if session is None:
            return quart.jsonify({
                "error": {"message": f"Session '{session_id}' not found", "type": "session_not_found"}
            }), 404

        # Inject the tool results from the client into the session state
        _inject_tool_results_into_session(session, messages)

        # Build prompt from messages (excluding tool messages for the prompt text)
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "tool":
                continue
            content = message_text(msg.get("content", ""))
            if content:
                prompt_parts.append(f"[{role}]: {content}")
        prompt = "\n\n".join(prompt_parts) if prompt_parts else session.prompt

    else:
        # New request: build prompt from conversation history
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = message_text(msg.get("content", ""))
            if content:
                prompt_parts.append(f"[{role}]: {content}")
        prompt = "\n\n".join(prompt_parts)

    if not stream:
        # Non-streaming mode: collect all steps and return the final result
        result = await plan_and_execute(
            prompt,
            model_name=model_name,
            max_steps=max_steps,
            max_tokens=max_tokens,
            tools=tools,
            thinking=thinking,
            session_id=session_id if is_continuation else None,
        )

        # Format tool calls for OpenAI-compatible response
        has_tool_calls = "tool_calls" in result
        message: dict[str, Any] = {
            "role": "assistant",
            "content": result.get("content", ""),
        }
        if has_tool_calls:
            message["tool_calls"] = _format_tool_calls(result["tool_calls"])

        completion: dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_info["name"],
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if has_tool_calls else "stop",
            }],
        }
        # Include session_id so the client can resume
        if "session_id" in result:
            completion["session_id"] = result["session_id"]
        return quart.jsonify(completion)

    # Streaming mode: emit SSE chunks in real-time using an asyncio.Queue
    async def completion_stream():
        # Queue for real-time chunk delivery from on_step callback
        chunk_queue: asyncio.Queue[str] = asyncio.Queue()
        stream_done = asyncio.Event()

        # Send an initial keep-alive so the client gets a response right away
        yield ":\n\n"

        async def on_step(step_idx: int, instruction: str, step_result: str):
            # Emit a step header chunk
            header_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_info["name"],
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": f"\n\n--- Step {step_idx}: {instruction} ---\n",
                    },
                    "finish_reason": None,
                }],
            }
            await chunk_queue.put(f"data: {json.dumps(header_chunk)}\n\n")

            # Emit the step result as a single content chunk
            content_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_info["name"],
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": step_result,
                    },
                    "finish_reason": None,
                }],
            }
            await chunk_queue.put(f"data: {json.dumps(content_chunk)}\n\n")

        # Run the plan-and-execute loop in a background task
        async def run_plan():
            try:
                result = await plan_and_execute(
                    prompt,
                    model_name=model_name,
                    max_steps=max_steps,
                    max_tokens=max_tokens,
                    tools=tools,
                    thinking=thinking,
                    on_step=on_step,
                    session_id=session_id if is_continuation else None,
                )
                # If the LLM requested tool calls the client should execute,
                # emit them as a streaming delta with finish_reason=tool_calls
                if "tool_calls" in result:
                    formatted_tool_calls = _format_tool_calls(result["tool_calls"], for_stream=True)
                    tool_calls_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_info["name"],
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": formatted_tool_calls,
                            },
                            "finish_reason": "tool_calls",
                        }],
                    }
                    await chunk_queue.put(f"data: {json.dumps(tool_calls_chunk)}\n\n")
                    # Emit session_id as a special metadata chunk so the client can resume
                    session_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_info["name"],
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }],
                        "session_id": result.get("session_id"),
                    }
                    await chunk_queue.put(f"data: {json.dumps(session_chunk)}\n\n")
                else:
                    # Final done chunk
                    await chunk_queue.put("data: [DONE]\n\n")
            finally:
                stream_done.set()

        # Start the plan execution in the background
        plan_task = asyncio.create_task(run_plan())

        # Drain the queue in real-time, yielding chunks as they arrive
        while True:
            try:
                # Wait for a chunk with a timeout to allow periodic cleanup checks
                chunk = await asyncio.wait_for(chunk_queue.get(), timeout=1.0)
                yield chunk
            except asyncio.TimeoutError:
                # Check if the plan is done and queue is empty
                if stream_done.is_set() and chunk_queue.empty():
                    break
                continue

        # Ensure the plan task is complete
        await plan_task

    response = quart.Response(completion_stream(), mimetype="text/event-stream")
    response.timeout = None
    response.headers["Cache-Control"] = "no-cache"
    return response
