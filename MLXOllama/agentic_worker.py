"""
Plan-and-Execute agentic loop and HTTP endpoint.

Takes a user prompt, asks the LLM to produce a plan, then executes each
step with the outputs from earlier steps available as context. Tool calls
that cannot be handled server-side pause the session for a client round-trip.
"""

import asyncio
import base64
import json
import logging
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

import quart

from . import app, config, utils, worker
from .common import message_text, read_chat_request
from .openai import _format_tool_calls

logger = logging.getLogger(__name__)


PLAN_SYSTEM_PROMPT = """You are an assistant that breaks down tasks into clear, sequential steps.
You break a user request into a short list of concrete, sequential steps.
Output ONLY a numbered list. Nothing else.

Format rules (strict):
1. First step
2. Second step
3. Third step
4. ...

- One step per line
- Start exactly with the number, a period, and a space
- No "Step 1:", no bullets, no introduction, no conclusion, no extra text
- Each step must be a single clear action the next stage can execute
- Prefer steps that can be done with the available tools
- Do not invoke tools during planning; use the available tool names to make steps actionable
- Maximum 8 steps unless the task is extremely complex"""

EXECUTE_SYSTEM_PROMPT = """You are executing exactly one step of a larger plan.
You will receive:
- The original user request
- Outputs from previous steps (if any)
- The single step you must perform now

Rules:
- Do only the current step. Do not plan ahead or re-plan.
- If tools are available, use them when needed.
- Be concise but complete for this step.
- If the step cannot be completed, say so clearly and stop.
- Do not invent information that is not in the previous outputs or tool results."""

FINAL_ANSWER_PROMPT = """You are answering the user's original request.
Use the outputs from the completed plan steps as evidence. Resolve them into one
clear, useful response to the original request. Do not repeat the plan or add
unsupported information."""

DEFAULT_MAX_TOKENS = 32768
MAX_REPLANS = 2
SESSION_TTL_SECONDS = 1800


@dataclass
class AgenticSession:
    """State needed to resume a plan after a client-side tool call."""

    session_id: str
    prompt: str
    model_name: str
    plan_system_prompt: str
    execute_system_prompt: str
    max_steps: int
    tools: Optional[list[dict[str, Any]]]
    steps: list[str]
    current_step_index: int
    step_outputs: list[str]
    accumulated: str
    execute_messages: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    generation_options: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking: bool = True
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    replans_used: int = 0


_active_sessions: dict[str, AgenticSession] = {}


def _cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [
        sid for sid, session in _active_sessions.items()
        if now - session.last_active > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _remove_session(sid)
        logger.info("Cleaned up expired session %s", sid)
    if expired:
        logger.info("Cleaned up %d expired sessions", len(expired))


def _save_session(session: AgenticSession) -> None:
    session.last_active = time.time()
    _active_sessions[session.session_id] = session
    logger.info("Saved agentic session %s at step %d", session.session_id, session.current_step_index)


def _get_session(session_id: str) -> Optional[AgenticSession]:
    return _active_sessions.get(session_id)


def _remove_session(session_id: str) -> None:
    _active_sessions.pop(session_id, None)


async def plan_and_execute(
    prompt: str,
    *,
    model_name: str = config.MAIN_MODEL,
    plan_system_prompt: str = PLAN_SYSTEM_PROMPT,
    execute_system_prompt: str = EXECUTE_SYSTEM_PROMPT,
    max_steps: int = 20,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tools: Optional[list[dict[str, Any]]] = None,
    thinking: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    on_step: Optional[Callable[[int, str, str], Coroutine[Any, Any, None]]] = None,
    on_step_start: Optional[Callable[[int, str], Coroutine[Any, Any, None]]] = None,
    on_token: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
    cancel_event: Optional[threading.Event] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a plan, execute it, and return the result or a pending tool call."""
    model_info = utils.loaded_models.get(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found in loaded models")

    generation_options: dict[str, Any] = {
        "max_tokens": max_tokens,
        "min_p": 0.05,
        "xtc_probability": 0.0,
        "xtc_threshold": 0.1,
    }
    for key, value in (("temp", temperature), ("top_p", top_p), ("top_k", top_k), ("seed", seed)):
        if value is not None:
            generation_options[key] = value
    if cancel_event is not None:
        generation_options["cancel_event"] = cancel_event

    if session_id is not None:
        session = _get_session(session_id)
        if session is not None:
            resumed_options = dict(session.generation_options)
            if cancel_event is not None:
                resumed_options["cancel_event"] = cancel_event
            return await _resume_session(
                session, model_info, resumed_options, on_step,
                on_step_start=on_step_start, on_token=on_token,
            )
        logger.warning("Session %s not found; starting fresh.", session_id)

    logger.info("Thinking mode: %s", thinking)

    # Open WebUI's task prompt is an instruction for a single completion, not
    # a request to create and execute a multi-step agent plan.
    if "### task:" in prompt.lower():
        direct_result = await _collect_completion(
            [{"role": "user", "content": prompt}], model_info,
            _phase_options(generation_options, 0.2), tools=tools,
            think=False, on_token=on_token,
        )
        return {"content": direct_result.get("content", "")}

    plan_messages = [
        {"role": "system", "content": plan_system_prompt},
        {"role": "user", "content": prompt},
    ]
    logger.info("Planning steps for prompt: %s", prompt[:80])
    plan_result = await _collect_completion(
        plan_messages, model_info, _phase_options(generation_options, 0.3),
        tools=tools, think=thinking, on_token=on_token,
    )
    plan_text = plan_result.get("content", "")
    logger.info("Plan:\n%s", plan_text)

    steps = _parse_steps(plan_text)
    if not steps:
        logger.warning("No steps extracted from plan; returning raw plan output.")
        return {"content": plan_text}
    if len(steps) > max_steps:
        logger.warning("Plan has %d steps; capping at %d.", len(steps), max_steps)
        steps = steps[:max_steps]

    session = AgenticSession(
        session_id=f"agentic-{uuid.uuid4().hex[:12]}",
        prompt=prompt,
        model_name=model_name,
        plan_system_prompt=plan_system_prompt,
        execute_system_prompt=execute_system_prompt,
        max_steps=max_steps,
        tools=tools,
        steps=steps,
        current_step_index=1,
        step_outputs=[],
        accumulated="",
        execute_messages=[],
        pending_tool_calls=[],
        generation_options={key: value for key, value in generation_options.items() if key != "cancel_event"},
        max_tokens=max_tokens,
        thinking=thinking,
    )
    return await _run_steps(
        session, model_info, generation_options, on_step,
        on_step_start=on_step_start, on_token=on_token,
    )


def _phase_options(options: dict[str, Any], default_temperature: float) -> dict[str, Any]:
    call_options = dict(options)
    call_options.setdefault("temp", default_temperature)
    return call_options


async def _run_steps(
    session: AgenticSession,
    model_info: dict[str, Any],
    options: dict[str, Any],
    on_step: Optional[Callable[[int, str, str], Coroutine[Any, Any, None]]] = None,
    *,
    on_step_start: Optional[Callable[[int, str], Coroutine[Any, Any, None]]] = None,
    on_token: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
    initial_output: str = "",
) -> dict[str, Any]:
    """Execute the current and remaining steps, then synthesize a final answer."""
    delta_parts = [initial_output] if initial_output else []
    tools = session.tools
    idx = session.current_step_index

    while idx <= min(len(session.steps), session.max_steps):
        instruction = session.steps[idx - 1]
        if on_step_start is not None:
            await on_step_start(idx, instruction)

        previous_context = "\n\n".join(
            f"Step {i} output:\n{out}" for i, out in enumerate(session.step_outputs, start=1)
        )
        execute_user_content = (
            f"Original request:\n{session.prompt}\n\n"
            f"{'---' if previous_context else ''}"
            f"{previous_context}\n\n---\n"
            f"Execute this step:\n**Step {idx}: {instruction}**"
        )
        execute_messages = [
            {"role": "system", "content": session.execute_system_prompt},
            {"role": "user", "content": execute_user_content},
        ]

        logger.info("Executing step %d/%d: %s", idx, len(session.steps), instruction[:80])
        step_result = await _collect_completion(
            execute_messages, model_info, _phase_options(options, 0.5),
            tools=tools, think=session.thinking, on_token=on_token,
        )
        step_content = step_result.get("content", "")
        logger.info("Step %d result (truncated): %s", idx, step_content[:120])

        if step_result.get("tool_calls"):
            session.current_step_index = idx
            session.execute_messages = execute_messages
            session.pending_tool_calls = step_result["tool_calls"]
            _save_session(session)
            return {
                "content": "\n\n".join(delta_parts).strip(),
                "tool_calls": step_result["tool_calls"],
                "done_reason": "tool_call",
                "session_id": session.session_id,
            }

        if _is_step_failure(step_content) and session.replans_used < MAX_REPLANS:
            revised_steps = await _replan_steps(
                session, idx, step_content, model_info, options, on_token,
            )
            if revised_steps:
                session.steps = session.steps[:idx - 1] + revised_steps
                session.replans_used += 1
                logger.info("Replanned remaining work after step %d failure", idx)
                continue

        step_block = f"\n\n--- Step {idx}: {instruction} ---\n{step_content}"
        session.step_outputs.append(step_content)
        session.accumulated += step_block
        delta_parts.append(step_block)
        if on_step is not None:
            await on_step(idx, instruction, step_content)
        idx += 1
        session.current_step_index = idx

    previous_context = "\n\n".join(
        f"Step {i} output:\n{out}" for i, out in enumerate(session.step_outputs, start=1)
    )
    if on_step_start is not None:
        await on_step_start(len(session.steps) + 1, "Final answer")
    final_messages = [
        {"role": "system", "content": FINAL_ANSWER_PROMPT},
        {
            "role": "user",
            "content": f"Original request:\n{session.prompt}\n\nCompleted step outputs:\n{previous_context}",
        },
    ]
    final_result = await _collect_completion(
        final_messages, model_info, _phase_options(options, 0.7),
        tools=None, think=session.thinking, on_token=on_token,
    )
    final_answer = final_result.get("content", "")
    final_block = f"\n\n--- Final answer ---\n{final_answer}"
    session.accumulated += final_block
    delta_parts.append(final_block)
    _remove_session(session.session_id)
    return {"content": "\n\n".join(part.strip() for part in delta_parts if part.strip())}


async def _replan_steps(
    session: AgenticSession,
    failed_step_index: int,
    failure_output: str,
    model_info: dict[str, Any],
    options: dict[str, Any],
    on_token: Optional[Callable[[str], Coroutine[Any, Any, None]]],
) -> list[str]:
    completed = "\n\n".join(
        f"Step {i} output:\n{out}" for i, out in enumerate(session.step_outputs, start=1)
    ) or "(none)"
    messages = [
        {
            "role": "system",
            "content": session.plan_system_prompt + "\nOutput only the revised remaining steps; do not repeat completed steps.",
        },
        {
            "role": "user",
            "content": (
                f"Original request:\n{session.prompt}\n\nCompleted steps and outputs:\n{completed}\n\n"
                f"Step {failed_step_index} ({session.steps[failed_step_index - 1]}) could not be completed:\n"
                f"{failure_output}\n\nProvide a revised numbered list of steps that can still satisfy the request."
            ),
        },
    ]
    result = await _collect_completion(
        messages, model_info, _phase_options(options, 0.3),
        tools=session.tools, think=session.thinking, on_token=on_token,
    )
    return _parse_steps(result.get("content", ""))


def _is_step_failure(text: str) -> bool:
    return bool(re.search(
        r"\b(?:cannot|can't|unable to|could not|couldn't|not possible|failed to)\b",
        text,
        re.IGNORECASE,
    ))


async def _resume_session(
    session: AgenticSession,
    model_info: dict[str, Any],
    options: dict[str, Any],
    on_step: Optional[Callable[[int, str, str], Coroutine[Any, Any, None]]] = None,
    *,
    on_step_start: Optional[Callable[[int, str], Coroutine[Any, Any, None]]] = None,
    on_token: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
) -> dict[str, Any]:
    """Resume the paused step and return only output generated in this request."""
    session.last_active = time.time()
    idx = session.current_step_index
    instruction = session.steps[idx - 1]
    if on_step_start is not None:
        await on_step_start(idx, instruction)

    result = await _collect_completion(
        session.execute_messages, model_info, _phase_options(options, 0.5),
        tools=session.tools, think=session.thinking, on_token=on_token,
    )
    if result.get("tool_calls"):
        session.pending_tool_calls = result["tool_calls"]
        _save_session(session)
        return {
            "content": "",
            "tool_calls": result["tool_calls"],
            "done_reason": "tool_call",
            "session_id": session.session_id,
        }

    step_content = result.get("content", "")
    if _is_step_failure(step_content) and session.replans_used < MAX_REPLANS:
        revised_steps = await _replan_steps(session, idx, step_content, model_info, options, on_token)
        if revised_steps:
            session.steps = session.steps[:idx - 1] + revised_steps
            session.replans_used += 1
            session.current_step_index = idx
            return await _run_steps(
                session, model_info, options, on_step,
                on_step_start=on_step_start, on_token=on_token,
            )

    step_block = f"\n\n--- Step {idx}: {instruction} ---\n{step_content}"
    session.step_outputs.append(step_content)
    session.accumulated += step_block
    if on_step is not None:
        await on_step(idx, instruction, step_content)
    session.current_step_index = idx + 1
    return await _run_steps(
        session, model_info, options, on_step,
        on_step_start=on_step_start, on_token=on_token,
        initial_output=step_block,
    )


async def _collect_completion(
    messages: list[dict[str, Any]],
    model_info: dict[str, Any],
    options: dict[str, Any],
    tools: Optional[list[dict[str, Any]]] = None,
    think: bool = True,
    on_token: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
) -> dict[str, Any]:
    """Collect a streamed completion while keeping reasoning out of saved output."""
    parts: list[str] = []
    state: dict[str, Any] = {}
    final_chunk: dict[str, Any] = {}
    tool_calls: list[dict[str, Any]] = []

    async for chunk in worker.generate_stream(
        stream=on_token is not None,
        model_info=model_info,
        msg_history=messages,
        options=dict(options),
        tools=tools,
        think=think,
        state=state,
    ):
        if isinstance(chunk, str):
            chunk = json.loads(chunk.strip())
        final_chunk = chunk
        msg = chunk.get("message") or {}
        if msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
        content = msg.get("content", "")
        if content:
            parts.append(content)
            if on_token is not None:
                await on_token(content)
        if chunk.get("done"):
            break

    result: dict[str, Any] = {"content": _strip_thinking("".join(parts))}
    if tool_calls:
        result["tool_calls"] = tool_calls
        result["done_reason"] = "tool_call"
    elif final_chunk.get("done_reason") == "tool_call":
        result["done_reason"] = "tool_call"
    return result


def _strip_thinking(text: str) -> str:
    """Remove rendered and native reasoning blocks before persisting model text."""
    text = re.sub(r"<thought\b[^>]*>.*?(?:</thought\s*>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think\b[^>]*>.*?(?:</think\s*>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel>|<channel\|>", "", text)
    return text.strip()


def _parse_steps(plan_text: str) -> list[str]:
    """Extract numbered steps from LLM plan output."""
    steps: list[str] = []
    for match in re.finditer(r"^\s*\d+[.)]\s+(.+)$", plan_text, re.MULTILINE):
        steps.append(match.group(1).strip())
    if not steps:
        for match in re.finditer(r"^\s*Step\s+\d+[\s:]+\s*(.+)$", plan_text, re.MULTILINE):
            steps.append(match.group(1).strip())
    if not steps:
        steps = [part.strip() for part in plan_text.split("\n\n") if part.strip()]
    return steps


def _has_tool_results(messages: list[dict[str, Any]]) -> bool:
    """Only the final tool message indicates a tool-round-trip continuation."""
    return bool(messages) and messages[-1].get("role") == "tool"


def _extract_session_id(data: dict[str, Any]) -> Optional[str]:
    """Read an explicit session ID from the request or its metadata."""
    if data.get("session_id"):
        return data["session_id"]
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("agentic_session_id"):
        return metadata["agentic_session_id"]
    return None


def _find_session_by_tool_call_ids(messages: list[dict[str, Any]]) -> Optional[AgenticSession]:
    result_ids = {
        msg.get("tool_call_id") for msg in messages
        if msg.get("role") == "tool" and msg.get("tool_call_id")
    }
    if not result_ids:
        return None
    candidates = [
        session for session in _active_sessions.values()
        if result_ids.intersection(call.get("id") for call in session.pending_tool_calls)
    ]
    return max(candidates, key=lambda session: session.last_active) if candidates else None


def _inject_tool_results_into_session(
    session: AgenticSession,
    messages: list[dict[str, Any]],
) -> bool:
    """Append only new tool results that correspond to this session's pending calls."""
    pending_by_id = {
        call.get("id"): call for call in session.pending_tool_calls if call.get("id")
    }
    matched_results: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    for msg in messages:
        call_id = msg.get("tool_call_id")
        if (msg.get("role") == "tool" and isinstance(call_id, str)
                and call_id in pending_by_id and call_id not in matched_ids):
            matched_ids.add(call_id)
            matched_results.append(msg)

    if not matched_results or matched_ids != set(pending_by_id):
        logger.warning("Continuation did not provide all pending tool results for session %s", session.session_id)
        return False

    matched_calls = [pending_by_id[call_id] for call_id in matched_ids]
    session.execute_messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": matched_calls,
    })
    for tool_msg in matched_results:
        session.execute_messages.append({
            "role": "tool",
            "content": message_text(tool_msg.get("content", "")),
            "tool_call_id": tool_msg.get("tool_call_id"),
        })
    session.pending_tool_calls = [
        call for call in session.pending_tool_calls if call.get("id") not in matched_ids
    ]
    _save_session(session)
    logger.info("Injected %d tool result(s) into session %s", len(matched_results), session.session_id)
    return True


@app.route("/v1/chat/agentic-completions", methods=["POST"])
async def agentic_completions():
    """OpenAI-compatible plan-and-execute endpoint with SSE streaming."""
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

    max_depth = 4 if uploaded_images else 8
    depth = utils.inference_worker.pending_count()
    if depth >= max_depth:
        wait = 30 + int(random.uniform(0, 30))
        logger.warning(
            "Rejecting agentic request: inference queue full (%d/%d), retry-after %ds",
            depth, max_depth, wait,
        )
        response = quart.jsonify({
            "error": {"message": f"Queue full ({depth}/{max_depth}); retry in {wait}s.", "type": "server_error"}
        })
        response.headers["Retry-After"] = str(wait)
        return response, 503

    model_name = data.get("model", config.MAIN_MODEL)
    model_info = utils.loaded_models.get(model_name)
    if model_info is None:
        return quart.jsonify({
            "error": {"message": f"model '{model_name}' not found", "type": "model_not_found"}
        }), 404

    stream = data.get("stream", True)
    request_id = f"chatcmpl-{base64.urlsafe_b64encode(str(time.time_ns()).encode()).decode().rstrip('=')}"
    created = int(time.time())
    max_steps = data.get("max_steps", 20)
    max_tokens = data.get("max_completion_tokens", data.get("max_tokens", DEFAULT_MAX_TOKENS))
    thinking = data.get("think", True)
    tools = data.get("tools") or None

    is_continuation = _has_tool_results(messages)
    session_id = _extract_session_id(data)
    if is_continuation:
        if session_id is None:
            session = _find_session_by_tool_call_ids(messages)
            session_id = session.session_id if session is not None else None
        if session_id is None:
            return quart.jsonify({
                "error": {"message": "Continuation request requires a matching tool-call session", "type": "invalid_request_error"}
            }), 400
        session = _get_session(session_id)
        if session is None:
            return quart.jsonify({
                "error": {"message": f"Session '{session_id}' not found", "type": "session_not_found"}
            }), 404
        if not _inject_tool_results_into_session(session, messages):
            return quart.jsonify({
                "error": {"message": "Continuation must include results for all pending tool calls", "type": "invalid_request_error"}
            }), 400
        prompt = session.prompt
    else:
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = message_text(msg.get("content", ""))
            if content:
                prompt_parts.append(f"[{role}]: {content}")
        prompt = "\n\n".join(prompt_parts)

    temperature = data.get("temperature")
    top_p = data.get("top_p")
    top_k = data.get("top_k")
    seed = data.get("seed")

    async def run_request(
        on_step: Optional[Callable[[int, str, str], Coroutine[Any, Any, None]]] = None,
        on_step_start: Optional[Callable[[int, str], Coroutine[Any, Any, None]]] = None,
        on_token: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        return await plan_and_execute(
            prompt,
            model_name=model_name,
            max_steps=max_steps,
            max_tokens=max_tokens,
            tools=tools,
            thinking=thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            on_step=on_step,
            on_step_start=on_step_start,
            on_token=on_token,
            cancel_event=cancel_event,
            session_id=session_id if is_continuation else None,
        )

    if not stream:
        result = await run_request()
        has_tool_calls = "tool_calls" in result
        message: dict[str, Any] = {"role": "assistant", "content": result.get("content", "")}
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
        if "session_id" in result:
            completion["session_id"] = result["session_id"]
        return quart.jsonify(completion)

    async def completion_stream():
        chunk_queue: asyncio.Queue[str] = asyncio.Queue()
        stream_done = asyncio.Event()
        cancel_event = threading.Event()
        include_role = True

        def stream_chunk(delta: dict[str, Any], finish_reason: Optional[str] = None) -> str:
            nonlocal include_role
            if include_role:
                delta = {"role": "assistant", **delta}
                include_role = False
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_info["name"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        async def on_token(token: str) -> None:
            if token:
                await chunk_queue.put(stream_chunk({"content": token}))

        async def on_step_start(step_idx: int, instruction: str) -> None:
            await chunk_queue.put(stream_chunk({
                "content": f"\n\n--- Step {step_idx}: {instruction} ---\n"
            }))

        async def run_plan() -> None:
            try:
                result = await run_request(
                    on_step_start=on_step_start,
                    on_token=on_token,
                    cancel_event=cancel_event,
                )
                if "tool_calls" in result:
                    tool_delta = {
                        "tool_calls": _format_tool_calls(result["tool_calls"], for_stream=True)
                    }
                    await chunk_queue.put(stream_chunk(tool_delta))
                    session_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_info["name"],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                        "session_id": result.get("session_id"),
                    }
                    await chunk_queue.put(f"data: {json.dumps(session_chunk)}\n\n")
                    finish_reason = "tool_calls"
                else:
                    finish_reason = "stop"
                await chunk_queue.put(stream_chunk({}, finish_reason))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Agentic completion failed")
                error = {"error": {"message": str(exc), "type": "server_error"}}
                await chunk_queue.put(f"data: {json.dumps(error)}\n\n")
            finally:
                await chunk_queue.put("data: [DONE]\n\n")
                stream_done.set()

        yield ":\n\n"
        plan_task = asyncio.create_task(run_plan())
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=15.0)
                    yield chunk
                except asyncio.TimeoutError:
                    if stream_done.is_set() and chunk_queue.empty():
                        break
                    yield ":\n\n"
                if stream_done.is_set() and chunk_queue.empty():
                    break
            await plan_task
        except asyncio.CancelledError:
            cancel_event.set()
            plan_task.cancel()
            await asyncio.gather(plan_task, return_exceptions=True)
            raise

    response = quart.Response(completion_stream(), mimetype="text/event-stream")
    response.timeout = None
    response.headers["Cache-Control"] = "no-cache"
    return response
