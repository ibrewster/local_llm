import asyncio
import copy
import datetime
import gc
import json
import logging
import queue
import threading
import time
import random
import re
import traceback

from pathlib import Path

import numpy

from mlx_lm import generate, stream_generate, sample_utils
import mlx.core as mx

from . import utils, config, local_tools, tts_queue
from .cache_utils import get_cache


def gen_goodnight(prompt):
    t0 = time.time()
    logging.info("Goodnight generation beginning")
    # client=genai.Client(api_key=config.GOOGLE_API_KEY)

    current_month = datetime.datetime.now().strftime("%B")
    day_of_year = datetime.datetime.now().timetuple().tm_yday
    item_pick = random.randint(1, 50)
    
    system_instruction = (
        f"You are a Fairbanks, Alaska storyteller. It is currently {current_month}. "
        'Write a gentle, short, 2-stanza rhythmic bedtime poem using AABB or AABBCC rhyme schemes in the style of "Goodnight Moon" or other bedtime stories'
        "Vary the style depending on the day of the year. "
        "Pick an item from the specified special object category in this manner:"
        "  - mentally generate a list of 50 unique, plausible items within that category. Rank these 50 items based on their cultural iconic status and thematic intensity. "
        "  - IF the category is Alaskan Lifestyle Object, avoid highly specialized, scientific, or industrial items that you would not find in a normal or traditional Alaskan house. Instead, Prioritize Alaska Native items like ulus, totem poles, various native garments, etc. "
        "  - Treat the provided 50-sided die result as the index for that list. Select the item located at that exact index. "
        "Use this item naturally somewhere in the poem. "

        f"- Use Fairbanks imagery for {current_month}. " 
        "- Weave in the specific Low and High temperatures and the weather conditions for tomorrow naturally. "
        "- ONLY mention the Aurora if the user explicitly states an alert is active."
        "- Use the day of the week tomorrow to further color the tone (e.g., Friday's look toward the weekend). "
        "- End the second stanza with a quiet, local Fairbanks closing thought. "
        "NO MARKDOWN. CLEAN TEXT ONLY. "
    )
    
    special_items = ["Alaskan Animal", "Alaskan Lifestyle Object"]
    special_item = random.choice(special_items)
    random_info = f"\n--- MAPPING DATA ---\n50-sided die (Item Index): {item_pick}\nSpecial Object Category: {special_item}.\nThe day of the year is: {day_of_year}\n-------------------"
    
    message = [
        {"role": "system","content": system_instruction,}, 
        {"role": "user","content": prompt + random_info,}
    ]
    
    
    mod_info = utils.loaded_models[config.MAIN_MODEL]
    model = mod_info['model']
    tokenizer = mod_info['tokenizer']
    
    cache, _ = asyncio.run(get_cache(mod_info, message, None, is_static=True))
    if cache:
        cache = copy.copy(cache)
        message = message[1:]
    
    sampler = sample_utils.make_sampler(
        temp=0.95,
        top_p=0.92
    )
    
    formatted_prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    response = ""
    sentences = []
    first_sentence = None
    with utils.mlx_inference_lock:
        mx.random.seed(int(time.time()))
        for token in stream_generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            prompt_cache=cache,
            sampler=sampler
        ):
            response += token.text
            if '\n' in response:
                portions = response.split('\n')
                
                if first_sentence is None:
                    first_sentence = time.time() - t0
                    tts_queue.put((portions[0].strip(), 0.8, 'af_bella'))
                    
                sentences.append(portions[0])
                if len(portions) > 1:
                    response = "\n".join(portions[1:])
                else:
                    response = ""
            
        if response.strip():
            sentences.append(response)
            tts_queue.put((" ".join(sentences[1:]), 0.8, 'af_bella'))
    
    tts_queue.put("__FLUSH__")
    
    response = "\n".join(sentences)
    # Remove any "gemma" tags
    response = re.sub(r'<\|[^>]+>.*?<[^>]+?\|>', '', response, flags=re.DOTALL)
    
    logging.info(f"Got goodnight text in {time.time() - t0}. First sentence in {first_sentence}")
    
    out_dir: Path = Path(__file__).parent / "GenOut"
    out_dir.mkdir(exist_ok=True)
        
    with open(out_dir / "goodnight.txt", 'w') as f:
        f.write(response)
    
    return
        
def speak_thread(speak_queue):
    logging.info("Listening for prompts")
    mod_info = utils.loaded_models[config.MAIN_MODEL]
    model = mod_info['model']
    tokenizer = mod_info['tokenizer']
        
    while True:
        mx.clear_cache()
        gc.collect()
        mx.clear_cache()         
        try:
            msg = speak_queue.get(timeout=660) # 11 minutes
        except queue.Empty:
            run_dummy_inference()
            continue
        
        if isinstance(msg, tuple):
            system, prompt = msg
        else:
            system: str = None
            prompt: str = msg
        
        if prompt == "QUIT":
            break
        if prompt == "UPDATE":
            time.sleep(1)
            run_dummy_inference()
            continue
        
        ############ Bedtime Message ###################
        if prompt.startswith("It is bedtime"):
            gen_goodnight(prompt)
            continue
        ################################################
        
        t1=time.time()
        use_thinking = prompt.startswith("/think")
        clean_prompt = prompt.replace("/think", "").strip()
        message = [
            {"role": "user","content": clean_prompt,}
        ]
        
        cache = None
        if system:
            message = [
                {"role": "system","content": system}
            ] + message
            
            cache, _ = asyncio.run(get_cache(mod_info, message, None, is_static=True))
            if cache:
                cache = copy.copy(cache)
                message = message[1:]
            
        formatted_prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=use_thinking
        )
        
        try:
            with utils.mlx_inference_lock:
                buffer = ""
                mx.random.seed(int(time.time_ns() % 2**32))
                
                stream = stream_generate(
                    model,
                    tokenizer,
                    prompt=formatted_prompt,
                    sampler=utils.FUN_SAMPLER,
                    max_kv_size=2048,
                    max_tokens=4096,
                    prefill_step_size=6144, 
                    prompt_cache=cache
                )
                
                is_thinking = False
                sentence_count = 0
                for chunk in stream:
                    token_text = chunk.text
                    
                    # Check for state changes
                    if "<think>" in token_text or '<channel|>' in token_text:
                        is_thinking = True
                        # Strip the tag itself if it's bundled with other text
                        token_text = token_text.replace("<think>", "")
                        token_text = token_text.replacte('<channel|>', "")
                        
                    if "</think>" in token_text or '<|channel>' in token_text:
                        is_thinking = False
                        # Strip the tag and continue—only text AFTER this is the 'answer'
                        token_text = token_text.split("<|channel>")[-1]
                        if not token_text:
                            continue
                
                    # If the model is currently 'thinking', skip sending to buffer/TTS
                    if is_thinking:
                        continue
                        
                    content = token_text
                    if not content:
                        continue
                    buffer += content
                    
                    pattern = utils.sentence_endings_conservative if sentence_count < 2 else utils.paragraph_split
                    matches = list(pattern.finditer(buffer))
                    
                    if matches:
                        last_match = matches[-1]
                        split_point = last_match.end()
                        
                        complete = buffer[:split_point]
                        buffer = buffer[split_point:]
                        
                        if sentence_count == 0:
                            logging.info(f"Mode: {'Reasoning' if use_thinking else 'Instant'}")
                            logging.info(f"Time to first sentence: {time.time()-t1}")
                        sentence_count += 1
                        logging.debug(complete.strip())
                        tts_queue.put(complete.strip())
        
            # Flush remaining text
            if buffer:
                tts_queue.put(buffer)
        finally:
            tts_queue.put("__FLUSH__")
            logging.info(f"Completed inference in {time.time() - t1}")
            
    logging.info("Prompt processing thread exited")
    
def run_dummy_inference():
    """Run the fastest possible inference, just to keep things alive/in ram"""    
    for mod_name, mod_info in utils.loaded_models.items():
        model = mod_info['model']
        tokenizer = mod_info['tokenizer']
        
        t1 = time.time()
        message = [{"role": "user", "content": "Hi"}]
    
        formatted = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    
        sampler = sample_utils.make_sampler(
            temp=0.0,
            top_p=1.0,
        )
    
        with utils.mlx_inference_lock:            
            _ = generate(
                model,
                tokenizer,
                prompt=formatted,
                sampler=sampler,
                max_tokens=1
            )
        logging.info(f"Ran keep-alive inference for {mod_name} in {time.time() - t1}")
   
            
async def _stream_tokens(
    model_info: dict, msg_history: str,
    options: dict, state: dict=None,
    tools: list = None,
    think: bool = False
):
    """
    Yields (token_str, is_done, stats) tuples.
    Runs MLX in a thread to avoid blocking the event loop.
    Adapt this to however your existing pipeline streams tokens.
    """
    t0 = time.time_ns()
    eval_count = 0
    try:
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        cache, is_ha = await get_cache(model_info, msg_history, tools)
        if cache:
            cache = copy.copy(cache)
            tools = None
            msg_history = msg_history[1:]

        if is_ha:
            think = False

        if "### task:" in msg_history[-1].get("content", "").lower():
            options['temp'] = 0.1
            think = False
            options['max_tokens'] = 256

        sampler = sample_utils.make_sampler(
            temp=options.get("temp", 0.65), 
            min_p=0.05,
            xtc_probability=0.5,
            xtc_threshold=0.1
        )

        formatted_prompt = tokenizer.apply_chat_template(
            msg_history,
            tools=tools, 
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think
        )
        
        if formatted_prompt.strip().endswith("<think>"):
            yield "<think>\n", False, None
            
        if "<|channel>" in formatted_prompt:
            yield "<|channel>" + formatted_prompt.rsplit("<|channel>")[1], False, None
        
        suffix_tokens = tokenizer.encode(formatted_prompt)
        logging.info(f"{len(suffix_tokens)} Tokens encoded")
        
        available_space = 262144 - len(suffix_tokens) - 100
        safe_max = min(options.get("max_tokens", 64000), available_space)
        
        full_output_tokens = []
        
        token_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        def thread_worker():
            try:
                with utils.mlx_inference_lock:
                    # This runs in a worker thread
                    for response in stream_generate(
                        model,
                        tokenizer,
                        prompt=suffix_tokens,
                        max_tokens=safe_max,
                        sampler=sampler,
                        prompt_cache=cache         
                    ):
                        loop.call_soon_threadsafe(
                            token_queue.put_nowait,
                            (response.text, response.token, False, None) 
                        )
                loop.call_soon_threadsafe(
                    token_queue.put_nowait, ("", None, True, None)
                )

            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, (None, None, True, e))
                
        thread = threading.Thread(target=thread_worker, daemon=True)
        thread.start()
        
        while True:
            result = await token_queue.get()
        
            text, token,is_done, err = result
            
            if isinstance(err, Exception):
                # Optional: yield partial + log, then re-raise or swallow
                logging.error(f"Inference error: {err}")
                yield "", True, {"error": str(err)}
                break
            if is_done:
                break
            
            eval_count += 1
            full_output_tokens.append(token)
            yield text, False, None

        state['context'] = tokenizer.encode(formatted_prompt) + full_output_tokens
        state['eval_count'] = len(tokenizer.encode(formatted_prompt))
    except Exception as e:
        logging.error(f"Unable to generate output ({e})")
        state['context'] = None
        state['eval_count'] = 0
        
    elapsed = time.time_ns() - t0
    yield "", True, {"eval_count": eval_count, "eval_duration": elapsed}
    
async def generate_stream(stream, model_info, msg_history, options,
                          is_gen=False, tools=None, think=False):
    t1 = time.time_ns()
    
    for msg in msg_history:
        if not "content" in msg:
            msg["content"] = ""
    
    model_name = model_info.get("name", config.QUICK_MODEL)
    
    if stream:            
        if utils.cache_lock.locked():
            yield json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "One moment, Captain... systems are currently updating.",
                    "tool_calls": None
                },
                "done": False,
            }) + "\n"
            
            async with utils.cache_lock:
                pass
        
    
    for iteration in range(config.MAX_TOOL_CALLS):
        t2 = time.time()
        full_text = ""
        stats = {}
        state = {}    
        tool_call_detected = False    
        buffer = ""
        unicode_buf = ""
        first_token = True
        
        async for token, done, stats in _stream_tokens(
            model_info, msg_history, options, state, tools=tools, think=think
        ):
            if unicode_buf or '\\' in token:
                unicode_buf += token
                if len(unicode_buf) >= 6:  # \uXXXX is 6 chars
                    token = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), unicode_buf)
                    unicode_buf = ""
                else:
                    continue
                
            if first_token:
                first_token = False
                if "thought" in token and not "<|channel>" in token:
                    logging.warning("Missing <|channel>! Prepending")
                    token = "<|channel>" + token

            full_text += token
            
            if tool_call_detected:
                continue
            
            # See if we have a tool call. If so, stop streaming and accumulate tool calls
            if buffer or "<" in token:
                buffer += token
                if ">" in buffer:
                    if "<tool_call>" in buffer or "<|tool_call>" in buffer:
                        tool_call_detected = True
                        continue

                    # Clean up Gemma 4 output
                    elif "<channel|>" in buffer:
                        buffer = re.sub(r'(<\|channel>.*?)?<channel\|>', '', buffer, flags=re.DOTALL)    
                        token = buffer
                        buffer = ""
                    elif "<|channel>" in buffer:
                        continue                        
                    else:
                        token = buffer
                        buffer = ""
                else:
                    continue  # still accumulating, don't stream yet
            
            if not tool_call_detected and stream: # and not done:
                chunk = {
                    "model": model_name,
                    "done": done
                }
                if is_gen:
                    chunk["response"] = token
                else:
                    chunk['message'] =  {"role": "assistant", "content": token}
    
                line = json.dumps(chunk) + "\n"
                yield line
    
        logging.info(f"Generated response in {time.time() - t2}")
        logging.info(f"Raw Response: {full_text}")
        
        tool_calls,cleaned_text = utils.parse_tool_calls(full_text)
        
        if not tool_calls or is_gen:
            break

        executed_msgs, remaining_calls = await try_server_tools(tool_calls)
            
        if remaining_calls:
            # Forward only what's left
            done = False if executed_msgs else True
            msg = {
                "model": model_name,
                "created_at": utils.now_iso(),
                "message": {
                    "role": "assistant",
                    "content": cleaned_text or "",
                    "tool_calls": remaining_calls,
                },
                "done": done,
            }
            if done:
                msg["done_reason"] = "tool_call"
                
            yield json.dumps(msg) + "\n"
                
        if executed_msgs:
            msg_history.append({
                "role": "assistant",
                "content": cleaned_text or "",
                "tool_calls": tool_calls,
            })            
            msg_history.extend(executed_msgs)
            
            if iteration == config.MAX_TOOL_CALLS - 1:
                logging.warning(f"Tool call limit of {config.MAX_TOOL_CALLS} reached")
                msg_history.append({
                    "role": "user",
                    "content": (
                        "Tool call limit reached. You have access to the results already retrieved — "
                        "use them to provide the best answer you can. Do not attempt any further tool calls. "
                        "If the information is incomplete, acknowledge that clearly in your response."
                    )
                })
                
        else:
            break
        
        
    total_time = time.time_ns() - t1
    stats_fields = {
        "model": model_name,
        "done": True,
        "done_reason": "stop",
        "created_at": utils.now_iso(),
        "eval_count": stats.get("eval_count") if stats else None,
        "eval_duration": stats.get("eval_duration") if stats else None,
        "prompt_eval_count": state.get('eval_count', 0),
        "prompt_eval_duration": 0,
        "total_duration": int(total_time),
        "load_duration": 0,
    }    
    assistant_resp = {"role": "assistant", "content": full_text}    
    if is_gen:
        msg_history.append(assistant_resp)
        new_key = hash(tuple(state['context']))
        utils.conversation_store[new_key] = msg_history
        if stream:
            yield json.dumps({"response": "", **stats_fields}) + "\n"
        else:
            yield json.dumps({"response": full_text, "context": state['context'], **stats_fields}) + "\n"
    else:
        if stream:
            yield json.dumps({"message": {"role": "assistant", "content": ""}, **stats_fields}) + "\n"
        else:
            yield json.dumps({"message": assistant_resp, **stats_fields}) + "\n"


async def try_server_tools(tool_calls: list) -> tuple[list[dict], list[dict]]:
    """
    Attempt to execute tool calls using local Python functions or internal MCP client.

    Returns:
        executed: list of {"role": "tool", "content": ...} messages for calls that were executed
                  (successful result or tool-reported error via MCP)
        to_forward: list of original tool_call dicts that could not be executed here
                    (local function crash, unknown tool name on MCP, connection failures, etc.)
    """
    executed = []
    to_forward = []
    server_tools = utils.MCP_CLIENT.available_tools | local_tools.LOCAL_TOOLS.keys() 
    for tc in tool_calls:
        fn = tc.get("function", {})
        tool_name = fn.get("name")
        tool_args = fn.get("arguments", {})
        exec_success = False
        tool_result_content = None
        if tool_name not in server_tools:
            to_forward.append(tc)
            continue
        
        if tool_name in local_tools.LOCAL_TOOLS:
            try:
                func = getattr(local_tools, tool_name)
                tool_result_content = await func(**tool_args)
                exec_success = True
            except Exception as local_exc:
                logging.error(f"Local execution of {tool_name} failed.")
                traceback.print_exc()
                tool_result_content = json.dumps({
                    'result': 'ERROR',
                    'content': f"Execution of {tool_name} failed: {local_exc}",
                })
        else:
            try:
                result = await utils.MCP_CLIENT.call_tool(tool_name, tool_args)
                tool_result_content = "\n".join(
                    block.text for block in result.content
                    if hasattr(block, "text")
                )
                exec_success = True
                
                if result.is_error:
                    # do NOT set exc_success to False here - the tool 
                    # *did* run, it just returned an error result.
                    logging.error(f"Tool {tool_name} returned error: {tool_result_content}")
                    
            except Exception as e:
                # Can't handle locally. Fall back to sending to the caller.
                logging.error(f"MCP tool call failed: {e}")
                
        if exec_success:
            executed.append({
                "role": "tool",
                "content": tool_result_content or "",
                "tool_call_id": tc.get("id"),
                # "name": name,
            })
        else:
            # Either never got a result, or got an error from MCP
            to_forward.append(tc)
            
    return executed, to_forward