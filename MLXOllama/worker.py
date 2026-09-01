import asyncio
import copy
import datetime
import json
import logging
import queue
import time
import random
import re
import traceback

from dataclasses import dataclass
from typing import Any, cast

from mlx_vlm import generate, stream_generate, sample_utils
import mlx.core as mx

from . import utils, config, local_tools, tts_queue, cache_utils
from .cache_utils import get_cache
from .common import message_text
from .utils import inference_worker

alaskan_content = {
    "Alaskan Animals": [
        ("Grizzly Bear", "a massive brown bear often found fishing for salmon"),
        ("Grey Wolf", "a social apex predator that hunts in the vast wilderness"),
        ("Moose", "the largest member of the deer family with massive palmate antlers"),
        ("Bald Eagle", "a white-headed bird of prey common along the coast"),
        ("Orca", "the black-and-white 'killer whale' found in Alaskan fjords"),
        ("Caribou", "the arctic deer known for long-distance migrations"),
        ("Lynx", "a silent feline with tufted ears and large paws for snow"),
        ("Humpback Whale", "a massive marine mammal known for breaching and singing"),
        ("Dall Sheep", "a white, thin-horned sheep found on high mountain ridges"),
        ("Wolverine", "a small but incredibly fierce and solitary forest predator"),
        ("Musk Ox", "a prehistoric-looking herbivore with long, shaggy hair"),
        ("Sea Otter", "a furry marine mammal that floats on its back in kelp forests"),
        ("Puffin", "a colorful sea bird with a bright orange beak"),
        ("Walrus", "a heavy marine mammal with long ivory tusks"),
        ("Arctic Fox", "a small fox whose coat turns white in the winter"),
        ("Steller Sea Lion", "a massive, roaring pinniped found on rocky haul-outs"),
        ("Snowshoe Hare", "a rabbit whose fur changes color with the seasons"),
        ("Ptarmigan", "the state bird of Alaska, known for its feathered feet"),
        ("Beluga Whale", "a small, white whale found in the Cook Inlet"),
        ("Porpoise", "a fast-swimming black and white marine mammal"),
        ("Marmot", "a large ground squirrel found in rocky alpine areas"),
        ("Sitka Black-tailed Deer", "a small deer native to the rainforests of Southeast"),
        ("Mountain Goat", "a sure-footed white climber of the steep coastal peaks"),
        ("Harbor Seal", "a common, spotted seal often seen resting on ice floes"),
        ("Porcupine", "a slow-moving rodent covered in sharp defensive quills")
    ],
    "Alaskan Lifestyle Object": [
        ("Ulu", "a curved knife with a caribou antler handle used for subsistence"),
        ("Totem Pole", "a tall cedar monument carved with ancestral figures"),
        ("Birchbark Canoe", "a lightweight traditional Athabascan river vessel"),
        ("Lever-action Rifle", "a rugged firearm used for hunting and protection in the bush"),
        ("Atkuk", "a heavy sea-otter fur parka built for extreme arctic cold"),
        ("Mukluks", "traditional boots made of sealskin and decorated with beadwork"),
        ("Walrus Harpoon", "a historical hunting tool with a hand-carved ivory tip"),
        ("Baleen Basket", "a rare vessel woven from whale baleen with an ivory finial"),
        ("Bentwood Visor", "a wooden hunting hat carved to shield the eyes from sea glare"),
        ("Qiviut Scarf", "a garment made from the incredibly soft underwool of the musk ox"),
        ("Kuspuk", "a hooded Alaskan overshirt, often with a bright floral pattern"),
        ("Berry Basket", "a sturdy container made of woven birch bark for harvesting"),
        ("Gutskin Parka", "a traditional waterproof raincoat made from seal or bear intestines"),
        ("Moose-hide Mittens", "heavy winter gloves tanned from moose skin and beaded"),
        ("Dog Sled", "a wooden transport frame with ash runners for winter travel"),
        ("Soapstone Carving", "a heavy stone sculpture of a breaching whale or arctic animal"),
        ("Tlingit Bracelet", "a silver cuff carved with raven, eagle, or wolf crests"),
        ("Halibut Hook", "a large V-shaped hook carved from cedar and yew wood"),
        ("Dentalium Necklace", "jewelry made from white tusk-shaped sea shells"),
        ("Cradleboard", "a traditional wooden frame for carrying an infant on the back"),
        ("Snow Goggles", "walrus ivory with narrow slits to prevent snow blindness"),
        ("Spruce-root Hat", "a finely woven conical hat used in the rainy coastal regions"),
        ("Tináa", "a Tlingit ceremonial copper shield representing high status"),
        ("Taliq", "strips of dried, smoked salmon used as a staple food"),
        ("Qayaq", "a traditional driftwood-frame skin boat for maritime hunting"),
        ("Snowshoes", "large wooden frames with rawhide webbing for walking on deep drifts"),
        ("Devil's Club Salve", "a medicinal ointment made from the bark of a thorny tundra plant"),
        ("Spirit Mask", "a ceremonial whalebone mask adorned with raven feathers"),
        ("Oosik", "a polished and often carved walrus baculum bone"),
        ("Xtratuf Boots", "neoprene boots known as the unofficial footwear of coastal Alaska"),
        ("Labret", "a traditional lip ornament made of polished stone or bone"),
        ("Birch Syrup", "a sweet, earthy syrup tapped from local birch trees"),
        ("Salmon-skin Sewing Kit", "a pouch made of dried fish leather used for needlework"),
        ("Quillwork Vest", "a garment decorated with dyed porcupine quills"),
        ("Caribou Sleeping Mat", "a warm, thick hide used for bedding in winter camps"),
        ("Button Blanket", "a ceremonial wool robe decorated with mother-of-pearl buttons"),
        ("Cribbage Board", "a folk-art game board carved from a walrus ivory tusk"),
        ("Gold Nugget Pendant", "jewelry featuring raw gold found in Alaskan rivers"),
        ("Hudson Bay Tea", "dried leaves of the Labrador Tea plant used for brewing"),
        ("Baleen Tool", "a specialized scraper used for working with whale baleen"),
        ("Moose-antler Rack", "a functional wall hanger made from shed moose antlers"),
        ("Inupiaq Yo-yo", "two sealskin balls on a string used for traditional games"),
        ("Qulliq", "a traditional semi-circular stone oil lamp for heat and light"),
        ("Dance Fan", "a feathered or fur-trimmed fan used in traditional dancing"),
        ("Sinew Thread", "strong, traditional cordage made from animal tendons"),
        ("Fireweed Clip", "a modern hair accessory modeled after the purple tundra flower"),
        ("Fox Tail Hat", "a warm winter hat featuring a long arctic fox tail"),
        ("Lichen-dyed Blanket", "a wool blanket colored with dyes made from tundra moss"),
        ("Copper Earrings", "hand-hammered jewelry reflecting ancient metalworking"),
        ("Float Plane Model", "a miniature de Havilland Beaver or bush plane")
    ]
}

WEEKLY_STYLES = {
    "Monday": "the gentle, repetitive, and reassuring style of 'Goodnight Moon'",
    "Tuesday": "the soft, comforting style of a classic woodland animal bedtime tale",
    "Wednesday": "the dreamy, celestial style of a poem about the night sky and drifting stars",
    "Thursday": "the warm, rustic style of a cozy fireside cabin settling down for a long winter night",
    "Friday": "the quiet, whispering style of gentle snowfall blanketed over the quiet woods",
    "Saturday": "the peaceful style of a tired traveler or musher resting by a calm, frozen river",
    "Sunday": "the lulling, melodic style of a soft northern breeze singing the trees to sleep"
}

def gen_goodnight(prompt):
    t0 = time.time()
    logging.info("Goodnight generation beginning")

    current_month = datetime.datetime.now().strftime("%B")
    day_of_year = datetime.datetime.now().timetuple().tm_yday

    category_pick = random.choice(list(alaskan_content.keys()))
    logging.info(f"Selected category for goodnight content: {category_pick}")
    item_pick, item_description = random.choice(alaskan_content[category_pick])

    today_name = datetime.datetime.now().strftime("%A")
    style_pick = WEEKLY_STYLES[today_name]

    logging.info(f"Selected item for goodnight content: {item_pick} - {item_description}")

    dynamic_info = f"""
### INPUT DATA
SPECIAL_ITEM = {item_pick}
ITEM_DESCRIPTION = {item_description}
The day of the year is: {day_of_year}
CURRENT_MONTH = {current_month}
POEM_STYLE = {style_pick}
-------------------"""

    message = [
        {"role": "system","content": "###BEDTIME###",},
        {"role": "user","content": prompt + dynamic_info,}
    ]

    sampler = sample_utils.make_sampler(
        temp=0.95,
        top_p=0.92
    )

    token_queue: queue.Queue = cast(queue.Queue, submit_inference_job(message, sampler=sampler))
    
    remainder = ""
    body_sentences = []
    sentence_index = 0
    first_sentence = -1
     
    while True:
        result = token_queue.get()
        text, token, is_done, err = result
        if err:
            logging.error(f"Unable to generate result: {err}")
            remainder = "I'm sorry, but I was unable to generate a goodnight poem. Check the logs for more information."
            break        
        
        remainder += text
        if '\n' in remainder:
            head, *tail = remainder.split('\n')

            if sentence_index == 0:
                first_sentence = time.time() - t0
            if sentence_index < 2:
                tts_queue.put((head.strip(), 0.8, 'af_bella'))
            else:
                body_sentences.append(head)
            sentence_index += 1

            remainder = "\n".join(tail) # Handles empty automatically
            
        if is_done:
            break

    if remainder.strip():
        body_sentences.append(remainder)
        body_text = "\n".join(body_sentences)
        body_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', body_text)
        tts_queue.put((body_text, 0.8, 'af_bella'))

    tts_queue.put("__FLUSH__")

    logging.info(f"Got goodnight text in {time.time() - t0}. First sentence in {first_sentence}")

    return

def speak_thread(speak_queue):
    logging.info("Listening for prompts")

    while True:
        # mx.clear_cache()
        # gc.collect()
        # mx.clear_cache()
        try:
            try:
                msg = speak_queue.get(timeout=300) # 5 minutes
            except queue.Empty:
                run_dummy_inference()
                continue

            if isinstance(msg, tuple):
                system, prompt = msg
            else:
                system: str|None = None
                prompt: str = msg

            if prompt == "QUIT":
                break
            if prompt == "UPDATE":
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

            if system:
                message = [
                    {"role": "system","content": system}
                ] + message

            try:
                buffer = ""
                token_queue: queue.Queue = cast(
                    queue.Queue,
                    submit_inference_job( # Submits to another, dedicated, inference thread
                        message,
                        thinking=use_thinking,
                        sampler=utils.FUN_SAMPLER,
                        max_kv_size=2048,
                        max_tokens=4096)
                )

                is_thinking = False
                sentence_count = 0
                while True:
                    result = token_queue.get()
                    token_text, token, is_done, err = result
                    if err:
                        logging.error(f"Unable to generate result: {err}")
                        buffer = "I'm sorry, but an error occured. Please check the logs for more information."
                        break

                    # Check for state changes
                    if "<think>" in token_text or '<channel|>' in token_text:
                        is_thinking = True
                        # Strip the tag itself if it's bundled with other text
                        token_text = token_text.replace("<think>", "")
                        token_text = token_text.replace('<channel|>', "")

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
                        if is_done:
                            break
                        else:
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

                    if is_done: # failsafe, but we shouldn't get here.
                        break

                # Flush remaining text
                if buffer:
                    tts_queue.put(buffer)
            finally:
                tts_queue.put("__FLUSH__")
                logging.info(f"Completed inference in {time.time() - t1}")
        except Exception as e:
            logging.exception(f"Error in speak thread: {e}")

    logging.info("Prompt processing thread exited")

@dataclass
class InferenceOptions:
    model: Any
    processor: Any
    prompt_tokens: list
    cache: Any
    all_tokens:list
    model_name: str
    images: list | None = None

async def setup_inference(
    message: list,
    *,
    tools: list | None = None,
    thinking: bool = False,
    model_info: dict | None = None,
    images: list | None = None
) -> InferenceOptions:
    if model_info is None:
        model_info = utils.loaded_models[config.MAIN_MODEL]
    model = model_info['model']
    processor = model_info['processor']

    cache, unprocessed_tokens, all_tokens = await get_cache(
        model_info, message, tools, thinking, images=images
    )
    cache = copy.copy(cache)
    return InferenceOptions(
        model,
        processor,
        unprocessed_tokens,
        cache,
        all_tokens,
        model_info['name'],
        images,
    )
    
def submit_inference(
    opts: InferenceOptions,
    *, 
    sampler=None,
    max_kv_size=None,
        max_tokens:int = 256, # Default from mlx-vlm
    loop: asyncio.AbstractEventLoop | None = None    
) -> queue.Queue | asyncio.Queue:
    
    token_queue = asyncio.Queue() if loop else queue.Queue()
        
    def put(item):
        if loop:
            loop.call_soon_threadsafe(token_queue.put_nowait, item)
        else:
            token_queue.put_nowait(item)    

    def thread_worker():
        model = opts.model
        processor = opts.processor
        with utils.tokenizer_lock:
            formatted_prompt = opts.processor.tokenizer.decode(opts.prompt_tokens)
        cache = opts.cache
        all_tokens=opts.all_tokens
        try:
            with utils.mlx_inference_lock:
                mx.random.seed(int(time.time()))
                # This runs in a worker thread
                # stream_generate may use the processor/tokenizer internally;
                # keep it under the same lock as request-side encode/decode.
                with utils.tokenizer_lock:
                    for response in stream_generate(
                        model,
                        processor,
                        prompt=formatted_prompt,
                        image=opts.images,
                        sampler=sampler,
                        prompt_cache=cache,
                        max_tokens=max_tokens,
                        max_kv_size=max_kv_size
                    ):
                        put((response.text, response.token, False, None))

                cache_utils.dynamic_cache.insert_cache(
                    opts.model_name,
                    all_tokens,
                    cache
                )
            put(("", None, True, None))

        except Exception as e:
            logging.exception(
                "mlx-vlm stream_generate failed (image_count=%d, prompt_chars=%d)",
                len(opts.images or []),
                len(formatted_prompt),
            )
            put((None, None, True, e))

    inference_worker.submit(thread_worker)
    return token_queue

def submit_inference_job(
    message: list,
    *,
    tools=None,
    thinking=False,
    sampler=None,
    max_kv_size=None,
    max_tokens:int=256, # Default from mlx-vlm
    loop: asyncio.AbstractEventLoop | None = None
) -> queue.Queue | asyncio.Queue:
    opts = asyncio.run(setup_inference(message, tools=tools, thinking=thinking))
    return submit_inference(
        opts,
        sampler=sampler,
        max_kv_size=max_kv_size,
        max_tokens=max_tokens,
        loop=loop
    )

def run_dummy_inference():
    """Run the fastest possible inference, just to keep things alive/in ram"""
    for mod_name, mod_info in utils.loaded_models.items():
        model = mod_info['model']
        processor= mod_info['processor']
        t1 = time.time()

        WARMUP_PROMPT = """
{"domain": "home_automation", "action": "set_device", "entity_id": "light.living_room", "state": "on"} 
Good morning! Today is a clear day with scheduled tasks. Please review the upcoming calendar events and summarize the weather forecast.
"""

        def thread_worker():
            with utils.mlx_inference_lock:
                generate(
                    model,
                    processor,
                    prompt=WARMUP_PROMPT,
                    max_tokens=1,
                    verbose=False,
                )

                mx.synchronize()
        future = inference_worker.submit(thread_worker)
        
        future.result()

        if time.time() - t1 > 10:
            logging.warning(f"Dummy inference for {mod_name} took {time.time() - t1:.2f} seconds, which is quite long. Running a full refresh inference to keep the model warm.")
            full_refresh_model(mod_name, mod_info)

        logging.info(f"Ran keep-alive inference for {mod_name} in {time.time() - t1}")

def full_refresh_model(mod_name, mod_info):
    """Run a full inference to refresh the model in memory. This is more intensive than the dummy, but can help with performance after an extended idle period."""
    model = mod_info['model']
    t1 = time.time()

    def thread_worker():
        with utils.mlx_inference_lock:
            mx.eval(model.parameters())
            mx.synchronize()

    future = inference_worker.submit(thread_worker)
    future.result()

    logging.info(f"Ran full refresh inference for {mod_name} in {time.time() - t1}")

async def _stream_tokens(
    model_info: dict, msg_history: list[dict],
    options: dict, state: dict|None=None,
    tools: list|None = None,
    think: bool = False,
    images: list | None = None
):
    """
    Yields (token_str, is_done, stats) tuples.
    Runs MLX in a thread to avoid blocking the event loop.
    Adapt this to however your existing pipeline streams tokens.
    """
    if state is None:
        state = {}

    t0 = time.time_ns()
    eval_count = 0
    try:
        
        # OpenAI vision requests represent content as text/image parts rather
        # than the string used by Ollama requests.
        if "### task:" in message_text(msg_history[-1].get("content", "")).lower():
            options['temp'] = 0.1
            think = False
            options['max_tokens'] = 256
            
        opts = await setup_inference(
            msg_history,
            tools=tools,
            thinking=think,
            model_info=model_info,
            images=images,
        )

        sampler = sample_utils.make_sampler(
            temp=options.get("temp", 0.65),
            min_p=0.05,
            xtc_probability=0.5,
            xtc_threshold=0.1
        )

        prompt_tokens=opts.prompt_tokens
        tokenizer = opts.processor.tokenizer
        formatted_prompt = tokenizer.decode(prompt_tokens)

        if formatted_prompt.strip().endswith("<think>"):
            yield "<think>\n", False, None

        if "<|channel>" in formatted_prompt:
            yield "<|channel>" + formatted_prompt.rsplit("<|channel>")[1], False, None

        with utils.tokenizer_lock:
            suffix_tokens = tokenizer.encode(formatted_prompt)

        logging.info(f"{len(suffix_tokens)} Tokens encoded")

        available_space = 262144 - len(suffix_tokens) - 100
        safe_max = min(options.get("max_tokens", 64000), available_space)

        full_output_tokens = []
        
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue = cast(
            asyncio.Queue,
            submit_inference(
                opts,
                sampler=sampler,
                max_tokens=safe_max,
                loop=loop)
        )

        while True:
            result = await token_queue.get()

            text, token, is_done, err = result

            if isinstance(err, Exception):
                # Optional: yield partial + log, then re-raise or swallow
                logging.error("Inference error: %s (%r)", str(err), err)
                logging.error("Inference traceback:\n%s", "".join(
                    traceback.format_exception(type(err), err, err.__traceback__)
                ))
                yield "", True, {"error": str(err) or repr(err)}
                break
            if is_done:
                break

            eval_count += 1
            full_output_tokens.append(token)
            yield text, False, None

        with utils.tokenizer_lock:
            prompt_tokens = tokenizer.encode(formatted_prompt)
        state['context'] = prompt_tokens + full_output_tokens
        state['eval_count'] = len(prompt_tokens)
    except Exception as e:
        logging.exception(f"Unable to generate output ({e})")
        state['context'] = None
        state['eval_count'] = 0

    elapsed = time.time_ns() - t0
    yield "", True, {"eval_count": eval_count, "eval_duration": elapsed}

async def generate_stream(stream, model_info, msg_history, options,
                          is_gen=False, tools=None, think=False, images=None):
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

        try:
            async for token, done, stats in _stream_tokens(
                model_info, msg_history, options, state, tools=tools, think=think,
                images=images
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
                            if think:
                                token = buffer.replace('<channel|>', '</thought>\n')
                            else:                                
                                buffer = re.sub(r'(<\|channel>.*?)?<channel\|>', '', buffer, flags=re.DOTALL)
                                token = buffer
                            buffer = ""
                        elif "<|channel>" in buffer:
                            if think:
                                token = buffer.replace('<|channel>', '<thought>\n')
                                buffer = ""
                            else:
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
        except Exception as e:
            logging.exception(f"Unable to finish response: {e}")
        except asyncio.exceptions.CancelledError:
            logging.warning(f"Request Canceled")

        logging.info(f"Generated response in {time.time() - t2}")
        logging.info(f"Raw Response: {full_text}")

        tool_calls,cleaned_text = utils.parse_tool_calls(full_text)

        if not tool_calls or is_gen:
            break
        
        if "<channel|>" in cleaned_text:
            if think:
                # Replace channel with thought
                cleaned_text = cleaned_text.replace('<channel|>', '</thought>\n')
                cleaned_text = cleaned_text.replace('<|channel>', '<thought>\n')
            else:
                # Remove everything in the channel tags
                cleaned_text = re.sub(r'(<\|channel>.*?)?<channel\|>', '', cleaned_text, flags=re.DOTALL)       

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
        "total_duration": total_time,
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
