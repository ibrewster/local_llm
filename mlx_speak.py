import logging
import os
import pty
import queue
import re
import select
import subprocess
import time
import threading

from logging.handlers import RotatingFileHandler

import flask

from mlx_lm import load, generate, stream_generate, sample_utils

threading.current_thread().name = "MLX"

request_queue = queue.Queue()

sentence_endings_eager = re.compile(
    r'(?<!\w\.\w)'          # not mid-abbreviation like U.S.A.
    r'(?<![A-Z][a-z]\.)'   # not after titles: Mr. Dr. St. etc.
    r'[.!?]+'               # sentence-ending punctuation
    r'(?!\d)'               # not before a digit (avoids 2.5, 3.14, etc.)
    r'(?=\s|$)'             # followed by whitespace or end of buffer
)

sentence_endings_conservative = re.compile(
    r'(?<!\w\.\w)'          # not mid-abbreviation like U.S.A.
    r'(?<![A-Z][a-z]\.)'   # not after titles: Mr. Dr. St. etc.
    r'(?<!\d)'              # not after digit...
    r'[.!?]+'
    r'(?!\d)'               # ...or before digit (together these protect 2.5, -8.2)
    r'(?=\s+[A-Z]|$)'      # must be followed by uppercase or end of buffer
)

#model_path = "mlx-community/Qwen3-8B-4bit"
model_path = "mlx-community/Qwen3-30B-A3B-4bit"
quick_model = "mlx-community/Qwen2.5-3B-Instruct-4bit"
# model_path = "mlx-community/Qwen3-4B-Instruct-2507-4bit" # Faster, but not as good?

def llm_thread():
    print("Listening for prompts")
    while True:
        try:
            prompt = request_queue.get(timeout=600) # 10 minutes
        except queue.Empty:
            run_dummy_inference()
            continue
        
        if prompt == "QUIT":
            break
        
        t1=time.time()
        message = [
            {"role": "user","content": prompt,}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Uncomment if your mlx-lm version supports it (recent ones do for Qwen3)
        )
        
        try:
            master_fd, slave_fd = pty.openpty()
        
            say_proc = subprocess.Popen(
                ['say', '-a', 'BlackHole 2ch'], 
                stdin=slave_fd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True
            )
        
            os.close(slave_fd)
    
            buffer = ""
            
            first=True
            stream = stream_generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                sampler=SAMPLER,
                max_tokens=2048
            )
            
            for chunk in stream:
                token_text = chunk.text
                
                # Optional: filter any stray <think> tags
                content = token_text.replace("<think>", "").replace("</think>", "")
                if not content:
                    continue
                buffer += content
                
                pattern = sentence_endings_eager if first else sentence_endings_conservative
                matches = list(pattern.finditer(buffer))
                
                if matches:
                    last_match = matches[-1]
                    split_point = last_match.end()
                    
                    complete = buffer[:split_point]
                    buffer = buffer[split_point:]
                    
                    # if len(complete.strip()) > 40:  # prevents tiny speech bursts
                    if first:
                        first=False
                        app.logger.info(f"Time to first sentence: {time.time()-t1}")
                    app.logger.debug(complete.strip())
                    _, ready_to_write, _ = select.select([], [master_fd], [], 30.0)
                    if ready_to_write:
                        os.write(master_fd, (complete.strip() + "\n").encode("utf-8"))
    
            # Flush remaining text
            if buffer:
                print(buffer+"\n")
                _, ready_to_write, _ = select.select([], [master_fd], [], 30.0)
                if ready_to_write:
                    os.write(master_fd, (buffer + "\n").encode('utf-8'))
                    
            os.write(master_fd, b"\n")
    
        finally:
            os.write(master_fd, b'\x04') # EOF
            app.logger.info(f"Completed inference in {time.time() - t1}")
            say_proc.wait()
            os.close(master_fd)
            app.logger.info(f"Completed speaking in {time.time() - t1}")
            
    app.logger.info("Prompt processing thread exited")

############################

### Set up logging
def init_logging():
    log_path = "/var/log/tts/server.log"
        
    handler = RotatingFileHandler(
        log_path,
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-5s %(threadName)s %(message)s',
        datefmt='%d/%b/%Y:%H:%M:%S %z'
    )
    
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)    
    
init_logging()

logging.info("Loading model...")

model, tokenizer = load(model_path)
SAMPLER = sample_utils.make_sampler(
    temp=0.8,
    top_p=0.9,
)

logging.info("Model loaded and ready!")

process_thread = threading.Thread(target = llm_thread, name="Worker")
process_thread.start()

app = flask.Flask(__name__)
app.logger.handlers = []
app.logger.propagate = True

@app.route('/generate', methods = ['POST'])
def generate_response():
    prompt = flask.request.json['prompt']
    request_queue.put(prompt)
    return "Accepted", 202

def run_dummy_inference():
    """Run the fastest possible inference, just to keep things alive/in ram"""
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

    _ = generate(
        model,
        tokenizer,
        prompt=formatted,
        sampler=sampler,
        max_tokens=1
    )
    app.logger.info(f"Ran keep-alive inference in {time.time() - t1}")
    

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080)
    app.logger.info("Server shutting down")
    request_queue.put("QUIT")
    process_thread.join()
