from openai import OpenAI

# Point the client to your local endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1", # Update with your specific port and path
    api_key="sk-local" # Usually ignored by local layers, but the library requires a string
)

response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it-mxfp4", # Some layers require the exact model name, others ignore it
    messages=[{"role": "user", "content": "Hello! Are you receiving this?"}]
)

print(response.choices[0].message.content)