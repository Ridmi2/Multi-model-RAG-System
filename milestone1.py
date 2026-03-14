import anthropic
from dotenv import load_dotenv
import os

load_dotenv()  # Reads .env file

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain what a REST API is in exactly 2 sentences ."}
    ]
)

print(f"Response :{message.content[0].text}\n")
print(f"Input tokens:{message.usage.input_tokens}\n")
print(f"Output tokens:{message.usage.output_tokens}")