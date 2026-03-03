import anthropic

client = anthropic.Anthropic(api_key="")

#this is the API call(REST request)
message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    messages=[
        {"role":"user","content":"What is 2+2? Answer in one sentence. "}
    ]
)

#the response is an object - text
print(message.content[0].text)