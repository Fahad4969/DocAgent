import os
from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ],
)

print(response.choices[0].message.content)
print()
print(response.usage)