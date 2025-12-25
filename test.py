import openai
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["voc-21984595115617039598676931a96233e0f2.26137335"],
    base_url=os.environ["https://openai.vocareum.com/v1"]
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)
