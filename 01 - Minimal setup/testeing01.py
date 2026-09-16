from urllib import response
import os
from dotenv import load_dotenv
from groq import Groq

#load dotenv
load_dotenv()

#Call client so we can use the API
client = Groq()

#Define messages - three roles system, user and assistant
messages = [
    {
        "role": "system",
        "content": (
            "You're a helpfull musician teacher that knows a lot about eletronic music. "
            "Recommend musics related when possible. "
            "You're friendly and concise. "
            "If you don't know something, say so honestly"
        )
    },
    {
        "role": "user",
        "content": "Hi can you tell me what music style Klipsunmusic plays in detail."
    }
]


#Call the actual API
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    temperature=0,
    max_tokens=500,
)



#Extract response

content = response.choices[0].message.content
role = response.choices[0].message.role

print (f"Role: {role}\nContent: {content}\n".encode("utf-8", errors="replace").decode("utf-8"))
