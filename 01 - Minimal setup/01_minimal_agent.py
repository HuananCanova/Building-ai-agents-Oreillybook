"""
=============================================================================
LESSON 01 — The Minimal "Agent" (it's NOT really an agent yet!)
=============================================================================

GOAL: Understand the raw LLM API call — the foundation of every AI agent.

KEY CONCEPTS:
  - System prompt: tells the model WHO it is (the user never sees this)
  - User message: the actual human question
  - Assistant message: the model's response
  - Temperature: 0 = deterministic (reliable), 1 = creative (random)
  - The model is STATELESS: it remembers nothing between calls.
    YOU must send the full conversation every time.

WHAT THIS IS:
  A "chatbot" — it takes input and produces output.
  It CANNOT take actions, use tools, or remember past conversations.
  
WHAT'S MISSING (we'll add these in later lessons):
  ❌ Memory (conversation history)
  ❌ Tools (functions the model can call)
  ❌ A loop (perceive → think → act → observe → repeat)
  
WHY START HERE?
  Because the LLM API call is the "brain" of every agent.
  If you don't understand this, nothing else will make sense.
=============================================================================
"""

import os
from dotenv import load_dotenv
from groq import Groq

# ─── Step 1: Load our API key from .env ─────────────────────────────────────
# load_dotenv() reads the .env file and puts GROQ_API_KEY into os.environ
# This keeps secrets out of our code (never hardcode API keys!)
load_dotenv()

# ─── Step 2: Create the Groq client ─────────────────────────────────────────
# The client handles HTTP connections to Groq's API servers
# It automatically reads GROQ_API_KEY from the environment
client = Groq()

# ─── Step 3: Define the messages ─────────────────────────────────────────────
# Messages are a LIST of dictionaries. Each has a "role" and "content".
# 
# Roles:
#   "system"    → Instructions for the model (invisible to the user)
#   "user"      → The human's message
#   "assistant" → The model's reply (we'll use this for memory in Lesson 02)
#
# The ORDER matters: system first, then the conversation in chronological order.

"""
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful customer support agent for an e-commerce store. "
            "You are friendly, concise, and always try to solve the customer's problem. "
            "If you don't know something, say so honestly."
        ),
    },
    {
        "role": "user",
        "content": "Hi! I ordered a laptop 3 days ago and it still hasn't shipped. Can you help?",
    },
]"""

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

# ─── Step 4: Call the LLM ────────────────────────────────────────────────────
# This is the actual API call. Let's break down each parameter:
#
#   model:       Which LLM to use. openai/gpt-oss-120b is available on Groq for free.
#   messages:    The conversation so far (system + user messages).
#   temperature: 0 = always give the same answer (deterministic).
#                1 = be creative/random. For agents, use 0!
#   max_tokens:  Maximum length of the response (in "tokens" ≈ words × 1.3)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    temperature=0,
    max_tokens=500,
)

# ─── Step 5: Extract and display the result ──────────────────────────────────
# The response object has a specific structure:
#   response.choices[0].message.content  → the actual text reply
#   response.choices[0].message.role     → always "assistant"
#   response.usage.prompt_tokens         → tokens in our input
#   response.usage.completion_tokens     → tokens in the model's reply

assistant_message = response.choices[0].message

print("=" * 60)
print("LESSON 01 — Minimal LLM Call")
print("=" * 60)
print()
print(f"Role: {assistant_message.role}")
print(f"Content: {assistant_message.content}")
print()
print("--- Token Usage ---")
print(f"Prompt tokens (our input):    {response.usage.prompt_tokens}")
print(f"Completion tokens (reply):    {response.usage.completion_tokens}")
print(f"Total tokens:                 {response.usage.total_tokens}")
print()
print("--- Why this is NOT an agent yet ---")
print("1. No memory: run this twice, it won't remember the first run")
print("2. No tools: it can't actually check your order status")
print("3. No loop: it answers once and stops")
print("   An agent would: check order -> find shipping info -> reply")
