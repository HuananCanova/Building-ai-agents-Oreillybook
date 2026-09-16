"""
=============================================================================
LESSON 02 - The Conversation Loop (Adding Memory)
=============================================================================

GOAL: Turn our one-shot LLM call into an interactive chatbot with memory.

WHAT WE'RE ADDING:
  + Memory: the model now "remembers" previous messages
  + A loop: the conversation continues until the user says "quit"
  + History management: we append every message to a growing list

HOW MEMORY WORKS:
  The LLM itself has NO memory. It's stateless.
  The "trick" is simple: we store every message (user + assistant) in a list,
  and send the ENTIRE list with every API call. The model reads the full
  conversation and responds as if it "remembers" everything.

  Call #1: [system, user_msg_1]                       -> assistant_reply_1
  Call #2: [system, user_msg_1, assistant_reply_1, user_msg_2] -> assistant_reply_2
  Call #3: [system, user_msg_1, assistant_reply_1, user_msg_2, assistant_reply_2, user_msg_3] -> ...

  Notice: the list GROWS with every turn. This means:
    - More tokens used (= more cost and latency)
    - Eventually you'll hit the model's context window limit
    - Real agents use strategies like summarization or sliding windows

THE AGENT LOOP:
  while True:
      1. Get user input          (PERCEIVE)
      2. Append to history       (REMEMBER)
      3. Send history to LLM     (THINK)
      4. Get response            (ACT)
      5. Append response         (REMEMBER)
      6. Display to user         (OUTPUT)
=============================================================================
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Fix Windows console encoding (Windows uses cp1252 by default,
# which can't display many Unicode characters that LLMs produce)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

# ---- Setup (same as Lesson 01) ----
# .env is in the project root (one folder up), so we point to it explicitly
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
client = Groq()
MODEL = "openai/gpt-oss-120b"

# ---- System prompt ----
# We define this once. It stays at the FRONT of the messages list forever.
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful customer support agent for an e-commerce store called 'TechShop'. "
        "You are friendly, concise, and professional. "
        "You help customers with orders, shipping, returns, and product questions. "
        "If you don't know something, say so honestly. "
        "Keep responses short (2-3 sentences max)."
    ),
}

# ---- The conversation history ----
# This is our "memory". It starts with ONLY the system prompt.
# Every user message and assistant reply gets appended here.
conversation_history = [SYSTEM_PROMPT]


def chat(user_message: str) -> str:
    """
    Send a message to the LLM and get a response.
    
    This function does 4 things:
      1. Appends the user's message to history
      2. Sends the FULL history to the LLM
      3. Appends the assistant's reply to history
      4. Returns the reply text
    
    After this function runs, conversation_history has grown by 2 entries:
    the user message and the assistant reply.
    """
    
    # Step 1: Add the user's message to our history
    conversation_history.append({
        "role": "user",
        "content": user_message,
    })
    
    # Step 2: Send the ENTIRE history to the LLM
    # This is the key insight: we send EVERYTHING, not just the last message.
    # The model reads the whole conversation and generates a contextual reply.
    response = client.chat.completions.create(
        model=MODEL,
        messages=conversation_history,  # <-- THE FULL HISTORY!
        temperature=0,
        max_tokens=300,
    )
    
    # Step 3: Extract the assistant's reply
    assistant_message = response.choices[0].message.content
    
    # Step 4: Add the assistant's reply to our history too!
    # Next time we call the API, it will see this reply as context.
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message,
    })
    
    return assistant_message


# ---- The main loop ----
def main():
    print("=" * 60)
    print("LESSON 02 - Conversation Loop (with Memory)")
    print("=" * 60)
    print("Chat with TechShop support! Type 'quit' to exit.")
    print("Type 'history' to see the raw conversation history.")
    print("-" * 60)
    
    while True:
        # PERCEIVE: get user input
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("\nGoodbye! Thanks for chatting.")
            break
        
        # BONUS: Let the user peek at the raw history to understand what's happening
        if user_input.lower() == "history":
            print("\n--- Raw Conversation History ---")
            for i, msg in enumerate(conversation_history):
                role = msg["role"].upper()
                content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                print(f"  [{i}] {role}: {content}")
            print(f"  Total messages: {len(conversation_history)}")
            print("--- End History ---")
            continue
        
        # THINK + ACT: send to LLM, get response
        reply = chat(user_input)
        
        # OUTPUT: display the response
        print(f"\nAgent: {reply}")


if __name__ == "__main__":
    main()
