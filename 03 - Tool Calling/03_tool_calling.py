"""
=============================================================================
LESSON 03 - Tool Calling (Giving the Agent Hands)
=============================================================================

GOAL: Let the LLM call Python functions to take REAL actions.
      This is the moment our chatbot becomes a true agent.

WHAT WE HAD BEFORE (Lessons 01 & 02):
  ✅ LLM API call (the "brain")
  ✅ Memory (conversation history)
  ✅ A loop (continuous conversation)
  ❌ Tools ← THIS IS WHAT WE'RE ADDING

WHY TOOLS MATTER:
  Without tools, the model can only TALK. It can say "let me check your 
  order" but it can't actually check anything. It's like a customer support 
  agent with no access to the computer system — they can be polite, but 
  they can't help you.

  With tools, the model can:
    - Look up information in databases
    - Cancel or modify orders
    - Send emails
    - Call external APIs
    - Anything a Python function can do!

HOW TOOL CALLING WORKS (the key insight):
  The LLM does NOT execute code. It NEVER runs your functions directly.
  Instead, the process has TWO phases:

  Phase 1: LLM DECIDES what to call
    - You describe your functions (name, description, parameters)
    - The LLM reads the user's message
    - The LLM outputs a structured JSON saying: "call function X with args Y"
    - This is called a "tool call" — it's just text, not execution

  Phase 2: YOUR CODE executes the function
    - You parse the tool call from the LLM's response
    - You run the actual Python function with those arguments
    - You send the result BACK to the LLM as a "tool message"
    - The LLM reads the result and formulates a human-friendly response

  Visually:
    User: "What's the status of order 42?"
        ↓
    LLM thinks: "I should call get_order_status with order_id=42"
        ↓
    LLM outputs: tool_call(name="get_order_status", args={"order_id": "42"})
        ↓  (this is just JSON — nothing has executed yet!)
    YOUR CODE: result = get_order_status("42")  → "shipped"
        ↓
    You send result back to LLM as a tool message
        ↓
    LLM: "Your order #42 has been shipped! 🚚"

THE AGENT LOOP (upgraded):
  while True:
      1. Get user input                        (PERCEIVE)
      2. Append to history                     (REMEMBER)
      3. Send history + tool definitions       (THINK)
      4. IF LLM wants to call a tool:
         a. Execute the function               (ACT)
         b. Send result back to LLM            (OBSERVE)
         c. LLM generates final response       (THINK again)
      5. Display to user                       (OUTPUT)
=============================================================================
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

# ---- Setup (same as before) ----
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
client = Groq()
MODEL = "openai/gpt-oss-120b"


# =============================================================================
# STEP 1: Define our tools (just regular Python functions!)
# =============================================================================
# These are the "hands" of our agent. Each function does one specific thing.
# The LLM will DECIDE which one to call based on the user's message.

# Our fake "database" — in a real app this would be a real DB or API
ORDERS_DB = {
    "1001": {"product": "Mechanical Keyboard", "status": "shipped",      "delivery_date": "2025-02-15", "price": 89.99},
    "1002": {"product": "Wireless Mouse",      "status": "processing",   "delivery_date": "2025-02-20", "price": 34.99},
    "1003": {"product": "USB-C Hub",            "status": "delivered",    "delivery_date": "2025-02-10", "price": 49.99},
    "1004": {"product": "Laptop Stand",         "status": "cancelled",    "delivery_date": None,         "price": 29.99},
}

PRODUCTS_DB = {
    "Mechanical Keyboard": {"price": 89.99, "stock": 15, "category": "Peripherals"},
    "Wireless Mouse":      {"price": 34.99, "stock": 42, "category": "Peripherals"},
    "USB-C Hub":           {"price": 49.99, "stock": 8,  "category": "Accessories"},
    "Laptop Stand":        {"price": 29.99, "stock": 0,  "category": "Accessories"},
    "Webcam HD":           {"price": 59.99, "stock": 23, "category": "Peripherals"},
    "Monitor Light Bar":   {"price": 44.99, "stock": 5,  "category": "Accessories"},
}


def get_order_status(order_id: str) -> str:
    """Look up the status of an order by its ID."""
    order = ORDERS_DB.get(order_id)
    if not order:
        return json.dumps({"error": f"Order #{order_id} not found. Valid IDs are: {', '.join(ORDERS_DB.keys())}"})
    
    result = {
        "order_id": order_id,
        "product": order["product"],
        "status": order["status"],
        "delivery_date": order["delivery_date"],
        "price": order["price"],
    }
    return json.dumps(result)


def cancel_order(order_id: str) -> str:
    """Cancel an order. Only works if the order is still processing."""
    order = ORDERS_DB.get(order_id)
    if not order:
        return json.dumps({"error": f"Order #{order_id} not found."})
    
    if order["status"] == "cancelled":
        return json.dumps({"error": f"Order #{order_id} is already cancelled."})
    
    if order["status"] == "shipped":
        return json.dumps({"error": f"Order #{order_id} has already shipped. Cannot cancel. Please use returns instead."})
    
    if order["status"] == "delivered":
        return json.dumps({"error": f"Order #{order_id} has been delivered. Cannot cancel. Please use returns instead."})
    
    # Only "processing" orders can be cancelled
    order["status"] = "cancelled"
    order["delivery_date"] = None
    return json.dumps({"success": True, "message": f"Order #{order_id} has been cancelled. Refund of ${order['price']} will be processed in 3-5 business days."})


def check_product_availability(product_name: str) -> str:
    """Check if a product is in stock and its price."""
    # Simple case-insensitive search
    for name, info in PRODUCTS_DB.items():
        if product_name.lower() in name.lower():
            return json.dumps({
                "product": name,
                "price": info["price"],
                "in_stock": info["stock"] > 0,
                "stock_count": info["stock"],
                "category": info["category"],
            })
    
    # Not found — return available products to help
    available = list(PRODUCTS_DB.keys())
    return json.dumps({"error": f"Product '{product_name}' not found. Available products: {', '.join(available)}"})


# =============================================================================
# STEP 2: Define tool SCHEMAS for the LLM
# =============================================================================
# The LLM doesn't see your Python code. Instead, you provide a JSON Schema
# describing each tool: its name, what it does, and what parameters it expects.
#
# Think of this as a "menu" — the LLM reads the menu and decides what to order.
# The KEY fields:
#   - name:        must match your Python function name exactly
#   - description: helps the LLM decide WHEN to use this tool (be descriptive!)
#   - parameters:  JSON Schema for the function's arguments

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the current status, delivery date, and details of a customer order by its order ID number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID number, e.g. '1001'",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel a customer's order. Only works for orders that are still in 'processing' status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID number to cancel, e.g. '1002'",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_product_availability",
            "description": "Check if a specific product is currently in stock and get its price. Use when a customer asks about product availability or pricing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name (or partial name) of the product to search for, e.g. 'keyboard' or 'mouse'",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
]


# =============================================================================
# STEP 3: Map function names to actual Python functions
# =============================================================================
# When the LLM says "call get_order_status", we need to find the actual function.
# This dictionary is that lookup table.

available_tools = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
    "check_product_availability": check_product_availability,
}


# ---- System prompt (updated to mention tools) ----
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful customer support agent for an e-commerce store called 'TechShop'. "
        "You are friendly, concise, and professional. "
        "You have access to tools to look up orders, cancel orders, and check product availability. "
        "ALWAYS use the appropriate tool when a customer asks about an order or product — "
        "never make up order information. "
        "Keep responses short (2-3 sentences max)."
    ),
}

# ---- Conversation history ----
conversation_history = [SYSTEM_PROMPT]


# =============================================================================
# STEP 4: The chat function — now with tool calling!
# =============================================================================
# This is the BIG upgrade from Lesson 02. The flow is:
#
#   1. Send messages + tool definitions to the LLM
#   2. Check if the LLM wants to call any tools
#   3. If yes: execute the tool, send result back, get final response
#   4. If no: just return the text response (simple chat, no tool needed)

def chat(user_message: str) -> str:
    """
    Send a message and handle any tool calls the LLM makes.
    
    Returns the final assistant response text.
    """
    
    # Step 1: Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message,
    })
    
    # Step 2: First LLM call — "What should I do with this message?"
    # We pass `tools=tools` so the LLM knows what tools are available.
    response = client.chat.completions.create(
        model=MODEL,
        messages=conversation_history,
        tools=tools,           # <-- NEW! Tell the LLM about our tools
        tool_choice="auto",    # <-- "auto" = LLM decides if a tool is needed
        temperature=0,
        max_tokens=500,
    )
    
    # Step 3: Get the response message
    response_message = response.choices[0].message
    
    # Step 4: Check if the LLM wants to call any tools
    if response_message.tool_calls:
        # ========================================
        # THE TOOL CALLING LOOP
        # ========================================
        # The LLM decided it needs to use one or more tools.
        # We must:
        #   a) Add the LLM's response (with tool_calls) to history
        #   b) Execute each tool call
        #   c) Add each result as a "tool" message
        #   d) Call the LLM AGAIN to generate the final human-friendly response
        
        print(f"\n  🔧 Agent is using {len(response_message.tool_calls)} tool(s)...")
        
        # a) Add the assistant's "I want to call tools" message to history
        conversation_history.append(response_message)
        
        # b) Execute each tool call
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"  → Calling: {function_name}({function_args})")
            
            # Look up the actual Python function and call it
            function_to_call = available_tools[function_name]
            result = function_to_call(**function_args)
            
            print(f"  ← Result:  {result}")
            
            # c) Add the tool result to history
            # IMPORTANT: tool_call_id links this result back to the specific tool call
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": result,
            })
        
        # d) Second LLM call — "Now that you have the tool results, respond to the user"
        # This time the LLM reads the tool results and writes a human-friendly answer
        print("  🤖 Generating final response...\n")
        
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=conversation_history,
            tools=tools,
            temperature=0,
            max_tokens=500,
        )
        
        final_message = second_response.choices[0].message.content
        conversation_history.append({
            "role": "assistant",
            "content": final_message,
        })
        
        return final_message
    
    else:
        # ========================================
        # NO TOOL CALL — simple text response
        # ========================================
        # The LLM didn't need any tools (e.g., "Hello!" or "You're welcome!")
        
        assistant_message = response_message.content
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })
        
        return assistant_message


# =============================================================================
# STEP 5: The main loop (same structure as Lesson 02)
# =============================================================================
def main():
    print("=" * 60)
    print("LESSON 03 - Tool Calling (The Agent Gets Hands!)")
    print("=" * 60)
    print("Chat with TechShop support — now with REAL tools!")
    print()
    print("Try these:")
    print("  • 'What's the status of order 1001?'")
    print("  • 'I want to cancel order 1002'")
    print("  • 'Do you have any keyboards in stock?'")
    print("  • 'Cancel order 1001' (should fail — already shipped!)")
    print()
    print("Type 'quit' to exit, 'history' to see conversation.")
    print("-" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("\nGoodbye! Thanks for chatting.")
            break
        
        if user_input.lower() == "history":
            print("\n--- Raw Conversation History ---")
            for i, msg in enumerate(conversation_history):
                # Handle different message types
                if hasattr(msg, "role"):
                    # This is a ChatCompletionMessage object (from tool calls)
                    role = msg.role.upper()
                    if msg.tool_calls:
                        calls = [f"{tc.function.name}({tc.function.arguments})" for tc in msg.tool_calls]
                        content = f"[TOOL CALLS: {', '.join(calls)}]"
                    else:
                        content = msg.content[:80] + "..." if msg.content and len(msg.content) > 80 else (msg.content or "")
                else:
                    # This is a regular dict
                    role = msg["role"].upper()
                    content = msg.get("content", "")[:80]
                    if len(msg.get("content", "")) > 80:
                        content += "..."
                    if role == "TOOL":
                        content = f"[Tool result for {msg.get('name', '?')}]: {content}"
                
                print(f"  [{i}] {role}: {content}")
            print(f"  Total messages: {len(conversation_history)}")
            print("--- End History ---")
            continue
        
        # THINK + ACT (now with tools!)
        reply = chat(user_input)
        
        print(f"\nAgent: {reply}")


if __name__ == "__main__":
    main()
