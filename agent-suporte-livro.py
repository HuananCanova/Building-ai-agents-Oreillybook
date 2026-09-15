import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph

# Carrega a chave da API do arquivo .env
load_dotenv()

# Configura o modelo LLM (usando Groq com openai/gpt-oss-120b ou OpenAI)
# Nota: No livro foi impresso model="gpt-5" como placeholder futuro.
if os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# -- 1) Definir nossa ferramenta de negócios única
@tool
def cancel_order(order_id: str) -> str:
    """Cancelar um pedido que ainda não foi enviado."""
    # (Aqui você chamaria sua verdadeira API de backend)
    return f"O pedido {order_id} foi cancelado."


# Vincula a ferramenta ao modelo para que o LLM saiba quando chamá-la
model = llm.bind_tools([cancel_order])


# -- 2) O agente "brain": invocar LLM, executar ferramenta e invocar LLM novamente
def call_model(state):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "NÃO_IDENTIFICADO"})

    # Prompt do sistema diz ao modelo exatamente o que fazer
    prompt = (
        f"""Você é um agente de suporte de e-commerce.
        ORDER ID: {order['order_id']}
        Se o cliente pedir para cancelar, chame cancel_order(order_id)
        e então envie uma confirmação simples.
        Caso contrário, apenas responda normalmente."""
    )

    full = [SystemMessage(prompt)] + msgs

    # Primeira passagem pelo LLM: decide chamar ou não nossa ferramenta
    # Nota: No livro foi impresso 'AIMessage = ...', mas a linha seguinte usa 'first'
    first = model.invoke(full)
    out = [first]

    if getattr(first, "tool_calls", None):
        # Executa a ferramenta cancel_order
        tc = first.tool_calls[0]
        result = cancel_order.invoke(tc["args"])
        out.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # 2ª passagem pelo LLM: gera o texto final de confirmação
        # Nota: No livro foi impresso 'AIMessage = ...', mas na página seguinte usa 'out.append(second)'
        second = model.invoke(full + out)
        out.append(second)

    return {"messages": out}


# -- 3) Conecta tudo em um StateGraph
def construct_graph():
    # Nota: No livro estava StateGraph({"order": None, "messages": []}).
    # Nas versões atuais do LangGraph, passa-se dict diretamente.
    g = StateGraph(dict)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()


graph = construct_graph()

if __name__ == "__main__":
    example_order = {"order_id": "A12345"}
    convo = [HumanMessage(content="Por favor, cancele meu pedido A12345.")]
    result = graph.invoke({"order": example_order, "messages": convo})
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")
