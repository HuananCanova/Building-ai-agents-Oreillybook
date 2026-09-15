import os
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

# Carrega as variáveis de ambiente (como GROQ_API_KEY ou OPENAI_API_KEY do arquivo .env)
load_dotenv()


def get_llm():
    """
    Retorna o modelo de linguagem disponível.
    Prioriza o Groq (gratuito) se GROQ_API_KEY estiver configurada.
    Caso contrário, tenta o OpenAI.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key and not groq_key.startswith("gsk_sua_chave"):
        from langchain_groq import ChatGroq
        # openai/gpt-oss-120b está ativo na sua conta Groq e tem suporte excelente a tool-calling
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    elif openai_key and not openai_key.startswith("sua-chave"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        # Padrão: Groq com gpt-oss-120b
        from langchain_groq import ChatGroq
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ==============================================================================
# -- Backend Simulado (Banco de Dados em Memória)
# ==============================================================================
# Em produção, isso seria consultado em um banco SQL (PostgreSQL, SQLite) ou API REST.
# Cada pedido possui dados reais e status para validar as regras de negócio.
ORDERS_DB = {
    "12345": {
        "cliente": "João Silva",
        "status": "processando",  # Pedido elegível para cancelamento
        "itens": ["Camiseta Dev", "Caneca"],
    },
    "67890": {
        "cliente": "Maria Souza",
        "status": "enviado",  # Já despachado: NÃO pode ser cancelado
        "itens": ["Notebook Gamer"],
    },
    "99999": {
        "cliente": "Carlos Lima",
        "status": "cancelado",  # Já cancelado previamente
        "itens": ["Mouse Sem Fio"],
    },
}


# -- 1. Definir o Schema de Estado (State)
class AgentState(TypedDict):
    order: dict
    # Annotated com operator.add faz com que novas mensagens sejam anexadas à lista existente
    messages: Annotated[list, operator.add]


# -- 2. Definir nossa ferramenta de negócios única com validação real de backend
@tool
def cancel_order(order_id: str) -> str:
    """Cancelar um pedido que ainda não foi enviado."""
    # 1. Busca o pedido no backend
    if order_id not in ORDERS_DB:
        return f"Erro: Pedido {order_id} não encontrado no sistema."

    order = ORDERS_DB[order_id]
    current_status = order["status"]

    # 2. Regra de negócio: pedidos enviados não podem ser cancelados
    if current_status == "enviado":
        return (
            f"Falha no cancelamento: O pedido {order_id} já foi despachado para entrega "
            f"e não pode ser cancelado pelo sistema."
        )

    # 3. Regra de negócio: pedido já cancelado anteriormente
    if current_status == "cancelado":
        return f"Aviso: O pedido {order_id} já se encontra cancelado no sistema."

    # 4. Atualização de estado no banco de dados (persistência do cancelamento)
    order["status"] = "cancelado"
    return f"Pedido {order_id} cancelado com sucesso no sistema. O reembolso foi iniciado."


# -- 3. Função do Nó: Executa o Modelo e o Ciclo de Ferramentas
def call_model(state: AgentState):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "NÃO IDENTIFICADO"})

    # Prompt do sistema diz ao modelo o que fazer e injeta o contexto do pedido
    prompt = (
        f"Você é um assistente de suporte de e-commerce.\n"
        f"ORDER ID: {order['order_id']}\n"
        f"Se o cliente pedir para cancelar, chame cancel_order(order_id)\n"
        f"e então devolva uma confirmação do cancelamento.\n"
        f"Caso contrário, apenas responda normalmente."
    )

    full = [SystemMessage(content=prompt)] + msgs

    # Instancia o modelo e vincula a ferramenta para permitir function/tool calling
    llm = get_llm()
    llm_with_tools = llm.bind_tools([cancel_order])

    # Primeira passagem pelo LLM: decide se chama a ferramenta ou responde diretamente
    first = llm_with_tools.invoke(full)
    out = [first]

    # Verifica se o modelo solicitou a chamada de alguma ferramenta
    if getattr(first, "tool_calls", None):
        # Executa a ferramenta cancel_order com os argumentos decididos pelo modelo
        tc = first.tool_calls[0]
        result = cancel_order.invoke(tc["args"])

        # Adiciona o resultado da ferramenta como uma ToolMessage
        out.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # Segunda passagem pelo LLM: recebe a resposta da ferramenta e gera a confirmação final
        second = llm.invoke(full + out)
        out.append(second)

    # Retorna o estado atualizado com as novas mensagens
    return {"messages": out}


# -- 4. Construir o Grafo (StateGraph)
def construct_graph():
    workflow = StateGraph(AgentState)

    # Adiciona o nó que executa a lógica do agente
    workflow.add_node("agent", call_model)

    # Define o fluxo: Início -> Agente -> Fim
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


# Compila o grafo para execução
graph = construct_graph()


# Função utilitária para chamar o agente
def run_agent(order_id: str, user_msg: str):
    # Prepara estado inicial
    state = {
        "order": {"order_id": order_id},
        "messages": [HumanMessage(content=user_msg)],
    }
    final_state = graph.invoke(state)
    return final_state["messages"][-1].content


# -- 5. Execução de Exemplo
if __name__ == "__main__":
    example_order = {"order_id": "67890"}
    convo = [HumanMessage(content="Cancele o pedido 67890")]

    print("=== Estado do Banco de Dados ANTES ===")
    print(f"Pedido 12345 status: {ORDERS_DB['12345']['status']}")

    print("\n=== Executando o Grafo com o pedido 12345 ===")
    result = graph.invoke({"order": example_order, "messages": convo})

    for msg in result["messages"]:
        print(f"\n[{msg.type.upper()}]: {msg.content}")

    print("\n=== Estado do Banco de Dados DEPOIS ===")
    print(f"Pedido 12345 status: {ORDERS_DB['12345']['status']}")
