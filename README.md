# Building AI Agents - O'Reilly Hands-on

Implementações práticas e estudos baseados no livro **Building AI Systems / Building AI Agents** da **O'Reilly**, explorando a criação de agentes autônomos e sistemas inteligentes com **LangChain**, **LangGraph** e modelos de linguagem de última geração (LLMs) via **Groq** e **OpenAI**.

---

## 📌 Visão Geral do Projeto

Este repositório documenta a evolução do desenvolvimento de agentes conversacionais orientados a tarefas (Task-Oriented Agents) para atendimento de e-commerce:

1. **Agente de Suporte Inicial (`agent-suporte-livro.py`)**:
   - Implementação base direta das primeiras seções do livro.
   - Orquestração simplificada com `StateGraph(dict)`.
   - Tool calling dinâmico com a ferramenta `@tool` `cancel_order`.
   - Ciclo de 2 passos no LLM: (1) decisão de chamada da ferramenta e (2) resposta com confirmação ao usuário.

2. **Agente de Suporte Avançado - Capítulo 2 (`agent-suporte-cap2.py`)**:
   - Evolução estrutural com `AgentState(TypedDict)` e histórico de mensagens acumulativo via `Annotated[list, operator.add]`.
   - Ciclo de vida explícito do grafo conectando nós `START` -> `agent` -> `END`.
   - Simulação de backend / banco de dados relacional em memória (`ORDERS_DB`) com regras reais de negócio:
     - Validação de existência do pedido.
     - Bloqueio de cancelamento para pedidos já enviados (`status: "enviado"`).
     - Alerta para pedidos já cancelados (`status: "cancelado"`).
     - Persistência e alteração de estado no banco para pedidos em processamento.
   - Suporte transparente a múltiplos provedores (Groq com modelo rápido `openai/gpt-oss-120b` ou OpenAI `gpt-4o-mini`).

---

## 📂 Estrutura do Repositório

```text
├── .env.example             # Modelo para configuração das chaves de API
├── .gitignore               # Arquivos ignorados pelo Git (.env, .venv, pycache)
├── requirements.txt         # Dependências do projeto
├── agent-suporte-livro.py   # Versão 1: Agente inicial baseado no livro
├── agent-suporte-cap2.py    # Versão 2: Agente com schema tipado e validação de regras de negócio
└── README.md                # Documentação do projeto
```

---

## 🛠️ Tecnologias Utilizadas

- **[Python 3.10+](https://www.python.org/)**
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: Orquestração de agentes cíclicos com estados tipados e fluxos de decisão.
- **[LangChain Core](https://github.com/langchain-ai/langchain)**: Abstrações de mensagens (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`) e ferramentas (`@tool`).
- **[Groq API](https://console.groq.com/)**: Inferência de baixíssima latência com suporte nativo a Tool Calling (`openai/gpt-oss-120b`).
- **[OpenAI API](https://platform.openai.com/)**: Suporte a modelos como `gpt-4o-mini`.
- **[python-dotenv](https://github.com/theskumar/python-dotenv)**: Gerenciamento seguro de variáveis de ambiente.

---

## 🚀 Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/HuananCanova/Building-ai-agents-Oreillybook.git
cd Building-ai-agents-Oreillybook
```

### 2. Criar e ativar o ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar as variáveis de ambiente

Copie o arquivo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Abra o `.env` e insira sua chave da API:

```env
# Obtenha gratuitamente em https://console.groq.com/keys
GROQ_API_KEY=gsk_sua_chave_aqui

# (Opcional se usar Groq)
OPENAI_API_KEY=sua_chave_openai_aqui
```

---

### 5. Executando os Agentes

#### Exemplo 1: Agente Básico do Livro
```bash
python agent-suporte-livro.py
```

#### Exemplo 2: Agente Avançado com Regras de Negócio (Capítulo 2)
```bash
python agent-suporte-cap2.py
```

---

## 📖 Principais Conceitos Abordados

- **Tool Calling (Function Calling)**: Capacidade do LLM de inspecionar assinaturas e esquemas de funções Python e emitir payloads estruturados para invocação segura.
- **State Management**: Persistência de mensagens e metadados contextuais (como `order_id`) ao longo da trajetória de resolução.
- **Cyclic Graphs**: Uso de grafos direcionados para permitir que o modelo tome decisões, invoque ferramentas e processe os resultados de retorno para sintetizar a resposta final.
