import asyncio, json, re, ast
from typing import Sequence, TypedDict, Annotated, NotRequired

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


# ---------- State ----------
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    churn_payload: NotRequired[dict]
    churn_prediction: NotRequired[int]      # 1 or 0 (or None if unknown)
    churn_text: NotRequired[str]            # raw tool text (fallback)


# ---------- Helpers ----------
def extract_json_like(text: str):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S | re.I)
    candidate = m.group(1) if m else None
    if not candidate:
        m2 = re.search(r"\{.*\}", text, re.S)
        if m2:
            candidate = m2.group(0)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def is_churn_query(msg: str) -> bool:
    d = extract_json_like(msg)
    if not d:
        return False
    keys = {"gender","SeniorCitizen","tenure","MonthlyCharges","TotalCharges"}
    return bool(keys.intersection(d.keys()))

def parse_tool_prediction(text: str):
    """
    Try JSON first: {"prediction": 1/0}. Otherwise infer from text.
    Returns (pred:int|None, normalized_text:str).
    """
    norm = (text or "").strip()
    try:
        j = json.loads(norm)
        if isinstance(j, dict) and "prediction" in j:
            return int(j["prediction"]), norm
    except Exception:
        pass
    low = norm.lower()
    if "unlikely" in low:
        return 0, norm
    if "likely" in low:
        return 1, norm
    return None, norm


# ---------- Router (as condition function) ----------
def route(state: ChatState) -> str:
    human = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    text = (human[-1].content if human else "") or ""
    return "churn" if is_churn_query(text) else "chat_reason"


# ---------- Nodes ----------
def make_churn_node(session: ClientSession):
    async def churn_node(state: ChatState):
        # 1) extract payload
        human = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        text = human[-1].content if human else ""
        payload = extract_json_like(text)
        if payload is None:
            return {"messages": [AIMessage(content="Please include a JSON dict of customer features.")]}

        # 2) call MCP tool with the right arg shape
        result = await session.call_tool("PredictChurn", {"data": [payload]})
        parts = []
        for block in result.content:
            if getattr(block, "type", "") == "text":
                parts.append(block.text)
        tool_text = "\n".join(parts).strip() or "(no tool output)"

        # 3) normalize prediction
        pred, norm_text = parse_tool_prediction(tool_text)

        # 4) stash structured info in state; do not produce final wording here
        return {
            "churn_payload": payload,
            "churn_prediction": pred,
            "churn_text": norm_text,
        }
    return churn_node


def make_summarize_node(llm: ChatOllama):
    sys = SystemMessage(
        content=(
            "You summarize churn tool results.\n"
            "- Input includes: the raw tool text (which may be a sentence or JSON), "
            "the normalized prediction flag, and the original customer features.\n"
            "- Output a friendly final answer and a brief rationale (1–2 sentences). "
            "Do not invent probabilities."
        )
    )
    async def summarize(state: ChatState):
        pred = state.get("churn_prediction", None)
        raw = state.get("churn_text", "")
        payload = state.get("churn_payload", {})

        # Build a compact prompt
        human = HumanMessage(
            content=(
                f"Tool raw output:\n{raw}\n\n"
                f"Normalized prediction: {pred}\n"
                f"Customer features JSON:\n{json.dumps(payload)}\n\n"
                "Please produce:\n"
                "1) A one-line verdict (e.g., 'Yes — they are likely to churn.' or 'Unlikely to churn.').\n"
                "2) A brief rationale (1–2 sentences) grounded in the features (e.g., short tenure, month-to-month).\n"
            )
        )

        ai = await llm.ainvoke([sys, human])

        # Return final assistant message
        return {"messages": [AIMessage(content=ai.content)]}
    return summarize


def make_chat_reason_node(llm: ChatOllama):
    sys = SystemMessage(
        content=(
            "You are a helpful general chatbot. Provide a direct, correct answer.\n"
            "Also include a BRIEF rationale (1–2 sentences). "
            "OUTPUT STRICT JSON with keys: answer (string), brief_rationale (string)."
        )
    )
    async def chat_reason(state: ChatState):
        ai = await llm.ainvoke([sys] + list(state["messages"]))
        return {"messages": [ai]}
    return chat_reason


def make_chat_final_node():
    async def chat_final(state: ChatState):
        last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)][-1]
        # Try to parse the JSON from chat_reason
        try:
            data = json.loads(last_ai.content)
            answer = data.get("answer")
            rationale = data.get("brief_rationale")
            if answer:
                text = answer
                if rationale:
                    text += f"\n\nReason (brief): {rationale}"
                return {"messages": [AIMessage(content=text)]}
        except Exception:
            pass
        # Fallback: pass through
        return {"messages": [last_ai]}
    return chat_final

async def build_graph(llm: ChatOllama, session: ClientSession):
    builder = StateGraph(ChatState)

    # Churn path
    builder.add_node("churn", make_churn_node(session))
    builder.add_node("summarize", make_summarize_node(llm))

    # General chat path
    builder.add_node("chat_reason", make_chat_reason_node(llm))
    builder.add_node("chat_final", make_chat_final_node())

    # START -> router (as condition function)
    builder.add_conditional_edges(START, route, {
        "churn": "churn",
        "chat_reason": "chat_reason",
    })

    # churn -> summarize -> END
    builder.add_edge("churn", "summarize")
    builder.add_edge("summarize", END)

    # chat_reason -> chat_final -> END
    builder.add_edge("chat_reason", "chat_final")
    builder.add_edge("chat_final", END)

    return builder.compile()

# ---------- Main ----------
# async def main():
#     llm = ChatOllama(model="llama3", temperature=0.0)  # no tool-calling needed

#     server_params = StdioServerParameters(
#         command="uv",
#         args=["--directory", "/Users/aary/code/fastml/", "run", "server.py"],
#         env=None,
#     )

#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()

#             builder = StateGraph(ChatState)
#             # churn path
#             builder.add_node("churn", make_churn_node(session))
#             builder.add_node("summarize", make_summarize_node(llm))
#             # chat path
#             builder.add_node("chat_reason", make_chat_reason_node(llm))
#             builder.add_node("chat_final", make_chat_final_node())

#             # conditional from START
#             builder.add_conditional_edges(START, route, {
#                 "churn": "churn",
#                 "chat_reason": "chat_reason",
#             })

#             # wire churn -> summarize -> END
#             builder.add_edge("churn", "summarize")
#             builder.add_edge("summarize", END)

#             # wire chat path -> END
#             builder.add_edge("chat_reason", "chat_final")
#             builder.add_edge("chat_final", END)

#             graph = builder.compile()

#             # ------ Demo ------
#             # General chat
#             res = await graph.ainvoke({"messages": [HumanMessage(content="Explain cosine similarity simply.")]})
#             # print("\nChatbot:", res["messages"][-1].content)
#             for m in res['messages']:
#                 m.pretty_print()

#             # Churn + summarize
#             example_customer = {
#                 "gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":10,
#                 "PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No",
#                 "OnlineBackup":"Yes","DeviceProtection":"Yes","TechSupport":"No","StreamingTV":"No",
#                 "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
#                 "PaymentMethod":"Electronic check","MonthlyCharges":10,"TotalCharges":10
#             }
#             res2 = await graph.ainvoke({"messages": [
#                 HumanMessage(content=f"Will this person churn?\n```json\n{json.dumps(example_customer)}\n```")
#             ]})
#             # print("\nChurn:", res2["messages"][-1].content)
#             for m in res2['messages']:
#                 m.pretty_print()

# ---------- Chatbot runner (REPL) ----------
async def main():
    # 1) LLM (no tools needed)
    llm = ChatOllama(model="llama3", temperature=0.0)

    # 2) MCP server launch via uv
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "--directory", "/Users/aary/code/fastml/",  # <-- adjust if needed
            "run", "server.py",
        ],
        env=None,
    )

    # 3) Connect to MCP and build graph once; keep both alive
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            graph = await build_graph(llm, session)

            # 4) Multi-turn state
            state: ChatState = {"messages": []}

            print("Chatbot ready. Type '/quit' to exit, '/reset' to clear history.")
            while True:
                user = input("You: ").strip()
                if not user:
                    continue
                if user.lower() in {"/quit", "quit", "exit"}:
                    break
                if user.lower() in {"/reset", "reset"}:
                    state = {"messages": []}
                    print("Assistant: History cleared.")
                    continue

                # Append user message, run graph, print reply
                state["messages"] = list(state["messages"]) + [HumanMessage(content=user)]
                state = await graph.ainvoke(state)
                reply = state["messages"][-1].content
                print(f"Assistant: {reply}")
                
if __name__ == "__main__":
    asyncio.run(main())
