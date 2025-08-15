import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display


async def main():

    llm = ChatOllama(
        model = "llama3.1",
        temperature = 0.0)

    server_params = StdioServerParameters(
        command="uv",
        args=[
            "--directory",
            "/Users/aary/code/fastml/",
            "run",
            "server.py",
        ],
        env=None,
    )

    # System message
    # --- System prompt with tiny few-shot to enforce tool use + verbalization ---
    sys_msg = SystemMessage(content="Use the MCP churn tool to answer churn questions.")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            llm_with_tools = llm.bind_tools(tools)

            async def assistant(state: MessagesState):
                ai = await llm_with_tools.ainvoke([sys_msg] + list(state["messages"]))
                return {"messages": [ai]}

            builder = StateGraph(MessagesState)
            builder.add_node("assistant", assistant)
            builder.add_node("tools", ToolNode(tools))
            builder.add_edge(START, "assistant")
            builder.add_conditional_edges("assistant", tools_condition)
            builder.add_edge("tools", "assistant")
            graph = builder.compile()

            try:
                display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
            except Exception:
                pass
            
            example_customer = {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 10,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 10,
                "TotalCharges": 10,
            }

            user_msg = HumanMessage(content=f"Will this person churn? {example_customer}")
            result = await graph.ainvoke({"messages": [user_msg]})
            for m in result['messages']:
                m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())