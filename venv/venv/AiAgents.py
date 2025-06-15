from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_aws import ChatBedrock
import boto3
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph
from langgraph.checkpoint import MemorySaver

@tool
def compute_savings(monthly_cost: float) -> dict:
    """
    Tool to compute the potential savings when switching to solar energy based on the user's monthly electricity cost.
    
    Args:
        monthly_cost (float): The user's current monthly electricity cost.
    
    Returns:
        dict: A dictionary containing:
            - 'number_of_panels': The estimated number of solar panels required.
            - 'installation_cost': The estimated installation cost.
            - 'net_savings_10_years': The net savings over 10 years after installation costs.
    """
    # Assumptions for the calculation
    cost_per_kWh = 0.28  
    cost_per_watt = 1.50  
    sunlight_hours_per_day = 3.5
    panel_wattage = 350  
    system_lifetime_years = 10  

    # Monthly electricity consumption in kWh
    monthly_consumption_kWh = monthly_cost / cost_per_kWh
    
    # Required system size in kW
    daily_energy_production = monthly_consumption_kWh / 30
    system_size_kW = daily_energy_production / sunlight_hours_per_day
    
    # Number of panels and installation cost
    number_of_panels = system_size_kW * 1000 / panel_wattage
    installation_cost = system_size_kW * 1000 * cost_per_watt
    
    # Annual and net savings
    annual_savings = monthly_cost * 12
    total_savings_10_years = annual_savings * system_lifetime_years
    net_savings = total_savings_10_years - installation_cost
    
    return {
        "number_of_panels": round(number_of_panels),
        "installation_cost": round(installation_cost, 2),
        "net_savings_10_years": round(net_savings, 2)
    }

def handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.
    
    Args:
        state (dict): The current state of the AI agent.
    
    Returns:
        dict: A dictionary containing error messages.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> Runnable:
    """
    Creates a tool node with fallback error handling.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],
        exception_key="error"
    )

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State) -> dict:
        while True:
            result = self.runnable.invoke(state)
            
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                state["messages"].append(HumanMessage(content="Please provide valid input."))
            else:
                break

        return {"messages": [result]}

def get_bedrock_client(region: str = "us-east-1"):
    return boto3.client("bedrock-runtime", region_name=region)

def create_bedrock_llm(client):
    return ChatBedrock(
        model_id='anthropic.claude-3-sonnet-20240229-v1:0',
        client=client,
        model_kwargs={'temperature': 0}
    )

# Initialize components
llm = create_bedrock_llm(get_bedrock_client())
primary_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are a helpful customer support assistant for Solar Panels Belgium.
    You should get the following information from them:
    - monthly electricity cost
    If you are not able to discern this info, ask them to clarify! Do not attempt to guess.'''),
    ("placeholder", "{messages}"),
])

part_1_tools = [compute_savings]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

# Build the graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))

builder.add_edge("assistant", "tools")
builder.add_conditional_edges(
    "tools",
    lambda x: "assistant",
)
builder.set_entry_point("assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

def _print_event(event, _printed):
    """Helper function to print events"""
    message = event.get("messages", [])
    if message and str(message) not in _printed:
        print(message)
        _printed.add(str(message))

# Example conversation
tutorial_questions = [
    'hey',
    'can you calculate my energy saving',
    "my monthly cost is $100, what will i save"
]

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

_printed = set()
for question in tutorial_questions:
    events = graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config,
        stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)