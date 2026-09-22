from langgraph.graph import StateGraph, START, END
from State import State
from Nodes import (
    starting_node,
    getInput,
    continue_from_last_state,
    generate_roadmap,
    generate_meta_prompt,
    generate_content,
    createFile,
    humanIntervention,
    stop_the_process,
    process_completed,
)


def check_stopped(state: State, next_node: str) -> str:
    """Helper router to cleanly divert to stop_the_process if an intervention halted execution."""
    if state.get("status") == "stopped" or state.get("action") == "stop":
        return "stop_the_process"
    return next_node


def route_after_roadmap(state: State) -> str:
    return check_stopped(state, "generate_meta_prompt")


def route_after_resume(state: State) -> str:
    return check_stopped(state, "generate_meta_prompt")


def route_after_meta(state: State) -> str:
    return check_stopped(state, "generate_content")


def route_after_content(state: State) -> str:
    return check_stopped(state, "createFile")


# -----------------------------------------------------------------------------
# LangGraph Assembly
# -----------------------------------------------------------------------------

graph = StateGraph(State)

# Register Nodes
graph.add_node("getInput", getInput)
graph.add_node("continue_from_last_state", continue_from_last_state)
graph.add_node("generate_roadmap", generate_roadmap)
graph.add_node("generate_meta_prompt", generate_meta_prompt)
graph.add_node("generate_content", generate_content)
graph.add_node("createFile", createFile)
graph.add_node("stop_the_process", stop_the_process)
graph.add_node("process_completed", process_completed)

# Conditional Start: choose between fresh input or resume
graph.add_conditional_edges(
    START,
    starting_node,
    {
        "getInput": "getInput",
        "continue_from_last_state": "continue_from_last_state",
    },
)

# New curriculum path
graph.add_edge("getInput", "generate_roadmap")
graph.add_conditional_edges(
    "generate_roadmap",
    route_after_roadmap,
    {
        "generate_meta_prompt": "generate_meta_prompt",
        "stop_the_process": "stop_the_process",
    },
)

# Resumption path
graph.add_conditional_edges(
    "continue_from_last_state",
    route_after_resume,
    {
        "generate_meta_prompt": "generate_meta_prompt",
        "stop_the_process": "stop_the_process",
    },
)

# Module generation pipeline
graph.add_conditional_edges(
    "generate_meta_prompt",
    route_after_meta,
    {
        "generate_content": "generate_content",
        "stop_the_process": "stop_the_process",
    },
)

graph.add_conditional_edges(
    "generate_content",
    route_after_content,
    {
        "createFile": "createFile",
        "stop_the_process": "stop_the_process",
    },
)

# Post-file human approval router
graph.add_conditional_edges(
    "createFile",
    humanIntervention,
    {
        "generate_meta_prompt": "generate_meta_prompt",
        "stop_the_process": "stop_the_process",
        "process_completed": "process_completed",
    },
)

# Terminal edges
graph.add_edge("stop_the_process", END)
graph.add_edge("process_completed", END)

# Compile application graph
app = graph.compile()