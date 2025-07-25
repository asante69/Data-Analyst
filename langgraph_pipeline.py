import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Set up OpenAI API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Basic state for the simple workflow
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

# Extended state used in the conditional workflow
class EnhancedState(State):
    sentiment: str

# Initialize the OpenAI chat model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def classification_node(state: State) -> dict:
    """Classify text into News, Blog, Research, or Other."""
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Classify the following text into one of the categories: "
            "News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:"
        ),
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}


def entity_extraction_node(state: State) -> dict:
    """Extract entities from text."""
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Extract all the entities (Person, Organization, Location) from the "
            "following text. Provide the result as a comma-separated list."\
            "\n\nText:{text}\n\nEntities:"
        ),
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = [e.strip() for e in llm.invoke([message]).content.strip().split(",") if e]
    return {"entities": entities}


def summarization_node(state: State) -> dict:
    """Summarize text in one short sentence."""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}


def sentiment_node(state: EnhancedState) -> dict:
    """Determine whether text sentiment is Positive, Negative, or Neutral."""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Analyze the sentiment of the following text. Is it Positive, Negative, or Neutral?\n\nText:{text}\n\nSentiment:",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    sentiment = llm.invoke([message]).content.strip()
    return {"sentiment": sentiment}


def route_after_classification(state: EnhancedState) -> bool:
    """Return True if entity extraction should run."""
    category = state["classification"].lower()
    return category in ["news", "research"]


def build_conditional_app() -> StateGraph:
    """Construct the conditional workflow."""
    workflow = StateGraph(EnhancedState)
    workflow.add_node("classification_node", classification_node)
    workflow.add_node("entity_extraction", entity_extraction_node)
    workflow.add_node("summarization", summarization_node)
    workflow.add_node("sentiment_analysis", sentiment_node)

    workflow.set_entry_point("classification_node")
    workflow.add_conditional_edges(
        "classification_node",
        route_after_classification,
        path_map={True: "entity_extraction", False: "summarization"},
    )
    workflow.add_edge("entity_extraction", "summarization")
    workflow.add_edge("summarization", "sentiment_analysis")
    workflow.add_edge("sentiment_analysis", END)
    return workflow.compile()


if __name__ == "__main__":
    app = build_conditional_app()

    sample_text = """
    OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems. Additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
    """

    result = app.invoke({"text": sample_text})
    print("Classification:", result["classification"])
    print("Entities:", result.get("entities", "Skipped"))
    print("Summary:", result["summary"])
    print("Sentiment:", result["sentiment"])

    blog_text = """
    Here's what I learned from a week of meditating in silence. No phones, no talking—just me, my breath, and some deep realizations.
    """

    result = app.invoke({"text": blog_text})
    print("\nBlog example")
    print("Classification:", result["classification"])
    print("Entities:", result.get("entities", "Skipped"))
    print("Summary:", result["summary"])
    print("Sentiment:", result["sentiment"])
