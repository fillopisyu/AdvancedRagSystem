from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        customer_service: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    customer_service: bool
    documents: List[str]