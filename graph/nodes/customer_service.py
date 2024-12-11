from typing import Any, Dict
from langchain.schema import Document
from graph.state import GraphState


def customer_service(state: GraphState) -> Dict[str, Any]:
    print("---CUSTOMER SERVICE---")
    question = state["question"]
    documents = state["documents"]

    # Scan the document again
    service_results = ""
    for doc in documents:
        if question in doc.page_content:
            service_results += doc.page_content + "\n"

    if not service_results.strip():
        return {"response": "Please call customer service"}

    service_results = Document(page_content=service_results)
    if documents is not None:
        documents.append(service_results)
    else:
        documents = [service_results]

    return {"documents": documents, "question": question}