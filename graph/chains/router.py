from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "customerservice"] = Field(
        ...,
        description="Given a user question choose to route it to customer service or a vectorstore.",
    )

llm = ChatOpenAI(temperature=0)

structured_llm_router = llm.with_structured_output(RouteQuery)
system = """You are a digital assistant of a bank and your vector database contains various information about the bank.
If the user's question is related to banking issues such as bank products, services, loan applications, account transactions, etc., you retrieve documents related to such information from the database and forward them to the user.
If the user's question is not related to banking, e.g. a request for general information or help on another subject, then direct the user to the ‘Customer Representative’ (customer service) for more detailed support."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router