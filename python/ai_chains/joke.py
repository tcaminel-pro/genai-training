"""
The usual "tell me a joke" LLM call.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from python.ai_core.chain_registry import (
    Example,
    RunnableItem,
    register_runnable,
)
from python.ai_core.llm import get_llm
from python.ai_core.prompts import def_prompt


def chain(conf: dict):
    user_prompt = """Tell me a joke on {topic}"""
    joke_chain = (
        {"topic": RunnablePassthrough()}
        | def_prompt(user=user_prompt)
        | get_llm()  # get a LLM from configuration
        | StrOutputParser()
    )
    return joke_chain


# Register the chain
register_runnable(
    RunnableItem(
        tag="Agent",
        name="Joke",
        runnable=chain,
        examples=[Example(query=["French"])],
    )
)
