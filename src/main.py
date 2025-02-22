from typing import List, Sequence

from dotenv import load_dotenv
from regex import R
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain 

from langchain.globals import set_verbose

set_verbose(True)


REFLECT = "reflect"
GENERATE = "generate"

def generate_messages(messages: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": messages})

def reflect_messages(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res =  reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generate_messages)
builder.add_node(REFLECT, reflect_messages)
builder.set_entry_point(GENERATE)
builder.add_edge(GENERATE, REFLECT)

def should_continue(messages: Sequence[BaseMessage]) -> bool:
    if messages[-1].content:
        return GENERATE
    return END

builder.add_conditional_edges(REFLECT, should_continue)


graph = builder.compile()


if __name__ == "__main__":
    print("Hello World!")
    
    # response = chain.invoke({"question": "write me a Haiku"})
    # print(response)
    
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)
    
    print(response[-2].content)