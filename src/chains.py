from calendar import c
import re
import boto3
from langchain_aws import BedrockLLM, ChatBedrock, ChatBedrockConverse
from botocore.config import Config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

aws_region = os.getenv("AWS_REGION", "us-east-1")
bedrock_config = Config(
    connect_timeout=120, read_timeout=120, retries={"max_attempts": 0}
)
session = boto3.Session(profile_name="ba-dev")
bedrock_rt = session.client(
    "bedrock-runtime", region_name=aws_region, config=bedrock_config
)
model_kwargs =  { 
    "max_tokens": 512,
    "temperature": 0.0,
}
llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    config=bedrock_config,
    region_name=aws_region,
    verbose=True,
    credentials_profile_name="ba-dev",
    client=bedrock_rt,
    model_kwargs=model_kwargs,
)
# ***test***
# messages = [
#     ("human","{question}"),
# ]

# prompt = ChatPromptTemplate.from_messages(messages, MessagesPlaceholder(variable_name="messages"),)

# chain = prompt | llm | StrOutputParser()
# ***end test***

# llm = ChatBedrockConverse(
#     client=bedrock_rt,
#     model="anthropic.claude-3-5-sonnet-20240620-v1:0",
#     temperature=0,
#     max_tokens=None,
# )

# llm = ChatOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama",
#     model="llama3.1",
#     temperature=0,
# )

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
