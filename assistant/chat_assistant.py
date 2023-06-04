import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import OpenAI, ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(openai_api_key=openai_api_key, temperature=0.9)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are an assistant of a 9 year old girl named Gaia.  You will answer all queries, and explain the answers to her in terms that a 9 year old could clearly understand.  You will also engage in any conversations with her that she may wish, keeping in mind that she is a 9 year old girl, and should not be exposed to inappropriate content or themes."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

while True:
    query = input("Query: ")
    print(conversation.predict(input=query))
