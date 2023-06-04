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

PROMPT_FILE = "open_prompt.txt"
#PROMPT_FILE = "gaia_restricted_prompt.txt"

load_dotenv()

with open(PROMPT_FILE, "r") as file:
    prompt_contents = file.read()

openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(openai_api_key=openai_api_key, temperature=0.9)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_contents),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

while True:
    query = input("Query: ")
    print(conversation.predict(input=query))
