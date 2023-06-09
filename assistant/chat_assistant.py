import os
import sys
sys.path.append(os.getcwd())

from voice import text_to_speech

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

PROMPT_FILE = "assistant/open_prompt.txt"
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

megen_t = text_to_speech

while True:
    query = input("Query: ")
    
    answer = conversation.run(input=query)

    print(answer)

    megen_t.speak(answer[answer.find("\n\nSystem:") + len("System: "):])