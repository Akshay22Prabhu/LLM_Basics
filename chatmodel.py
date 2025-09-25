##### OPENAI CHAT MODELS

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat_llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],model='gpt-3.5-turbo', temperature=0.6)
chat_llm([
    SystemMessage(content="You are a comedian AI assistant"),
    HumanMessage(content="Please provide some comedy punchlines on AI")
])
## Model will generate an output as "AIMessage"


##### PROMPT
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

class Commaseparatedoutput(BaseOutputParser):
    def parse(self, text:str):
        return text.strip().split(",")


template = "Generate 5 words in comma separated way"
human_template="{text}"

chatprompt=ChatPromptTemplate.from_messages([
    ("system",template),
    ("human",human_template)
])

chain=chatprompt|chatllm|Commaseparatedoutput()
chain.invoke({"text":"intelligent"})


