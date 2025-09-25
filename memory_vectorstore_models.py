##### MEMORY MODELS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0.4)
prompt_template_places = PromptTemplate(
    input_variables=["country"],
    template="list the beautiful places to visit in the {country}"
)
memory = ConversationBufferMemory()

chain = LLMChain(
    llm=llm,
    prompt=prompt_template_places,
    memory=memory)

places = chain.run("India")
print(places)


##### SAVE COST AND OPTIMIZE -> USE CONVERSATIONCHAIN

from langchain.chains import ConversationChain

convo = ConversationChain(llm=OpenAI(temperature=0.6))
print(convo.prompt.template)
print(convo.memory.buffer)


from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=1)
convo = ConversationChain(
    llm = OpenAI(temperature=0.6),
    memory = memory)

convo.run("Who won the first cricket world cup?")
convo.run("What is 5+5 ?")
# Model will now store and remember only this output, the first conversation output is lost/forgotten

convo.run("Who was the winning captain of that world cup?")
## Model Output would be -> Sorry I don't know. (Because buffer is 1, it stores only latest Q & A)
 


##### MEMORY HISTORY

from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_ai_message("Hi...!!")
history.add_user_message("Hello, How are you doing today?")

history.messages ## above messages are stored here

from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.6)

ai_response = chat(history.messages)
ai_response



##### VECTORSTORES DB

from langchain.vectorstores import Chroma

store_chroma = Chroma.from_documents(
                documents = pages,
                embeddings = OpenAIEmbeddings(),
                collection_name = "annual_report"
)

similarities = store_chroma.similarity_search("profit")
print(len(similarities))
similarities[0].page_content

embeddings = OpenAIEmbeddings()

embedding_list = embeddings.embed_documents([doc.page_content for doc in pages])

print(f"you have {len(embedding_list)} embeddings")


