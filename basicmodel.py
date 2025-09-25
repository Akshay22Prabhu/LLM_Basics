import os

##### FOR OPENAI MODELS

from langchain.llms import OpenAI
from secret_key import openai_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = OpenAI(temparature=0.6)

output=llm.predict("What is the capital of India")
print(output)



##### FOR HUGGINGFACEHUB MODELS

from langchain import HuggingFaceHub
from secret_key import hf_token

os.environ['HUGGINGFACEHUB_API_KEY'] = hf_token

llm_hfhub = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temparature":0.6, "max_length":64})
output=llm_hfhub.predict("Can you tell the capital of Russia")


##### FOR HUGGINGFACEPIPELINE MODELS

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
# from langchain.llms import HuggingFacePipeline


model_2 = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
llm_hfp=HuggingFacePipeline(pipeline=model_2)

# predict is depricated, then use invoke
# output=llm_hfp.predict("Can you tell the capital of Russia")
output=llm_hfp.invoke("Can you tell the capital of Russia")
print(output)


##### FOR EMBEDDINGS MODELS

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(temperature=0.6)
text = "Please convert this text to its embeddings"

text_embeddings = embeddings.embed_query(text)
print(f"your embeddings length is {len(text_embeddings)}")
print(f"here is a sample : {text_embeddings[:5]} first 5 embeddings")



##### FOR PROMPT TEMPLATE & SIMPLE LLMCHAIN MODELS

# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

# from langchain.chains import LLMChain, SequentialChain
from langchain.chains.llm import LLMChain

prompt_temp = PromptTemplate(
    input_variables=['country'],
    template="Tell me the capital of {country}"
)

chain=LLMChain(llm=llm_hfp, prompt=prompt_temp)
chain.run("India")




##### FOR SIMPLE SEQUENTIAL CHAINS

from langchain.chains.sequential import SimpleSequentialChain, SequentialChain

capital_prompt = PromptTemplate(
    input_variables=["country"],
    template="Please tell me the capital of {country}"
)

capital_chain = LLMChain(llm=llm_hfp, prompt=capital_prompt)

famous_prompt = PromptTemplate(
    input_variables=["capital"],
    template="Please recommend amazing places to visit in {capital}"
)

famous_chain = LLMChain(llm=llm_hfp, prompt=famous_prompt)

# this model gives final output only
final_chain = SimpleSequentialChain(
    chains=[capital_chain, famous_chain]
)
final_chain.run("India")
#final_chain.run({"country":"India"})



# TO GET ALL OUTPUT IN THE CHAIN , USE SEQUENTIAL CHAIN

capital_prompt = PromptTemplate(
    input_variables=["country"],
    template="Please tell me the capital of {country}"
)

capital_chain = LLMChain(llm=llm_hfp, prompt=capital_prompt, output_keys="capital")

famous_prompt = PromptTemplate(
    input_variables=["capital"],
    template="Please recommend amazing places to visit in {capital}"
)

famous_chain = LLMChain(llm=llm_hfp, prompt=famous_prompt, output_keys="places")

# this model gives entire output , both stages
final_chain = SequentialChain(
    chains=[capital_chain, famous_chain],
    input_variables=['country'],
    output_variables=['capital','places']
)
final_chain({'country':"India"})
#final_chain.run({"country":"India"})

