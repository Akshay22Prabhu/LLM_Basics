##### FEW SHOT EXAMPLE PROMPT TEMPLATE MODELS

examples = [
    {
        "job_title" : "Data Scientist",
        "job role" : "work on machine learning and Gen AI model building and deployments"
    },
    {
        "job_title" : "Software Engineer",
        "job role" : "design, develop, test and maintain software applications"
    },
    {
        "job_title" : "Product Manager",
        "job role" : "define product versions, gather customer requirements and manage  project lifecycle"
    },
    {
        "job_title" : "Graphic Designer",
        "job role" : "create visual concepts using software to communicate ideas that inspire or inform"
    },
    {
        "job_title" : "Accountant",
        "job role" : "prepare and examine financial records, ensure accuracy and compliance"
    }
]

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
                        examples, # our examples
                        OpenAIEmbeddings(), # embedding model
                        FAISS, # vector store
                        k=3 #number of similar examples to retrieve
)

from langchain.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate(
                    input_variables=["job_title", "job_role"],
                    template = "for this Job Title : {job_title} the respective Job Role is : {job_role}"
)

few_shot_prompt = FewShotPromptTemplate(
                    example_selector=example_selector,
                    example_prompt=example_prompt,
                    prefix="You are an expert career advisor. Given a job title, provide a concise description of the job role",
                    suffix="Job Title: {input} \n Job Role:",
                    input_variables=["input"]
)

from langchain.llms import OpenAI

llm = OpenAI(model_name="gpt-4")
job_title = "Nurse"

final_prompt = few_shot_prompt.format(input=job_title)
response = llm(final_prompt)

print(response)