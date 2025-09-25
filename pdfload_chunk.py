##### LOAD PDF

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("annual_report.pdf")
data = loader.load()

print(f"found {len(data)} comments")

#display 1st document
print(data[0])

#extract page_content of first 2 documents
print(f"following is the data of first 2 documents: {''.join(x.page_content[:150] for x in data[:2])}")



##### CHUNK DOCUMENT

from langchain.text_splitter import PythonCodeTextSplitter

text_splitter = PythonCodeTextSplitter(chunk_size = 1000, chunk_overlap = 300)

pages = loader.load_and_split(text_splitter=text_splitter)

print(f"you have {len(pages)} documents")
print(pages[0].page_content,"\n")
print(pages[1].page_content)
