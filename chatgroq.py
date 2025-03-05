from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    groq_api_key='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',
    model="llama-3.3-70b-versatile"
)
response=llm.invoke("who is the prime minister of india in 2024")
# print(response.content)

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.accenture.com/in-en/careers/jobdetails?id=AIOC-S01568654_en&title=Trust%20%26%20Safety%20New%20Associate")
page_data=loader.load().pop().page_content
# print(page_data)


from langchain_core.prompts import PromptTemplate
prompt_extract =PromptTemplate.from_template(
    """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
)

chain_extract=prompt_extract | llm
res=chain_extract.invoke(input={'page_data': page_data})
# print(type(res.content))


from langchain_core.output_parsers import JsonOutputParser
json_parser=JsonOutputParser()
json_res= json_parser.parse(res.content)
# print(json_res)
# print(type(json_res))

import pandas as pd

df = pd.read_csv("my_portfolio.csv")
# print(df)

import uuid
import chromadb

client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(documents=row["Techstack"],
                       metadatas={"links": row["Links"]},
                       ids=[str(uuid.uuid4())])
        
job = json_res
links = collection.query(query_texts=job['skills'], n_results=2).get('metadatas', [])
# print(links)
# print(job['skills'])

prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Bhawna, a business development executive at XYZ. XYZ is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
        Remember you are Bhawna, BDE at XYZ. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )

chain_email = prompt_email | llm
res = chain_email.invoke({"job_description": str(job), "link_list": links})
print(res.content)