# rAIn Cat

## Technologies Used
* Langchain
* gpt-4-turbo (from https://oai.azure.com/portal)
* text-embedding-ada-002 (from https://oai.azure.com/portal)
* FAISS
* Streamlit

## Execution Flow

### Get skills information
Provides up-to-date context to GPT using the KayAiRetriever and TavilySearchAPIRetriever

### Generate summary
Pass the previous response to GPT to generate a summary

### Fetch information from the database
Use the sqlalchemy library to connect to a locally-hosted database which contains the internal data. Get the most popular/most taken courses within the same industry as the company. This is achieved using the FAISS vectordb, AzureOpenAIEmbeddings and the create_sql_agent tool from langchain.

### Fetch the relevant courses
Ask the GPT to return the courses most relevant to the company's required skills

### UI
Display all the information on web UI using streamlit
