# %%
# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import KayAiRetriever, TavilySearchAPIRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import URL
from configparser import ConfigParser
import streamlit as st
import os

def get_skills():
    '''
    Get information about company's skill areas based on retrieved information
    '''
    template = '''
    Answer the question based only on the following context:

    These are the company's financial reports, look for information regarding the company's focused area of development and enhancement:
    {context_kay}

    These are the trending skills in the market:
    {context_tavily_trends}

    And these are the job postings made by the company. Refer the roles and the skills mentioned here:
    {context_tavily_jobs}

    Include specific skillsets or technologies for each topic.
    Cite 3 sources of searches, sec and job postings by sharing the links in a seperate references section in the response.
    '''

    messages = [
        SystemMessage(content='You are a marketing agent of Skillsoft that identifies skill gaps in your customer companies. Identify top 7 skill gaps of the customer in the relevant GTM category.'),
        SystemMessagePromptTemplate.from_template(template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    output_parser = StrOutputParser()

    retriever_kay = KayAiRetriever.create(
    dataset_id="company", data_types=["10-K", "10-Q"], num_contexts=6
)

    retriever_tavily = TavilySearchAPIRetriever(k=10)
    chain = (
        RunnablePassthrough.assign(context_tavily_trends = (lambda x : f'What are the trending skills in {x['product_type']} ? ') | retriever_tavily, 
                                    context_tavily_jobs = (lambda x : f'What roles and skills is {x['company']} looking for in {x['product_type']} based on their job postings?') | retriever_tavily,
                                    context_kay = (lambda x : f'What skills does {x['company']} focus or wants to focus on for workforce and company development?') | retriever_kay)
        | prompt
        | chat
        | output_parser
    )

    return chain

def get_summary():
    '''
    Returns summary of the returned response
    '''

    hist_template = '''
    Following is the history of the previous conversation suggesting the skills that a company should invest in for workforce development.
    {skills_result}

    Answer the query only based on the above context. Keep only a paragraph length.
    '''

    hist_messages = [
        SystemMessage(content='You are an agent that summarizes based on a given context such that the Skillsoft marketing person can use it convice their customers'),
        SystemMessagePromptTemplate.from_template(hist_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    hist_prompt = ChatPromptTemplate.from_messages(hist_messages)

    output_parser = StrOutputParser()
    
    hist_chain = (       
        RunnablePassthrough.assign(skills_result =(lambda x: skills_result))
        | hist_prompt
        | chat
        | output_parser
    )

    return hist_chain
    

def get_info_from_db():
    '''
    Return agent executor for sql server database
    '''
    server = config.get('SQL Server', 'Server')
    database = config.get('SQL Server', 'Database')
    driver = config.get('SQL Server', 'Driver')
    conn_string = f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes"

    conn_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_string})

    db = SQLDatabase.from_uri(conn_url)

    vector_db = FAISS.load_local("companies_faiss_index", embeddings, allow_dangerous_deserialization= True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    description = """Use to look up values to filter on. Input is an approximate spelling of the company name, output is a
    valid company name. Use the name most similar to the search."""
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_company_names",
        description=description,
    )

    agent_executor = create_sql_agent(chat, db=db,extra_tools=[retriever_tool], agent_type="openai-tools", verbose=True)

    return agent_executor


def get_courses():
    '''
    Combine skills and database information to get relevant courses
    '''

    hist_template = '''
    Answer the question based only on the following context:

    These are the skills the company should invest in for workforce development:
    {skills_result}

    These are some courses Skillsoft offers that the company does not take but it's peers within the same industry do:
    {result_courses}

    Given the list of courses and their ids, give 10 courses covering different areas relevant to their wanted skills. Give the output in the form a table containing the course titles and ids. Say if you can not answer and do not give hypothetical information.
    '''

    hist_messages = [
        SystemMessage(content='You are a marketing agent of Skillsoft that recommends courses to customers based on their skill wants. '),
        SystemMessagePromptTemplate.from_template(hist_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    hist_prompt = ChatPromptTemplate.from_messages(hist_messages)

    output_parser = StrOutputParser()
    
    hist_chain = (       
        RunnablePassthrough.assign(skills_result =(lambda x: skills_result)).assign(result_courses = (lambda x : result_courses['output']))               
        | hist_prompt
        | chat
        | output_parser
    )
    
    return hist_chain


st.set_page_config(layout = "wide")
# st.image(image='rAIn Cat.png', width=100)
col1, col2 = st.columns(2, gap="large")

config = ConfigParser()
config.read('config.ini')

chat = AzureChatOpenAI(
    azure_deployment='gpt-4-turbo',
    azure_endpoint = "https://oai-playground-dev-03.openai.azure.com/", 
    api_key=config.get("gpt-4-turbo", "AZURE_OPENAI_KEY"),  
    # api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-02-15-preview"
)

embeddings = AzureOpenAIEmbeddings(
azure_deployment="text-embedding-ada-002",
openai_api_version="2024-02-15-preview",
api_key=config.get("text-embedding-ada-002", "AZURE_OPENAI_KEY"),  
azure_endpoint='https://oai-playground-dev-01.openai.azure.com/'
)

max_retries = 2

with col1.form('inputs'):
    company_name = st.text_input('Company Name')
    product_type = st.selectbox('Product Type', ['Technology and Development', 'Business and Leadership'], placeholder='Choose an option')
    user_query = st.text_area('Input Query')
    submit_state = st.form_submit_button()

if submit_state:
    attempt = 0
    is_bad_response = True
    sorry_words = ['sorry', 'technical issue', 'apologize', 'unable']

    with col1.status('Generating response', expanded=True) as status:
        try:
            st.write('Fetching skills information')
            chain = get_skills()
            skills_result = chain.invoke({'product_type': product_type, 'company': company_name, 'question':user_query})

            st.write('Generating summary')
            summary_chain = get_summary()
            print(summary_chain)
            summary_result = summary_chain.invoke({'question':"Summarize the previous conversation highlighting the suggested the skills areas and topics for the company"})

            st.write('Getting information from database')
            agent_executor = get_info_from_db()

            while is_bad_response and attempt <= max_retries:
                try:
                    result_courses = agent_executor.invoke(
                        {
                            "input": f"What courses has {company_name} not taken in the {product_type} gtm category compared to its peers in the same industry? Return 30 most taken courses. Give its title and course_id."
                        }
                    )
                except Exception as e:
                    print(f'Error occured while fetching from database : {str(e)}')
                    is_bad_response = True

                    if attempt > max_retries:
                        raise
                else:
                    is_bad_response = any(word in result_courses['output'] for word in sorry_words)
                attempt+=1

                
            st.write('Fetching relevant courses')
            hist_chain = get_courses()
            result_popular_courses = hist_chain.invoke({'question' : 'What 10 courses are most relevant to the skills that the company wants ?'})

            st.balloons()
            status.update(label='Response generation complete!!!', state='complete', expanded=False)
        except Exception as e:
            print(f'Error occured while generating response: {str(e)}')
            status.update(label='Response generation failed', state='error', expanded=False)

    tab1, tab2, tab3 = col2.tabs(["Skills", "Summary", "Courses"])

    tab1.subheader('7 skills suggested by GPT based on trending skills')
    tab1.write(skills_result)
    tab2.subheader('Summary')
    tab2.write(summary_result)
    tab3.subheader(f"10 courses that the company does not take but it's peers do")
    tab3.write(result_popular_courses)
