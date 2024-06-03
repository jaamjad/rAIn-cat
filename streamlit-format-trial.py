# %%
# Import Azure OpenAI
import streamlit as st
import time
import os

st.set_page_config(layout = "wide")

col1, col2 = st.columns(2,gap="large")

with col1.form('inputs'):
    company_name = st.text_input('Company Name')
    product_type = st.selectbox('Product Type', ['Technology and Development', 'Business and Leadership'], placeholder='Choose an option')
    user_query = st.text_area('Input Query')
    submit_state = st.form_submit_button()

if submit_state:
    attempt = 0
    is_bad_response = True
    sorry_words = ['sorry', 'technical issue', 'apologise', 'unable']

    with col1.status('Generating response', expanded=True) as status:
        try:
            st.write('Fetching skills information')
            time.sleep(2)
            skills_result = "it's me"
            st.write('Getting information from database')
            time.sleep(2)
            st.write('Fetching relevant courses')
            result_popular_courses = "me again"
            time.sleep(5)
            status.update(label='Response generation complete!!!', state='complete', expanded=False)
        except Exception as e:
            print(f'Error occured while generating response: {str(e)}')
            status.update(label='Response generation failed', state='error', expanded=False)

    st.balloons()
    tab1, tab2, tab3 = col2.tabs(["Skills", "Summary", "Courses"])

    tab1.write(skills_result)
    tab3.write(result_popular_courses)
