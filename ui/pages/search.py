import streamlit as st
from utils import retrieve_doc
from annotated_text import annotated_text


def annotate_answer(answer,context):
    start_idx = context.find(answer)
    end_idx = start_idx+len(answer)
    annotated_text(context[:start_idx],(answer,"ANSWER","#8ef"),context[end_idx:])

def write():      
    st.write("QA System Demo")
    st.sidebar.header("Options")
    top_k_reader = st.sidebar.slider("Max. number of answers",min_value=1,max_value=10,value=3,step=1)
    top_k_retriever = st.sidebar.slider("Max. number of documents from retriever",min_value=1,max_value=10,value=3,step=1)
    question = st.text_input("Please provide your query:",value="What, who, where ...?")
    run_query = st.button("Run")
    debug = st.sidebar.checkbox("Show debug info")
    if run_query:
        with st.spinner("Performing neural search on documents... "):
            results,raw_json = retrieve_doc(question,top_k_reader=top_k_reader,top_k_retriever=top_k_retriever)
        st.write("## Retrieved answers:")
        for result in results:
            annotate_answer(result['answer'],result['context'])
            '**Relevance:** ', result['relevance'] , '**Source:** ' , result['source']
        if debug:
            st.subheader('REST API JSON response')
            st.write(raw_json)


