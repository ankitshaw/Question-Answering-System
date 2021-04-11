import requests
import streamlit as st
import os
from utils import send_doc

API_ENDPOINT = os.getenv("API_ENDPOINT", "https://haystack-demo-api.deepset.ai")
MODEL_ID = "1"
DOC_REQUEST = "query"


def write():
	st.write("QA System Demo")
	file = st.file_uploader("Upload a file", type="txt")
	if file:
	    # data = pd.read_csv(file)
	    print(file)
	    res = send_doc(file)
	    st.write(res)