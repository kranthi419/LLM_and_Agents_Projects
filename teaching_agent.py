import streamlit as st
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from composio_phidata import Action, ComposioToolSet

import os
from agno.tools.arxiv import ArxivTools
from agno.utils.pprint import pprint_run_response
from agno.tools.serpapi import SerpApiTools


st.set_page_config(page_title="AI Teaching Agent", page_icon="🧠", layout="wide")


if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
if "composio_api_key" not in st.session_state:
    st.session_state["composio_api_key"] = ""
if "serpapi_api_key" not in st.session_state:
    st.session_state["serpapi_api_key"] = ""
if "topic" not in st.session_state:
    st.session_state["topic"] = ""


with st.sidebar:
    st.title("API Keys Configuration")
    st.session_state["openai_api_key"] = st.text_input("Enter your OpenAI API Key", type="password").strip()
    st.session_state["composio_api_key"] = st.text_input("Enter your Composio API Key", type="password").strip()
    st.session_state["serpapi_api_key"] = st.text_input("Enter your SerpApi API Key", type="password").strip()

    st.info("Note: You can also view detailed agent responses\nin your terminal after execution.")

if not st.session_state["openai_api_key"] or not st.session_state["composio_api_key"] or not st.session_state["serpapi_api_key"]:
    st.warning("Please enter your API keys to continue.")
    st.stop()


os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]

try:
    composio_toolset = ComposioToolSet(st.session_state["composio_api_key"])
    google_docs_tool = composio_toolset.get_tools(actions=[Action.GOOGLEDOCS_CREATE_DOCUMENT])[0]
    google_docs_tool_update = composio_toolset.get_tools(actions=[Action.GOOGLEDOCS_UPDATE_EXISTING_DOCUMENT])[0]
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()



