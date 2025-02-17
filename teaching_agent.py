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


professor_agent = Agent(
    name="Professor",
    role="Research and Knowledge Specialist",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state["openai_api_key"]),
    tools=[google_docs_tool],
    instructions=["Create a comprehensive knowledge base that covers fundamental concepts, advanced topics, and  current developments of the given topic.",
                  "Explain the topic from first principles first. Include key terminology, core principles, and practical applications and make it as a detailed report that anyone",
                  " who's starting out can read and get maximum value out of it. Make sure it is formatted in a way that is easy to read and understand. DONT FORGET TO CREATE THE GOOGLE DOCUMENT.",
                  "Open a new Google Doc and write down the response of the agent neatly with great formatting and structure in it. **Include the Google Doc link in your response.**"],
    show_tool_calls=True,
    markdown=True
)

academic_advisor_agent = Agent(
    name="Academic Advisor",
    role="Learning Path Designer",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state["openai_api_key"]),
    tools=[google_docs_tool],
    instructions=["Using the knowledge base for the given topic, create a detailed learning roadmap.",
                  "Break down the topic into logical subtopics and arrange them in order of progression, a detailed report of roadmap that includes all the subtopics",
                  " in order to be an expert in this topic. Include estimated time commitments for each section.",
                  "Present the roadmap in a clear, structured format. DONT FORGET TO CREATE THE GOOGLE DOCUMENT.",
                  "Open a new Google Doc and write down the response of the agent neatly with great formatting and structure in it. **Include the Google Doc link in your response.**"],
    show_tool_calls=True,
    markdown=True
)

researcher_agent = Agent(
    name="Research Librarian",
    role="Learning Resource Specialist",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state["openai_api_key"]),
    tools=[google_docs_tool, SerpApiTools(api_key=st.session_state["serpapi_api_key"])],
    instructions=["Make a list of high-quality learning resource for the given topic.",
                  "Use the SerpApi search tool to find current and relevant learning materials.",
                  "Using SerpApi search tool, Include technical blogs, Github repositories, official documentation, vide tutorials, and courses.",
                  "Present the resources in a curated list with descriptions and quality assessments. DONT FORGET TO CREATE THE GOOGLE DOCUMENT.",
                  "Open a new Google Doc and write down the response of the agent neatly with great formatting and structure in it. **Include the Google Doc link in your response.**"],
    show_tool_calls=True,
    markdown=True
)
teaching_assistant_agent = Agent(
    name="Teaching Assistant",
    role="Exercise Creator",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state['openai_api_key']),
    tools=[google_docs_tool, SerpApiTools(api_key=st.session_state['serpapi_api_key'])],
    instructions=[
        "Create comprehensive practice materials for the given topic.",
        "Use the SerpApi search tool to find example problems and real-world applications.",
        "Include progressive exercises, quizzes, hands-on projects, and real-world application scenarios.",
        "Ensure the materials align with the roadmap progression.",
        "Provide detailed solutions and explanations for all practice materials.DONT FORGET TO CREATE THE GOOGLE DOCUMENT.",
        "Open a new Google Doc and write down the response of the agent neatly with great formatting and structure in it. **Include the Google Doc link in your response.**",
    ],
    show_tool_calls=True,
    markdown=True,
)



