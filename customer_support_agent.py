import streamlit as st
import os
import json
from datetime import datetime, timedelta

from openai import OpenAI
from mem0 import Memory

# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())


# set up the streamlit app
st.title("Customer Support Agent 🤖 with memory 🧠")
st.caption("Chat with a customer support assistant who remembers your past interactions.")

# set the openai api key
openai_api_key = st.text_input("Enter OpenAI API key", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    class CustomerSupportAgent:
        def __init__(self):
            # Initialize Mem0 with Qdrant as the vector store
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": "localhost",
                        "port": 6333,
                    }
                },
            }
            try:
                self.memory = Memory.from_config(config)
            except Exception as e:
                st.error(f"Error initializing Mem0: {e}")
                st.stop()  # stop execution if memory initialization fails

            self.client = OpenAI()
            self.app_id = "customer-support"

        def handle_query(self, query, user_id=None):
            try:
                # search for relevant memories
                relevant_memories = self.memory.search(query=query, user_id=user_id)

                # Build context from relevant memories
                context = "Relevant past information:\n"
                if relevant_memories and "results" in relevant_memories:
                    for memory in relevant_memories["results"]:
                        if "memory" in memory:
                            context += f"- {memory['memory']}\n"

                # Generate a response using OpenAI
                full_prompt = f"{context}\nCustomer: {query}\nSupport Agent:"
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a customer Support Agent for KranthiTech.com, an online electronics store."},
                              {"role": "user", "content": full_prompt}]
                )
                answer = response.choices[0].message.content

                # Add the query and response to memory
                self.memory.add(query, user_id=user_id, metadata={"app_id": self.app_id, "role": "user"})
                self.memory.add(answer, user_id=user_id, metadata={"app_id": self.app_id, "role": "assistant"})

                return answer
            except Exception as e:
                st.error(f"Error handling query: {e}")
                return "I'm sorry, I'm not able to assist you at the moment. Please try again later."

        def get_memories(self, user_id=None):
            try:
                # retrieve all memories for the user
                memories = self.memory.get_all(user_id=user_id)
                return memories
            except Exception as e:
                st.error(f"Error getting memories: {e}")
                return None

        def generate_synthetic_data(self, user_id: st) -> dict | None:
            try:
                today = datetime.now()
                order_date = (today - timedelta(days=10)).strftime("%B %d %Y")
                expected_delivery = (today + timedelta(days=2)).strftime("%B %d %Y")

                prompt = f"""Generate a detailed customer profile and order history for a kranthiTech.com customer with ID {user_id}. Include:
                1. Customer name and basic info
                2. A recent order of a high-end electronic device (placed on {order_date}, to be delivered by {expected_delivery})
                3. Order details (product, price, order number)
                4. Customer's shipping address
                5. 2-3 previous orders from the past year
                6. 2-3 customer service interactions related to these orders
                7. Any preferences or patterns in their shopping behavior
                
                Format the output as a JSON object."""

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a data generation AI that created realistic customer profiles and order histories."},
                              {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                customer_data = json.loads(response.choices[0].message.content)

                # add generated data to memory
                for key, value in customer_data.items():
                    if isinstance(value, list):
                        for item in value:
                            self.memory.add(json.dumps(item),
                                            user_id, metadata={"app_id": self.app_id, "role": "system"})
                    else:
                        self.memory.add(f"{key}: {json.dumps(value)}",
                                        user_id, metadata={"app_id": self.app_id, "role": "system"})

                return customer_data
            except Exception as e:
                st.error(f"Error generating synthetic data: {e}")
                return None

    # Initialize the CustomerSupportAgent
    support_agent = CustomerSupportAgent()

    # sidebar for customer ID and memory view
    user_id = st.sidebar.title("Enter your Customer ID:")
    previous_customer_id = st.session_state.get("previous_customer_id", None)
    customer_id = st.sidebar.text_input("Enter your Customer ID")

    if customer_id != previous_customer_id:
        st.session_state.messages = []
        st.session_state.previous_customer_id = customer_id
        st.session_state.customer_data = None

    # add button to generate synthetic data
    if st.sidebar.button("Generate Synthetic Data"):
        if customer_id:
            with st.spinner("Generating synthetic data..."):
                st.session_state.customer_data = support_agent.generate_synthetic_data(customer_id)
                if st.session_state.customer_data:
                    st.success("Synthetic data generated successfully!")
                else:
                    st.error("Error generating synthetic data.")
        else:
            st.sidebar.error("Please enter a Customer ID to generate synthetic data.")

    if st.sidebar.button("View Customer Profile"):
        if st.session_state.customer_data:
            st.sidebar.json(st.session_state.customer_data)
        else:
            st.sidebar.error("Please generate synthetic data first.")

    if st.sidebar.button("View Memories"):
        if customer_id:
            memories = support_agent.get_memories(user_id=customer_id)
            if memories and "results" in memories:
                st.sidebar.write(f"Memories for Customer ID: {customer_id}")
                for memory in memories["results"]:
                    if "memory" in memory:
                        st.write(f"- {memory['memory']}")
            else:
                st.sidebar.error("No memories found for this Customer ID.")

        else:
            st.sidebar.error("Please enter a Customer ID to view memories.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("How can I assist you today?", key="query")

    if query and customer_id:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            response = support_agent.handle_query(query, user_id=customer_id)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    elif not customer_id:
        st.error("Please enter a Customer ID to start the conversation.")

else:
    st.warning("Please enter your OpenAI API key to get started.")









