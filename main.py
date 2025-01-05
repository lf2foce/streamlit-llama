import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os
import openai
from llama_index.core import SimpleDirectoryReader

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about MYOPIA !",
        }
    ]


# llm = OpenAI(
#     model="gpt-4o-mini",
#     # api_key="some key",  # uses OPENAI_API_KEY env var by default
# )
# resp = llm.complete("Paul Graham is ")
# st.text(resp)

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        for doc in docs:
            # Add metadata to documents (e.g., file name)
            # doc.metadata["source"] = doc.text.split("\n")[0]  # Extract title or use a specific metadata field
            filename = os.path.basename(doc.metadata.get('file_path', ''))
            page_label = f"Page {doc.metadata.get('page_label', 'N/A')}" if 'page_label' in doc.metadata else ''
            doc.metadata["source"] = f"{filename} {page_label}".strip()

        # Settings.llm = OpenAI(model="gpt-4", temperature=0.2
        # , system_prompt="""You are an expert on 
        # the Streamlit Python library and your 
        # job is to answer technical questions. 
        # Assume that all questions are related 
        # to the Streamlit Python library. Keep 
        # your answers technical and based on 
        # facts – do not hallucinate features.""")
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.2, 
                              system_prompt="""You are an expert on MYOPIA and your 
                              job is to answer questions with references. Include document references in responses. Keep 
                                your answers technical and analysis and based on 
                                facts – do not hallucinate features. Anwser in Vietnamese""")
        # Settings.llm = OpenAI(model="gpt-4o", temperature=0.5, 
        #                       system_prompt="""You are an expert on 
        #                         the international economics and your 
        #                         job is to create content for presentation. 
        #                         Assume that all questions are related 
        #                         to the international economics. Keep 
        #                         your answers technical and based on 
        #                         facts – do not hallucinate features. Anwser in Vietnamese""")

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        # Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        Settings.num_output = 512
        # Settings.context_window = 3900

        index = VectorStoreIndex.from_documents(docs)
        return index

index = load_data()

query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)


# response = query_engine.query(
#     "What was the impact of COVID? Show statements in bullet form and show"
#     " page reference after each statement."
# )
# response.print_response_stream()

# for node in response.source_nodes:
#     print("-----")
#     text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
#     print(f"Text:\t {text_fmt} ...")
#     print(f"Metadata:\t {node.node.metadata}")
#     print(f"Score:\t {node.score:.3f}")

# original
if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )



if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         response_stream = st.session_state.chat_engine.stream_chat(prompt)
#         st.write_stream(response_stream.response_gen)
#         message = {"role": "assistant", "content": response_stream.response}
#         # Add response to message history
#         st.session_state.messages.append(message)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        
        # Create a placeholder for the streaming response
        response_placeholder = st.empty()
        
        # Stream the response with source citations
        full_response = ""
        for chunk in response_stream.response_gen:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        # Update final response
        response_placeholder.markdown(full_response)
        
        # Add source information
        if hasattr(response_stream, 'source_nodes'):
            st.markdown("---")
            st.markdown("**Sources:**")
            for node in response_stream.source_nodes:
                st.markdown(f"- {node.metadata.get('source', 'Unknown')}")
        
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
