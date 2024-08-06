# import requied libraries
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
# from langchain_community.chains import ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory
)
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
website_url = os.environ.get('WEBSITE_URL', 'a website')

# set page configuration
st.set_page_config(page_title=f'Chat with {website_url}')
st.title('Chat with a website')


@st.cache_resource(ttl='1h')
def get_retriever():
    """Return a retriever from stired vector"""

    # initialize google palm embeddings
    embeddings = GooglePalmEmbeddings()

    # load chroma database
    vectordb = Chroma(persist_directory='db2', embedding_function=embeddings)

    # create a vector store retriever
    retriever = vectordb.as_retriever(search_type='mmr')

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(
            self,
            container: st.delta_generator.DeltaGenerator,
            initial_text: str = ''
        ):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


retriever = get_retriever()

# 
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=msgs,
    return_messages=True
)

# llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, streaming=True)
llm = ChatGroq(
    api_key=groq_api_key,
    model='llama3-groq-70b-8192-tool-use-preview'
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=False
)

if st.sidebar.button('Clear message history') or len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message(f'Ask me anything about {website_url}!')

avatars = {'human': 'user', 'ai': 'assistant'}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder='Ask me anything!'):
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.invoke(user_query, callbacks=[stream_handler])
