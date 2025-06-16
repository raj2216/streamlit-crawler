import streamlit as st
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

import os

# Set your Hugging Face API Token
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("key")
os.environ['HF_TOKEN'] = os.getenv("key")



# Streamlit UI
st.title("Web Crawling and Question-Answering")

# Text input for URL
url = st.text_input("Enter a URL to crawl", "https://www.innomatics.in/")

# Text input for Question
query = st.text_input("Enter your question about the webpage:")


# Button to execute
if st.button("Analyze and Answer"):

    if not url or not query:
        st.error("Please provide both a URL and a question.")
    else:

        # Perform crawling and QA
        async def main(url, query):
            # Crawling
            browser_config = BrowserConfig()
            run_config = CrawlerRunConfig()

            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)

                doc = Document(page_content=result.markdown.raw_markdown)
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100,
                )

                chunk = text_splitter.split_documents([doc])

                emb = HuggingFaceEmbeddings(model='avsolatorio/GIST-small-Embedding-v0')
                db = Chroma.from_documents(chunk, embedding=emb, persist_directory='chroma_db')

                docum = db.similarity_search(query, k=3)
                deepseek = HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-Prover-V2-671B',
                                                 provider='sambanova',
                                                 temperature=0.5,
                                                 max_new_tokens=10,
                                                 task='conversational')
                deep_seek = ChatHuggingFace(llm=deepseek,
                                             repo_id='deepseek-ai/DeepSeek-Prover-V2-671B',
                                             provider='sambanova',
                                             temperature=0.5,
                                             max_new_tokens=10,
                                             task='conversational')
                response = deep_seek.invoke(docum[0].page_content)
                return response.content

        with st.spinner("Crawling and retrieving answer..."):
            answer = asyncio.run(main(url, query))

        st.success("Answer:")
        st.write(answer)
