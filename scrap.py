# import necessary packages
import os

from apify_client import ApifyClient
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_core.documents import Document
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# load environment variables from a .env file
_ = load_dotenv(find_dotenv())

apify_api_token = os.getenv('APIFY_API_TOKEN')
google_api_key = os.getenv('GOOGLE_API_KEY')
target_website = os.getenv('WEBSITE_URL')

if __name__ == '__main__':
    # initialize an apify client
    apify_client = ApifyClient(apify_api_token)
    website_url = target_website

    # start an actor and wait for it to finish
    print(f'Extracting data from "{website_url}". Please wait...')

    # run the website content crawler actor to scrape the target website
    actor_run_info = apify_client.actor('apify/website-content-crawler').call(
        run_input={'startUrls': [{'url': website_url}]}
    )
    print('Saving data into the vector database. Please wait...')

    # load the scrapped documents
    loader = ApifyDatasetLoader(
        dataset_id=actor_run_info['defaultDatasetId'],
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["text"],
            metadata={"source": dataset_item["url"]}
        ),
    )
    documents = loader.load()

    # chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # initialize google palm emdebbing
    embedding = GooglePalmEmbeddings(google_api_key=google_api_key)

    # convert the documents into vectors using the embeddings
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory='db2',
    )
    vectordb.persist()
    print('All done!')
