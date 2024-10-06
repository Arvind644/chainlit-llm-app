__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from tempfile import NamedTemporaryFile
from typing import List

import chainlit as cl
from chainlit.types import AskFileResponse

import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
# from langchain_core.embeddings import Embeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.vectorstores.base import VectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

chromadb.api.client.SharedSystemClient.clear_system_cache()

def process_file(*, file: AskFileResponse) -> List[Document]:
    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")
    
    with NamedTemporaryFile() as tempfile:
        # tempfile.write(file.content)
        with open(file.path, "rb") as f:
            tempfile.write(f.read())

        loader = PDFPlumberLoader(tempfile.name)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        
        if not docs:
            raise ValueError("PDF file parsing failed")
        
        return docs


def create_search_engine(*, docs: List[Document], embeddings: Embeddings) -> VectorStore:

    client = chromadb.EphemeralClient()
    client_settings = Settings(allow_reset=True, anonymized_telemetry=False)

    # Reset the search engine to ensure we don't use old copies.
    # NOTE: we do not need this for production
    search_engine = Chroma(client=client, client_settings=client_settings)
    search_engine._client.reset()

    search_engine = Chroma.from_documents(
        client=client,
        documents=docs,
        # embeddings=embeddings,
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    return search_engine

@cl.on_chat_start
async def on_chat_start():

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please Upload the PDF file you want to chat with...",
            accept=["application/pdf"],
            max_size_mb=10,
        ).send()
    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    docs = process_file(file=file)
    cl.user_session.set("docs", docs)
    msg.content = f"`{file.name}` processed. Loading..."
    await msg.update()

    # Indexing documents into our search engine
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )

    try:
        search_engine = await cl.make_async(create_search_engine)(
            docs=docs,
            embeddings=embeddings
        )
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise SystemError
    msg.content = f"`{file.name}` loaded. You can now ask questions!"
    await msg.update()

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=search_engine.as_retriever(max_tokens_limit=4097),
    )

    # prompt = ChatPromptTemplate.from_messages(
    #      [
    #         (
    #             "system",
    #             "You are Chainlit GPT, a helpful assistant.",
    #         ),
    #         (
    #             "human",
    #             "{question}"
    #         ),
    #     ]
    # )

    # chain = LLMChain(
    #     llm=model,
    #     prompt=prompt,
    #     output_parser=StrOutputParser()
    # )

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):

    chain = cl.user_session.get("chain")

    response = await chain.arun(
        question=message.content,
        callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)]
    )

    answer = response["anwer"]
    sources = response["sources"].strip()

    # Get all of the documents from user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    # Adding sources to the answer
    source_elements = []
    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()