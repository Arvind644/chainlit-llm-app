from tempfile import NamedTemporaryFile
from typing import List

import chainlit as cl
from chainlit.types import AskFileResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.chains import LLMChain

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

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

    model = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages(
         [
            (
                "system",
                "You are Chainlit GPT, a helpful assistant.",
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )

    chain = LLMChain(
        llm=model,
        prompt=prompt,
        output_parser=StrOutputParser()
    )

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):

    chain = cl.user_session.get("chain")

    response = await chain.arun(
        question=message.content,
        callbacks=[cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=response).send()