# chainlit-llm-app

This is a **Chat with PDF RAG application**, that take pdf as input and you can query your questions and the application will give answer based on context available in the pdf.

## Tech stack
- Chainlit
- OpenAI
- Langchain
- ChromaDB
- Python

## Running the application

I have used devcontainers to create the application. So you can create a GitHub Codespace and and install the dependencies by running
```bash
pip install -r requirements.txt
```

Then copy .env.sample and create a .env file. And add your openai key to your application.

Then for starting the application
```bash
chainlit run app/app.py -w
```

## Application UI

![img1](/images/chainlit1.png)
![img2](/images/chainlit2.png)
![img3](/images/chainlit3.png)



## Future steps

- Create a react frontend with chainlit backend
- Add more LLM to the application.
