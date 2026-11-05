# TechStream PolicyBot

A Retrieval-Augmented Generation (RAG) chatbot that answers customer questions
about return, shipping and support policies, grounded only in company documents.

**Stack:** Python, LangChain (LCEL), FAISS, OpenAI API, Streamlit

## Features
- Upload `.pdf` / `.txt` policy documents from the UI
- Recursive chunking (300 chars, 50 overlap) + `text-embedding-3-small` embeddings
- FAISS similarity search (adjustable top-k)
- Grounded prompt: replies "That is not in our policy." when the answer isn't in the context
- Shows the retrieved source chunks for every answer

## Run
```bash
pip install -r requirements.txt
cp .env.example .env      # then add your new OpenAI key
streamlit run app.py
```

## Try these questions
- What is the return policy?
- Can I return a laptop if I opened the box?
- Is shipping free?
- Do you ship internationally?
- Who won the world cup? (should say it's not in the policy)
