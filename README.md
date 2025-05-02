# LLM Practice  


Prompt engineering, RAG, LangChain, etc  

## Prompt Engineering  

Based on the paper [Principled Instructions Are All You Need for Questioning LLaMA-1_2, GPT-3.5_4](https://arxiv.org/abs/2312.16171)  

Model: Gemini-1.5-flash  

Gemini API is utilized.  

## Fine-Tuning  

Finance Area  

Data : [BC Card AI Finance Data](https://huggingface.co/datasets/BCCard/BCAI-Finance-Kor)  

Base Model: [HuggingFace SmolLm2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct)  

Equipment: RTX 4090 24 GB  

## RAG (Retrieval Augmented Generation)  

1. Ensemble Sparse Retirver and Dense Retriever

Dense Retriever (Vector Store): [LangChain FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/)   

Sparse Retriever (Traditional Search Method): [BM25](https://github.com/dorianbrown/rank_bm25)  

rank_bm25 is a component of LangChain in sparse retriving process.  

2. RAG condidering Department and Rank in a Corporation.

Use LangGraph and Decide to do RAG or Not.  

Check groundedness score, context relevancy, response relevancy to measure the reponse and RAG performance. 

Use both Gemini API and OpenAI API 
