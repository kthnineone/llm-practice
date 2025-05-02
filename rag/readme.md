# RAG with LangChain and LangGraph

## langchain_rag-gemini-upload.ipynb  

BM25와 FAISS의 L2 distance를 앙상블한 RAG의 예시  

목적: Vector의 유사도 외에도 전통적인 Sparse retriving 방법도 사용  


## RAG_by_auth-upload.ipynb  

부서별, 직급별로 열람 가능한 문서의 등급이 다른 경우를 가정.  

부서마다, 직급마다 서로 retrieve할 수 있는 vector db를 설정하고 이에 따라서 RAG를 수행한다.  

이때 LangGraph를 사용해서 검색이 필요한지 아닌지부터 결정한다.  

그리고 Hallucination이나 context relevancy, 그리고 response relevancy도 함께 측정해서 RAG 자체의 성능도 평가한다.  



