# Multi model RAG System
# RAG -- Retrieval Augmented Generation
/*The problem RAG solves: LLMs have a knowledge cutoff and can hallucinate facts*/

# RAG PIPELINE

QUESTION
    |
    |
    |
RETRIEVE --> search vector DB --> get top 3 relevant docs
    |
    |
    |
AUGMENT --> Inject docs into LLM prompt as cotext
    |
    |
    |
GENERATE --> LLM answers using only the provided context
    |
    |
    |
ANSWER WITH CITATIONS                