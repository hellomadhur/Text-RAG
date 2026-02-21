from langchain.prompts import PromptTemplate


DEFAULT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful, concise assistant. Use the provided context to answer the question. "
        "If the answer is not contained in the context, say you don't know and do not hallucinate.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
)
