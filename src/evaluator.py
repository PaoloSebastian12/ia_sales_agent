
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-2.5-flash"))


def evaluate_rag(query, response, context):
    """
    Evaluates a single response using RAGAS and Gemini.
    """

    data = {
        "question": [query],
        "answer": [str(response)],
        "contexts": [[n.node.get_content() for n in context]]
    }
    dataset = Dataset.from_dict(data)
    metrics = [
        faithfulness,
        answer_relevancy
    ]

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=embeddings
    )
    print("\n" + "="*50)
    print("🧠 RAG DEBUG START")
    print("="*50)

    # 1. Query
    print("\n❓ QUERY:")
    print(query)

    # 2. Answer
    print("\n🤖 ANSWER:")
    print(response)

    # 3. Número de chunks
    print("\n📦 NUM CONTEXT CHUNKS:", len(context))

    # 4. Mostrar chunks
    print("\n📄 CONTEXTS:")
    for i, n in enumerate(context):
        try:
            text = n.node.get_content()
        except:
            text = str(n)

        print(f"\n--- Chunk {i} ---")
        print(text[:500])
    print("\n🔍 CONTEXT vs ANSWER OVERLAP:")
    answer_text = str(response).lower()
    for i, n in enumerate(context):
        try:
            content = n.node.get_content().lower()
        except:
            content = str(n).lower()

        overlap = any(word in content for word in answer_text.split()[:20])
        print(f"Chunk {i} overlap:", overlap)

    if len(context) == 0:
        print("\n⚠️ PROBLEM: No context retrieved → faithfulness FAIL seguro")

    elif all(len(n.node.get_content().strip()) == 0 for n in context):
        print("\n⚠️ PROBLEM: Context vacío")

    print("\n🧠 RAG DEBUG END")
    print("="*50)
    if len(context) == 0:
        print("⚠️ Posible causa: herramienta sin RAG (ej: email agent)")
    print("\n📊 RAGAS RESULT:")
    print(result)
    return result.to_pandas()
