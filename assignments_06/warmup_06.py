from dotenv import load_dotenv
import os
import string
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# Q1

# Scenario A: RAG
# The legal team should use RAG because they have hundreds of PDFs
# that change every quarter. RAG allows the assistant to retrieve
# current information from the internal policy documents.

# Scenario B: Fine-tuning
# The startup should use fine-tuning because they have 3,000 examples
# of their desired brand voice. These examples can help the model
# learn and consistently reproduce their specific writing style.

# Scenario C: Prompt Engineering
# The analyst can use prompt engineering because she only needs
# answers about one short report. She can provide the report text
# directly in the prompt and ask the model questions about it.


# Q2

# A confidently wrong answer is more harmful because people may
# believe the information without checking if it is correct.
# An answer that says "I am not sure" encourages people to verify
# the information before making a decision.

# Example:
# In healthcare, an AI assistant could confidently recommend
# the wrong medication dosage. This could put a patient's health
# at risk.

# The tone of an AI response affects how much people trust it.
# When the model sounds professional and certain, people may
# assume the information is accurate, even when it is not.


# Q3

# Correct order of the RAG pipeline:

# 1. Extract text from source documents
# Read the documents and obtain their text.

# 2. Split text into chunks
# Divide long documents into smaller pieces of text.

# 3. Convert text chunks into embeddings
# Convert each chunk into a numerical representation of its meaning.

# 4. Receive the user's query
# Get the question the user wants the AI to answer.

# 5. Embed the user's query
# Convert the question into a numerical representation.

# 6. Retrieve the most relevant chunks
# Find the document chunks whose meanings are most similar to the query.

# 7. Inject retrieved chunks into the prompt
# Provide the relevant information to the language model as context.

# 8. Generate a response from the LLM
# The language model uses the question and retrieved information
# to generate an answer.


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Q1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# Search the documents using keyword retrieval.
results = simple_keyword_retrieval(query, documents, verbose=True)

# Print the name of the selected document.
print("\nSelected document:", results[0][0])

# Reflection:
# The function selected loyalty.txt because the word "your"
# appears in both the query and that document.
# The hours.txt document also received a score of 1 because
# it matched the word "weekends".
# Since the scores were tied, reverse sorting selected loyalty.txt.
# This shows that keyword retrieval can select an irrelevant
# document when different documents have the same score.


# Keyword Q2

query = "Do you have anything without caffeine?"

# Search the same documents using keyword retrieval.
results = simple_keyword_retrieval(query, documents, verbose=True)

# Print the selected document.
print("\nSelected document:", results[0][0])

# Reflection:
# The function should return "None found" because none of the
# documents contains the keywords "anything", "without", or "caffeine".
#
# Keyword RAG did not find a useful answer because it only looks
# for matching words, not similar meanings.
#
# Semantic retrieval using embeddings would do better because it
# can find documents related to the meaning of the question.
# However, the documents do not confirm which drinks are caffeine-free,
# so we would need more information to answer accurately.


# Keyword Q3

# Prediction:
# I predict that loyalty.txt will be selected because it contains
# information about the loyalty program and earning points.
# The word "rewards" may not match the exact words in the document,
# but "sign up" may have some keyword overlap.

query = "How do I sign up for rewards?"

# Search the same documents using keyword retrieval.
results = simple_keyword_retrieval(query, documents, verbose=True)

# Print the selected document.
print("\nSelected document:", results[0][0])

# Reflection:
# My prediction was incorrect.
# I expected loyalty.txt to be selected because a loyalty program
# is related to rewards.
# However, keyword retrieval only looks for exact matching words.
# The query words did not match the words in loyalty.txt,
# so the function returned "None found".
# Semantic retrieval using embeddings would do better because
# it can recognize similar meanings, such as rewards and loyalty.


# Semantic Q1

# 1. What is a vector embedding?
# A vector embedding is a list of numbers that represents the
# meaning of a piece of text. It helps a computer understand
# how similar or related two pieces of text are.

# 2. Two text chunks have cosine similarity scores of 0.85 and 0.30.
# The chunk with a score of 0.85 is more relevant to the query.
# A higher score means the meanings of the two texts are more similar.
# A score closer to 0 means they have less similarity.

# 3. Why can semantic search find relevant text without exact words?
# Semantic search looks at the meaning of words and sentences,
# not just exact keyword matches. For example, it can understand
# that "rewards" and "loyalty program" are related ideas,
# even though they use different words.


# Semantic Q2

# | Feature                 | Keyword RAG                    | Semantic RAG                         |
# |-------------------------|--------------------------------|--------------------------------------|
# | What is compared?       | Exact word overlap             | Meaning of text using embeddings     |
# | What is retrieved?      | Full document                  | Most relevant text chunks            |
# | Can it handle synonyms? | No                             | Yes                                  |
# | Storage format          | Plain text dictionary          | Vector embeddings in a vector store  |
# | Relevance score         | Number of overlapping keywords| Cosine similarity score              |


# LlamaIndex 

# Set the embedding model.
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Set the language model.
Settings.llm = OpenAI(model="gpt-4o-mini")

# Load the Brightleaf Solar PDF documents.
pdf_documents = SimpleDirectoryReader(
    "../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs"
).load_data()

print("Number of documents loaded:", len(pdf_documents))

# Display information about the loaded PDF documents.
for i, doc in enumerate(pdf_documents):
    print(f"\nDocument {i + 1}:")
    print(doc.text[:300])


# LlamaIndex Q1 

# Create an in-memory vector index from the PDF documents.
index = VectorStoreIndex.from_documents(pdf_documents)

# Create a query engine that retrieves the 3 most relevant chunks.
query_engine = index.as_query_engine(similarity_top_k=3)

# Questions to ask the BrightLeaf Solar documents.
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

# Ask each question and display the results.
for question in questions:
    print("\n" + "=" * 60)
    print("Question:", question)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response.response)

    print("\nRetrieved source nodes:")

    for i, source_node in enumerate(response.source_nodes):
        score = source_node.score

        if score is not None:
            print(f"\nSource {i + 1} similarity score: {score:.4f}")
        else:
            print(f"\nSource {i + 1} similarity score: Not available")

        print("Chunk text:", source_node.node.text[:150])


        # Reflection:
# The retrieved chunks were mostly relevant to each question.
# The benefits answer included detailed information, while the
# security answer correctly described BrightLeaf's security policies.
# Some unrelated chunks were also retrieved, showing that semantic
# search does not always return perfectly relevant results.


# LlamaIndex Q2 

# Use the employee benefits question from Question 1.
question = "What employee benefits does BrightLeaf offer?"

# Test with similarity_top_k=1.
query_engine_1 = index.as_query_engine(similarity_top_k=1)
response_1 = query_engine_1.query(question)

print("\n" + "=" * 60)
print("Results with similarity_top_k=1")
print("\nAnswer:")
print(response_1.response)

print("\nSource node scores:")
for source_node in response_1.source_nodes:
    print(f"Similarity score: {source_node.score:.4f}")


# Test with similarity_top_k=5.
query_engine_5 = index.as_query_engine(similarity_top_k=5)
response_5 = query_engine_5.query(question)

print("\n" + "=" * 60)
print("Results with similarity_top_k=5")
print("\nAnswer:")
print(response_5.response)

print("\nSource node scores:")
for source_node in response_5.source_nodes:
    print(f"Similarity score: {source_node.score:.4f}")


# Reflection:
# Both answers were very similar. Using 5 chunks provided more
# context and some additional details. However, more context is not
# always better because some retrieved chunks may be unrelated
# and could distract the model.


# LlamaIndex Q3 

# Ask a question that combines information from different documents.
question = "How does BrightLeaf support its employees while also protecting its business?"

# Retrieve the 5 most relevant chunks.
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(question)

print("\n" + "=" * 60)
print("LlamaIndex Question 3")
print("\nQuestion:", question)

print("\nAnswer:")
print(response.response)

print("\nRetrieved source nodes:")

for i, source_node in enumerate(response.source_nodes):
    print(f"\nSource {i + 1} similarity score: {source_node.score:.4f}")
    print("Chunk text:")
    print(source_node.node.text)


# Reflection:
# I expected the pipeline to retrieve information about employee
# benefits and company security. It successfully included both topics
# in its answer. However, some retrieved chunks were unrelated.
# I would improve the system by using better retrieval and filtering
# to select only the most relevant chunks.


# LlamaIndex Q4 

# Set up the LLM that will judge the responses.
judge_llm = OpenAI(model="gpt-4o-mini")

# Create the two evaluators.
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

# Use the same index and top_k setting as Question 1.
query_engine_q1 = index.as_query_engine(similarity_top_k=3)

# Test 1: A question covered by the BrightLeaf documents.
q = "What employee benefits does BrightLeaf offer?"

response = query_engine_q1.query(q)

print("\n" + "=" * 60)
print("Evaluation 1: Employee Benefits")
print("Question:", q)

faithfulness_result = faithfulness_evaluator.evaluate_response(
    response=response
)

relevancy_result = relevancy_evaluator.evaluate_response(
    query=q,
    response=response
)

print("Faithfulness score:", faithfulness_result.score)
print("Relevancy score:", relevancy_result.score)


# Test 2: A question about information not in the documents.
q2 = "What discounts does BrightLeaf offer employees at local restaurants?"

response2 = query_engine_q1.query(q2)

print("\n" + "=" * 60)
print("Evaluation 2: Information Not in Documents")
print("Question:", q2)

faithfulness_result2 = faithfulness_evaluator.evaluate_response(
    response=response2
)

relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2
)

print("Faithfulness score:", faithfulness_result2.score)
print("Relevancy score:", relevancy_result2.score)

# Reflection:
#
# 1. A faithfulness score of 1.0 means the answer is fully
# supported by the retrieved documents. A score of 0.0 means
# the answer is not supported by the sources.
#
# 2. Relevancy measures how well the answer addresses the question.
# Faithfulness checks whether the answer is supported by the documents.
#
# 3. The first query received 1.0 for both scores because the
# information was available in the documents. The second received
# 0.0 because restaurant discounts were not covered in the documents.
#
# 4. LLM-as-a-judge uses another language model to evaluate answers.
# It is useful for RAG because answers can be correct in different
# words, so exact-match accuracy is not enough.