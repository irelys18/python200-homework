from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import os


# Step 1
load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "OpenAI API key not found. Check your .env file."

print("API key loaded successfully.")

docs_dir = Path("../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs")

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print("Groundwork documents directory found.")



# Step 2
documents = SimpleDirectoryReader(
    input_dir=str(docs_dir)
).load_data()

print("\nNumber of documents loaded:", len(documents))

print("\nLoaded documents:")
for document in documents:
    print("-", document.metadata["file_name"])



# Step 3
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)
Settings.llm = OpenAI(
    model="gpt-4o-mini"
)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")



# Step 4
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n" + "=" * 60)
    print("Question:", question)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    top_node = response.source_nodes[0]

    print("\nTop retrieved source:")
    print("Document:", top_node.node.metadata["file_name"])

    print("Similarity score:", round(top_node.score, 4))

    print("Chunk preview:", top_node.node.get_content()[:200])


# Reflection:
# The assistant sounded confident and provided accurate answers to all five
# questions. I was surprised that some answers were correct even when the
# top retrieved source was not the most relevant document. For example,
# the weekend hours question retrieved our_story.txt instead of faq.txt.
# This shows that RAG can generate accurate answers, but the retrieval
# process can still be improved to find the most relevant source chunks.



# Step 5
failure_question = (
    "Does Groundwork offer free coffee refills for customers "
    "who bring their own reusable cups?"
)

print("\n" + "=" * 60)
print("Step 5: Find a Failure")
print("Question:", failure_question)

failure_response = query_engine.query(failure_question)

print("\nFull Response:")
print(failure_response)

print("\nAll Retrieved Source Nodes:")

for i, source_node in enumerate(failure_response.source_nodes, start=1):
    print(f"\nSource Node {i}:")
    print("Document:", source_node.node.metadata["file_name"])
    print("Similarity score:", round(source_node.score, 4))
    print("Chunk preview:", source_node.node.get_content()[:200])

# Reflection:
# I asked whether Groundwork offers free coffee refills for customers
# who bring reusable cups. I expected this question to be difficult
# because the documents do not mention free refills or reusable-cup
# discounts.
#
# The retrieval system returned our_story.txt, menu.txt, and
# wholesale_catering.txt, but none of these sources contained the
# requested information. The assistant correctly responded that
# there was no information available instead of making up an answer.
#
# Its tone remained direct and confident, even though the information
# was missing. This shows that we should not assume an AI response
# is correct just because it sounds confident.
#
# To improve the system, I would add a prompt instruction telling
# the model to answer only using the provided documents and clearly
# state when the information is not available. I would also improve
# retrieval so it selects more relevant source chunks.


# Step 6 

# 1. LlamaIndex made it much easier to build a RAG system.
# My implementation used only a few lines of code to set the embedding
# model, create the vector index, and build the query engine.
# I did not have to manually write all the code for chunking,
# generating embeddings, storing vectors, and retrieving information.
# In the keyword-based RAG lesson, I had to write more code to process
# questions, compare keywords, and select documents.
# This shows me that frameworks save development time and make
# complicated tasks easier to manage.

# 2. Another useful application would be an employee support assistant
# for a company. Employees could ask questions about vacation policies,
# health benefits, company rules, and onboarding procedures.
# The assistant could retrieve information from company documents
# and provide answers without employees having to search through
# many files or contact HR for every question.

# 3. One failure mode that RAG cannot fully prevent is hallucination.
# Even when the retrieval system finds relevant documents, the language
# model might misunderstand the information or generate details that
# are not supported by the sources. This means users should still
# verify important answers against the original documents.