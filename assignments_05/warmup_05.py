from dotenv import load_dotenv
from openai import OpenAI

# Q1
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("AI Response:")
print(response.choices[0].message.content)
print("Model Used:", response.model)
print("Total Tokens Used:", response.usage.total_tokens)

# Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ], 
        temperature=temperature
    )

print(f"\nTemperature: {temperature}")
print("AI Response:", response.choices[0].message.content)

# Reflection:
# At temperature 0, the response is generally more consistent
# and predictable. At 0.7, the response may be more creative
# while still making sense. At 1.5, the response can be more
# varied and unpredictable.
#
# I would use temperature 0 if I needed a consistent,
# reproducible output.

# Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices, start=1):
    print(f"\nCompletion {i}:")
    print(choice.message.content)


# Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print("\nAPI Question 4:")
print(response.choices[0].message.content)

# Reflection:
# The response was cut short because max_tokens was set to 15.
# This limits how many tokens the model can generate.
# In a real application, max_tokens can help control costs,
# prevent overly long responses, and manage response length.


# System Q1
messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Question 1 - Patient Python Tutor:")
print(response.choices[0].message.content)


# Second personality: Strict computer science professor
messages = [
    {
        "role": "system",
        "content": "You are a strict computer science professor. Explain concepts using technical language and focus on precise definitions."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Question 1 - Strict Professor:")
print(response.choices[0].message.content)


# Reflection:
# The first personality explains list comprehensions in a simple,
# friendly, and encouraging way. The second uses more technical
# language and precise definitions. The system message changes
# the model's tone and style, even though the user question is the same.


# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Question 2:")
print(response.choices[0].message.content)


# Reflection:
# The model knows Jordan's name because we included the previous
# conversation messages in the API request. Even though the API
# is stateless, the conversation history provides the context
# needed to answer the question.


# Prompt Engineering Q1
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of this review as exactly "
                    "'positive', 'negative', or 'mixed'. "
                    "Return only the sentiment label.\n\n"
                    f"Review: {review}"
                )
            }
        ]
    )

    print(f"Review {i}: {response.choices[0].message.content}")


# Prompt Engineering Q2
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of each review as exactly "
                    "'positive', 'negative', or 'mixed'.\n\n"
                    "Example:\n"
                    'Review: "Fast shipping but the item arrived damaged."\n'
                    "Sentiment: mixed\n\n"
                    f'Review: "{review}"\n'
                    "Sentiment:"
                )
            }
        ]
    )

    print(f"Review {i}: {response.choices[0].message.content.strip()}")


# Reflection:
# Adding one example helped show the model the expected output format.
# The results may be more consistent because the model can follow
# the example, compared to Q1, where no example was provided.



# Prompt Question 3 

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the sentiment of each review as exactly "
                    "'positive', 'negative', or 'mixed'.\n\n"

                    "Example 1:\n"
                    'Review: "The staff was friendly and helpful."\n'
                    "Sentiment: positive\n\n"

                    "Example 2:\n"
                    'Review: "The product broke and customer service was terrible."\n'
                    "Sentiment: negative\n\n"

                    "Example 3:\n"
                    'Review: "The food was delicious, but the service was slow."\n'
                    "Sentiment: mixed\n\n"

                    f'Review: "{review}"\n'
                    "Sentiment:"
                )
            }
        ]
    )

    print(f"Review {i}: {response.choices[0].message.content.strip()}")


# Reflection:
# Zero-shot uses instructions without examples. I would choose it
# for simple tasks that are easy to understand.
#
# One-shot provides one example. I would choose it when I want
# to demonstrate a specific response format.
#
# Few-shot provides several examples. I would choose it when a task
# is more complex or needs consistent classifications.


# Prompt Q4
prompt = """
A data engineer earns $85,000 per year. She gets a 12% raise,
then 6 months later takes a new job that pays $7,500 more per year
than her post-raise salary.

Solve this problem by explaining the calculation in clear,
numbered steps. Then label the final answer clearly.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nPrompt Question 4 - Chain of Thought:")
print(response.choices[0].message.content)


# Reflection:
# Asking the model to break a problem into steps helps it organize
# the calculations and handle each part separately. This can reduce
# mistakes in multi-step problems, although it does not guarantee
# that the answer will always be correct.



# Prompt Q5
import json

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Analyze the following review and return ONLY valid JSON. "
                "Do not include Markdown or explanations outside the JSON.\n"
                "Use exactly these keys:\n"
                "- sentiment: positive, negative, or mixed\n"
                "- confidence: a float between 0 and 1\n"
                "- reason: one sentence explaining the sentiment\n\n"
                f"Review: {review}"
            )
        }
    ],
    response_format={"type": "json_object"}
)

raw_response = response.choices[0].message.content

print("\nPrompt Question 5 - Raw JSON Response:")
print(raw_response)

try:
    result = json.loads(raw_response)

    print("\nParsed JSON Fields:")
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except (json.JSONDecodeError, KeyError, TypeError) as error:
    print("\nCould not parse the response correctly.")
    print("Error:", error)
    print("Raw response for debugging:", raw_response)



# Prompt Q6
# First test: Text containing instructions
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your "
    "sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Question 6 - Instructions:")
print(response.choices[0].message.content)


# Second test: Regular text without instructions
user_text = (
    "Technology has changed the way people communicate. "
    "Many people use smartphones and computers to stay connected."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Question 6 - Regular Prose:")
print(response.choices[0].message.content)


# Reflection:
# Delimiters help prevent confusion between the user's text
# and the instructions given to the AI. They clearly separate
# the content to analyze from the task the model must perform.


# Ollama Question 1

# Ollama Question 1

# Prompt used for both models
prompt = "Explain what a large language model is in two sentences."

# OpenAI API Response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nOpenAI API Response:")
print(response.choices[0].message.content)


# Ollama Response:
# A large language model is an AI system trained on vast amounts of
# text to understand and generate human-like conversations, enabling
# tasks like writing, customer service, and problem-solving. It uses
# deep learning algorithms to process and interpret large datasets,
# allowing it to adapt to various contexts and generate accurate
# responses.


# Reflection:
# Both models explain what a large language model is, but their
# wording and level of detail may be different.
#
# Advantage of running a model locally:
# It offers more privacy because prompts can stay on my computer.
# It can also work without internet after downloading the model.
#
# Disadvantage of running a model locally:
# It uses my computer's memory and processing power, and a small
# local model may be less capable than a larger cloud-based model.