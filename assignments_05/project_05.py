from dotenv import load_dotenv
from openai import OpenAI
import json

# Task 1
load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

SYSTEM_PROMPT = """
You are a friendly job application coach.

Help people who are changing careers, especially those moving
into technology or data-related jobs.

You can:
- Improve resume bullet points.
- Highlight transferable skills.
- Write cover letters.
- Answer job application questions.

Always:
- Stay focused on job applications.
- Never invent experience or skills.
- Ask questions when you need more information.
- Use simple and professional language.
- Remind users to review and edit your work before submitting it.
- Explain that industry expectations may vary and users should
  use their own judgment.
"""

# Reflection:
# I included instructions not to invent experience or skills
# because job applications must be honest and accurate.


# Task 2
def rewrite_bullets(bullets: list[str]) -> list[dict]:

    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach.
    Improve each resume bullet point using strong action verbs
    and clear, professional language.

    Do not invent facts or achievements.

    Return a JSON object with two lists:
    "original" and "improved".

    Each list must contain the same number of bullet points.
    Keep the original bullet points unchanged.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    response = response.choices[0].message.content

    results = json.loads(response)

    # Print original and improved bullets together
    for original, improved in zip(results["original"], results["improved"]):
        print("\nOriginal:", original)
        print("Improved:", improved)

    return results

# Reflection:
# The original bullets are weak because they are general
# and do not explain specific actions or results.
# The model suggested stronger action verbs and more
# professional wording to improve the descriptions.

# Test the bullet point rewriter

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

print("Starting bullet point rewriter...")

rewrite_bullets(bullets)


# Task 3
def generate_cover_letter(job_title: str, background: str) -> str:

    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_completion(messages)

    return response

# Test the cover letter generator

job_title = "Junior Data Engineer"

background = (
    "Five years of experience as a middle school math teacher; "
    "recently completed a Python course and built data pipelines using Pandas."
)

cover_letter = generate_cover_letter(job_title, background)

print("\nCover Letter Opening:")
print(cover_letter)

# Reflection:
# I chose examples from different careers to show how people
# can connect their previous experience with new technical skills.
# Few-shot prompting helps control the tone, style, and level
# of detail in the AI's response.


# Task 4
def is_safe(text: str) -> bool:

    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("This message cannot be processed. Please rephrase it.")

        return False

    return True

# Test the moderation function

safe_text = "Can you help me improve my resume?"

flagged_text = "I want to hurt someone and cause serious harm."

print("\nModeration Tests:")

print("Safe message:", is_safe(safe_text))

print("Flagged message:", is_safe(flagged_text))


# Task 5
def run_chatbot():

    # 1. Initialize conversation history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:

        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Check moderation
        if not is_safe(user_input):
            continue

        # 5. Rewrite resume bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():

            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if line:
                    raw_bullets.append(line)

            # Check the bullet points before sending them to the AI
            if raw_bullets and is_safe("\n".join(raw_bullets)):
                rewrite_bullets(raw_bullets)

        # 6. Generate a cover letter
        elif "cover letter" in user_input.lower():

            job_title = input(
                "Job Application Helper: What is the job title? "
            ).strip()

            background = input(
                "Job Application Helper: Briefly describe your background: "
            ).strip()

            # Check the additional information before sending it to the AI
            if not is_safe(job_title + "\n" + background):
                continue

            cover_letter = generate_cover_letter(job_title, background)

            print("\nJob Application Helper:")
            print(cover_letter)

        # 7. Handle regular questions
        else:

            # Add the user's message to conversation history
            messages.append(
                {"role": "user", "content": user_input}
            )

            # Get the AI response
            reply = get_completion(messages)

            print("\nJob Application Helper:", reply)

            # Save the AI response in conversation history
            messages.append(
                {"role": "assistant", "content": reply}
            )


if __name__ == "__main__":
    run_chatbot()


# Task 6
# Format chosen: Option A - Comment block

# 1. AI can give biased advice because it learns from existing text.
# It may favor certain industries, communication styles, or backgrounds.
# This could make its advice less useful for people with different experiences.

# 2. Users should review the AI's work before sending it to employers.
# The AI could include incorrect information or describe skills
# and experience in a way that does not accurately represent the user.

# 3. One guardrail I would add is a warning reminding users to
# review all generated content before submitting it.
# This helps users make sure their applications are accurate and honest.