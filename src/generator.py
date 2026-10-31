import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def generate_answer(query: str, context_chunks: list[str]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not found. Please provide an API key in the sidebar."

    client = Groq(api_key=api_key)
    context_str = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are an enterprise document assistant. Answer the user query strictly based on "
        "the provided Context. If the answer cannot be determined directly from the Context, "
        "explicitly state that the document does not contain sufficient information. "
        "Include source page references where available."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

    # Hardcoded standard Groq chat completion models
    model_candidates = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "model="llama-3.3-70b-versatile"
    ]

    last_error = ""
    for model_name in model_candidates:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            continue

    return f"Error executing generation: {last_error}"
