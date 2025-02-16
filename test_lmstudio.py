from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key=""  # No API key needed for local LLMs
)

try:
    response = client.chat.completions.create(
        model="stable-code-instruct-3b",  # Use the EXACT model name from LM Studio
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")