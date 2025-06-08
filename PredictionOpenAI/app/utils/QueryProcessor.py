import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_query(query):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, '..', 'prompts', 'prompt_template.txt')

    with open(os.path.abspath(prompt_path), 'r') as f:
        prompt_template = f.read()

    db_type = os.getenv("DB_TYPE", "")
    full_prompt = (prompt_template
                   .replace("{{query}}", query)
                   .replace("{{db_type}}", db_type))


    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": full_prompt}
        ]
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nRaw response: {content}")

    # Normalize tables/joins fields if they were returned as strings
    for key in ("tables", "joins"):
        if key in result and isinstance(result[key], str):
            try:
                result[key] = json.loads(result[key])
            except Exception:
                # Fallback to comma separated list for tables
                if key == "tables":
                    result[key] = [t.strip() for t in result[key].split(',')]
    return result
