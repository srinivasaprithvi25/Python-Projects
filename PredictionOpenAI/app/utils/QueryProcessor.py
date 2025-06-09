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

    schema_file = os.path.join(base_dir, '..', 'data', 'schema_info.json')
    schema_info = ''
    if os.path.exists(schema_file):
        with open(schema_file, 'r') as f:
            try:
                schema_info = json.dumps(json.load(f))
            except Exception:
                schema_info = ''

    db_type = os.getenv("DB_TYPE", "")
    full_prompt = (prompt_template
                   .replace("{{query}}", query)
                   .replace("{{db_type}}", db_type)
                   .replace("{{schema_metadata}}", schema_info))


    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}]
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nRaw response: {content}")

    # Normalise common fields returned as strings
    if "target_column" in result and isinstance(result["target_column"], str):
        result["target_column"] = [result["target_column"]]

    if "limit" in result:
        try:
            result["limit"] = int(result["limit"])
        except Exception:
            pass

    return result
