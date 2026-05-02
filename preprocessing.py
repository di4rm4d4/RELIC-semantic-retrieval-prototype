"""""
import re
import json
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "dolly-v2-12b"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


json_schema = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter_number": {"type": "number"},
                    "chapter_title": {"type": "string"},
                    "page_start": {"type": "number"},
                    "page_end": {"type": "number"},
                    "content": {"type": "string"}
                },
                "required": ["chapter_number", "chapter_title", "page_start", "page_end", "content"]
            }
        }
    },
    "required": ["chapters"]
}


def preprocess_text(file_path):
    with open(file_path, "r") as file:
        text = file.read()

    
    pages = re.split(r"\f", text)
    return pages

def generate_json(text_pages, json_schema, tokenizer, model):
    formatted_pages = []
    for page_number, page_content in enumerate(text_pages, start=1):
        chapter_match = re.search(r"Chapter (\\d+): (.+)", page_content)
        if chapter_match:
            chapter_number = int(chapter_match.group(1))
            chapter_title = chapter_match.group(2).strip()
        else:
            chapter_number = None
            chapter_title = None

        
        formatted_pages.append({
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "page_start": page_number,
            "page_end": page_number,
            "content": page_content.strip()
        })

    
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt="Process the following textbook pages into structured JSON:")
    output_json = jsonformer()
    return output_json


file_path = "worldviews.txt"
text_pages = preprocess_text(file_path)


structured_json = generate_json(text_pages, json_schema, tokenizer, model)


output_file = "textbook.json"
with open(output_file, "w") as json_file:
    json_file.write(json.dumps(structured_json, indent=4))

print(f"JSON output saved to {output_file}")

"""""



import re
import json


from jsonformer import Jsonformer


from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Dolly properly this time 

model_name = "/Users/dianamcdermott/Paulina/dolly-v2-12b"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# JSON schema 

json_schema = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter_number": {"type": "number"},
                    "chapter_title": {"type": "string"},
                    "page_start": {"type": "number"},
                    "page_end": {"type": "number"},
                    "content": {"type": "string"}
                },
                "required": [
                    "chapter_number",
                    "chapter_title",
                    "page_start",
                    "page_end",
                    "content"
                ]
            }
        }
    },
    "required": ["chapters"]
}


def preprocess_text(file_path):
    """
    Read the textbook text from a file, split on form feed (\f),
    and return a list of page contents.
    """
    with open(file_path, "r") as file:
        text = file.read()
    
    pages = re.split(r"\f", text)
    return pages


def generate_json(text_pages, json_schema, tokenizer, model):
    """
    1) Build a big prompt that includes all pages.
    2) Call Jsonformer to produce JSON matching the schema.
    """



  
    prompt_text = "Process the following textbook pages into structured JSON.\n\n"
    for i, page_content in enumerate(text_pages, start=1):
        prompt_text += f"--- Page {i} ---\n"
        prompt_text += page_content.strip() + "\n\n"

    

    jsonformer = Jsonformer(
        model=model,
        tokenizer=tokenizer,
        schema=json_schema,
        prompt=prompt_text,
        max_length=2048  
    )

    
    output_json = jsonformer()

    return output_json




file_path = "worldviews.txt"
text_pages = preprocess_text(file_path)

structured_json = generate_json(text_pages, json_schema, tokenizer, model)


output_file = "textbook.json"
with open(output_file, "w") as json_file:
    json.dump(structured_json, json_file, indent=4)

print(f"JSON output saved to {output_file}")



