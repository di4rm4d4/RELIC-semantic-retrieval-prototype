import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from transformers import T5Tokenizer, T5ForConditionalGeneration

# load preprocessed data
with open("output.json", "r") as file:
    data = json.load(file)

# our mini model
model = SentenceTransformer("all-MiniLM-L6-v2")

# initialising pinecone
API_KEY = ""
REGION = "us-east-1"     
pc = Pinecone(api_key=API_KEY)

#creating an index
index_name = "semantic-search"
if not any(index.name == index_name for index in pc.list_indexes()):
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )
index = pc.Index(index_name)

# embedding data
for record in data:
    try:
        chunk = record["chunk"]
        metadata = {
            "chunk": record["chunk"], 
            "page": record["page"],
            "chapter": record["chapter"],
            "topic": record["topic"]
        }
        embedding = model.encode(chunk).tolist()
        index.upsert([{"id": f"chunk-{record['page']}", "values": embedding, "metadata": metadata}])
        print(f"Uploaded chunk {record['page']}")
    except Exception as e:
        print(f"Error processing record {record}: {e}")


t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")



def semantic_search(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    print(f"Query results: {results.matches}")  # Debugging
    cleaned_results = [
        {"chunk": res.metadata["chunk"], "score": res.score, "metadata": res.metadata}
        for res in results.matches if res.score > 0.8
    ]
    return cleaned_results

def generate_answer(query, top_results, max_length=300):
    context = "\n".join(
        f"Topic: {res['metadata']['topic']}, Page {res['metadata']['page']}: {res['chunk']}" 
        for res in top_results[:3]
    )
    print(f"Generated Context:\n{context}")  # Debugging
    input_text = f"Question: {query} \nContext: {context}\nProvide a detailed and coherent answer."
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
    output_ids = t5_model.generate(
        input_ids,
        max_length=max_length,
        num_beams=15,
        no_repeat_ngram_size=3,
        top_k=50,
        temperature=0.7
    )
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)





def attach_references(answer, top_results):
    references = [
        {
            "page": res["metadata"]["page"],
            "topic": res["metadata"]["topic"],
            "score": res["score"]
        }
        for res in top_results
    ]
    return {"answer": answer, "references": references}

#test questions
query = "What is an example of a worldview in history?"
search_results = semantic_search(query, top_k=10)  # Increase top_k to get more relevant chunks


answer = generate_answer(query, search_results, max_length=350)  # Set a higher max_length for more elaboration
final_result = attach_references(answer, search_results)


print(f"Answer: {final_result['answer']}")
for ref in final_result["references"]:
    print(f"Reference - Page: {ref['page']}, Topic: {ref['topic']}, Score: {ref['score']}")
