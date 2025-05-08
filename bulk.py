import os
import json
from openai import OpenAI
import numpy as np
from math import sqrt

# Initialize OpenAI v1 client using API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1) Synthetic dataset of 15 restaurants
database = {
    "Restaurant 1":  "A spicy spot offering vegan dishes in a cozy setting.",
    "Restaurant 2":  "A vegan and elegant restaurant known for its casual ambiance.",
    "Restaurant 3":  "A cozy café with barbecue options and fine‑dining touches.",
    "Restaurant 4":  "An elegant and casual grill offering seafood specialties.",
    "Restaurant 5":  "A spicy barbecue joint with vegan‑friendly sides in a cozy venue.",
    "Restaurant 6":  "A fine‑dining seafood restaurant with an elegant atmosphere.",
    "Restaurant 7":  "A casual vegan café with cozy seating and barbecue flavors.",
    "Restaurant 8":  "A seafood and vegan fusion spot in an elegant, cozy room.",
    "Restaurant 9":  "A spicy fine‑dining restaurant offering casual service.",
    "Restaurant 10": "A cozy barbecue house with casual, vegan‑friendly menu items.",
    "Restaurant 11": "An elegant vegan fine‑dining experience with barbecue notes.",
    "Restaurant 12": "A casual seafood café with spicy and cozy vibes.",
    "Restaurant 13": "A vegan fine‑dining restaurant in a cozy, elegant space.",
    "Restaurant 14": "A spicy seafood grill with casual, cozy décor.",
    "Restaurant 15": "An elegant barbecue restaurant offering vegan and seafood dishes."
}

# 2) Attributes list
attributes = [
    "spicy", "vegan", "cozy", "elegant", "casual", "seafood", "barbecue", "fine-dining"
]

# 3) LLM‑based attribute extraction helper with JSON cleaning
def extract_attribute_vectors(db: dict, attrs: list[str]) -> dict[str, dict[str, float]]:
    prompt = (
        f"You are an expert restaurant recommender. Here are {len(db)} items and their descriptions:\n"
        f"{json.dumps(db, indent=2)}\n"
        "Score each item on these attributes from 1 to 5, and return ONLY a JSON object mapping item names to attribute-score dicts."
        f" Attributes: {attrs}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        else:
            raise ValueError(f"Cannot parse JSON from LLM response: {raw}")

# 4) Cosine similarity function
def cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a*b for a, b in zip(u, v))
    nu  = sqrt(sum(a*a for a in u))
    nv  = sqrt(sum(b*b for b in v))
    return dot/(nu*nv) if nu and nv else 0.0

# 5) Algo2: user‑driven attribute recommendation
def recommend_algo2(attrs: list[str], attr_vecs: dict[str, dict[str, float]], user_vec: list[float], top_k: int = 5) -> list[str]:
    scores = []
    for item, vec in attr_vecs.items():
        item_v = [vec.get(a, 0.0) for a in attrs]
        scores.append((item, cosine(user_vec, item_v)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scores[:top_k]]

# 6) Simulated users for hold‑out testing
test_users = [
    {"liked": ["Restaurant 3","Restaurant 11","Restaurant 5"],  "hold_out": "Restaurant 11"},
    {"liked": ["Restaurant 13","Restaurant 7","Restaurant 2"],  "hold_out": "Restaurant 7"},
    {"liked": ["Restaurant 10","Restaurant 1","Restaurant 4"],  "hold_out": "Restaurant 1"},
    {"liked": ["Restaurant 6","Restaurant 14","Restaurant 9"],  "hold_out": "Restaurant 14"},
    {"liked": ["Restaurant 12","Restaurant 15","Restaurant 8"],  "hold_out": "Restaurant 12"},
    {"liked": ["Restaurant 2","Restaurant 9","Restaurant 5"],   "hold_out": "Restaurant 2"},
    {"liked": ["Restaurant 7","Restaurant 14","Restaurant 3"],  "hold_out": "Restaurant 14"},
    {"liked": ["Restaurant 4","Restaurant 11","Restaurant 1"],  "hold_out": "Restaurant 4"},
    {"liked": ["Restaurant 8","Restaurant 12","Restaurant 6"],  "hold_out": "Restaurant 6"},
    {"liked": ["Restaurant 15","Restaurant 10","Restaurant 13"], "hold_out": "Restaurant 15"}
]

# 7) Main test flow
if __name__ == "__main__":
    # Extract attribute vectors via LLM once
    attribute_vectors = extract_attribute_vectors(database, attributes)

    hits = []
    for case in test_users:
        seeds = [r for r in case["liked"] if r != case["hold_out"]]
        seed_vecs = [np.array(list(attribute_vectors[r].values())) for r in seeds]
        user_vector = np.mean(seed_vecs, axis=0).tolist()

        top5 = recommend_algo2(attributes, attribute_vectors, user_vector)
        hit = 1 if case["hold_out"] in top5 else 0
        hits.append(hit)
        print(f"Hold-out {case['hold_out']}: {'HIT' if hit else 'MISS'} -> {top5}")

    precision = sum(hits)/len(hits)
    print(f"Overall Precision@5 for Algo2: {precision:.2f}")

