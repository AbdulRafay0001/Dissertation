import os
import json
from openai import OpenAI
import numpy as np
from math import sqrt
import re

# Initialize OpenAI v1 client using API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1) Synthetic dataset of 15 restaurants
database = {
    "Restaurant 1":  "Bold Indian cuisine specializing in rich curries and fiery spices, served in a warm, colorful space with traditional decor and lively music.",
    "Restaurant 2":  "Modern vegetarian and vegan dishes crafted from locally sourced produce, presented in a bright, casual atmosphere with communal tables and natural light.",
    "Restaurant 3":  "Contemporary Japanese sushi and sashimi made from premium fish, featuring inventive rolls and minimalist décor in a sleek, downtown loft setting.",
    "Restaurant 4":  "Farm-to-table seasonal menu highlighting local meats and produce, served in a cozy farmhouse-style dining room with reclaimed wood and soft lighting.",
    "Restaurant 5":  "Light seafood plates and ocean-inspired salads, enjoyed with panoramic waterfront views and a relaxed beach-house vibe of whitewashed walls and rattan accents.",
    "Restaurant 6":  "Authentic Mexican street tacos with a variety of fillings, fresh house-made salsas, and ice-cold margaritas in a colorful, festive taqueria setting.",
    "Restaurant 7":  "Mediterranean small plates like hummus, falafel, and grilled meats, served alongside warm pita in a friendly taverna with mosaic tile and lantern lighting.",
    "Restaurant 8":  "Artisanal coffees and freshly baked pastries crafted daily, enjoyed in a bright, airy café with plush seating, plants everywhere, and free Wi-Fi.",
    "Restaurant 9":  "Hand-pulled noodles and richly flavored broths from Sichuan and Northern China, served in a bustling, steam-filled shop with communal wooden tables.",
    "Restaurant 10": "Wood-fired Neapolitan pizzas topped with San Marzano tomatoes and fresh mozzarella, served in a rustic trattoria lined with red brick and vintage posters.",
    "Restaurant 11": "Creative plant-based takes on comfort classics, from jackfruit tacos to cashew-cream pasta, in a trendy, eco-friendly space decorated with repurposed pallets.",
    "Restaurant 12": "Classic French bistro fare—coq au vin, steak frites—paired with a curated wine list, served in an intimate dining room with art-nouveau accents.",
    "Restaurant 13": "Slow-smoked ribs, brisket, and pulled pork slathered in house-made sauces, served picnic-style on butcher-paper-lined tables in a laid-back smokehouse.",
    "Restaurant 14": "Bold blend of Asian and Latin flavors—Korean BBQ tacos, sushi burritos—served in a modern, industrial-chic space with open kitchen stalls.",
    "Restaurant 15": "Traditional Thai curries and noodle dishes bursting with spice and fresh herbs, served in a lantern-lit dining hall adorned with silk tapestries."
}

# 2) LLM call to pick the top k attributes with flexible parsing
def extract_top_attributes(db: dict, k: int = 3) -> list[str]:
    prompt = (
        f"Review the following items and their descriptions:\n{json.dumps(db, indent=2)}\n"
        "Identify the three single-word attributes that most distinctly characterise and differentiate this set. "
        "Output exactly three words in lowercase, separated by spaces, with no additional text or punctuation. "
        "Example output for a different set might be: modern organic relaxed"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()

    # 1) Try JSON array parse
    try:
        attrs = json.loads(raw)
        if isinstance(attrs, list) and len(attrs) == k:
            return attrs
    except json.JSONDecodeError:
        pass

    # 2) Fallback: split on whitespace if exactly k tokens
    tokens = raw.split()
    if len(tokens) == k:
        return [t.lower() for t in tokens]

    # 3) Fallback: extract bracketed list and parse
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        try:
            attrs = json.loads(raw[start:end+1])
            if isinstance(attrs, list) and len(attrs) == k:
                return attrs
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot parse attribute list from LLM response: {raw!r}")

# 3) LLM-based attribute extraction helper with robust JSON extraction
def extract_attribute_vectors(db: dict, attrs: list[str]) -> dict[str, dict[str, float]]:
    prompt = (
        "Score each of these restaurants on these attributes (1–5) and return ONLY a JSON object "
        "mapping each restaurant to its scores:\n" +
        json.dumps(db, indent=2) + "\nAttributes: " + json.dumps(attrs)
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing scores JSON: {e}\nRaw JSON: {raw[start:end]}")
    raise ValueError(f"Cannot parse attribute vectors from LLM response: {raw!r}")

# 4) Cosine similarity helper
def cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a*b for a, b in zip(u, v))
    nu = sqrt(sum(a*a for a in u))
    nv = sqrt(sum(b*b for b in v))
    return dot/(nu*nv) if nu and nv else 0.0

# 5) Recommendation logic
def recommend(attrs: list[str], attr_vecs: dict[str, dict[str, float]], user_vec: list[float], top_k: int = 5) -> list[str]:
    scores = []
    for item, vec in attr_vecs.items():
        item_vector = [vec.get(a, 0.0) for a in attrs]
        scores.append((item, cosine(user_vec, item_vector)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scores[:top_k]]

# 6) Simulated users for hold-out testing
test_users = [
    {"liked": ["Restaurant 6","Restaurant 9","Restaurant 13"], "hold_out": "Restaurant 13"},
    {"liked": ["Restaurant 7","Restaurant 4","Restaurant 8"],  "hold_out": "Restaurant 8"},
    {"liked": ["Restaurant 14","Restaurant 3","Restaurant 11"], "hold_out": "Restaurant 11"},
    {"liked": ["Restaurant 4","Restaurant 9","Restaurant 2"],  "hold_out": "Restaurant 2"},
    {"liked": ["Restaurant 5","Restaurant 2","Restaurant 15"], "hold_out": "Restaurant 15"},
    {"liked": ["Restaurant 13","Restaurant 10","Restaurant 3"], "hold_out": "Restaurant 3"},
    {"liked": ["Restaurant 14","Restaurant 6","Restaurant 12"], "hold_out": "Restaurant 12"},
    {"liked": ["Restaurant 4","Restaurant 8","Restaurant 2"],  "hold_out": "Restaurant 1"}
]

top_k = 5
precisions, recalls = [], []

for case in test_users:
    # build the user vector from the non-held-out likes:
    seeds      = [r for r in case["liked"] if r not in case["hold_out"]]
    seed_vecs  = [np.array(list(attr_vecs[r].values())) for r in seeds]
    user_vec   = np.mean(seed_vecs, axis=0).tolist()

    # get top-K recommendations:
    recs = recommend(attrs, attr_vecs, user_vec, top_k=top_k)

    # count how many held-out items were recovered:
    hits      = sum(1 for h in case["hold_out"] if h in recs)
    num_hold  = len(case["hold_out"])

    precision = hits / top_k
    recall    = hits / num_hold

    precisions.append(precision)
    recalls.append(recall)

    print(f"User held-out {case['hold_out']}: "
          f"Precision@{top_k}={precision:.2f}, Recall@{top_k}={recall:.2f}")

# Aggregate over all users:
mean_prec = sum(precisions) / len(precisions)
mean_rec  = sum(recalls)    / len(recalls)
print(f"\nMean Precision@{top_k}: {mean_prec:.2f}")
print(f"Mean Recall@{top_k}:    {mean_rec:.2f}")
