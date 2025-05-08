import math

# Step 1: Define the database (for reference, though not used directly in the similarity calculation).
perfume_data = {
    "Chanel No. 5": "A timeless classic with floral aldehydes, jasmine, and sandalwood.",
    "Dior J'adore": "Luxurious feminine fragrance with ylang-ylang, damascus rose, and jasmine.",
    "Gucci Bloom": "Evocative bouquet of white flowers including jasmine, tuberose, and Rangoon creeper.",
    "Tom Ford Black Orchid": "Deep opulent scent with black truffle, bergamot, black orchid, and black plum.",
    "Jo Malone Peony & Blush Suede": "Charming blend of red apple, peony, and jasmine with hints of suede."
}

# Step 2: Define the attribute vectors for each perfume.
attribute_vectors={
"Chanel No. 5": {
"floral": 5,
"luxurious": 3,
"opulent": 2
},
"Dior J'adore": {
"floral": 5,
"luxurious": 5,
"opulent": 3
},
"Gucci Bloom": {
"floral": 5,
"luxurious": 4,
"opulent": 2
},
"Tom Ford Black Orchid": {
"floral": 2,
"luxurious": 4,
"opulent": 5
},
"Jo Malone Peony & Blush Suede": {
"floral": 4,
"luxurious": 3,
"opulent": 2
}
} 

# Step 3: Define the user preference vector in the order [timeless, floral, luxurious].
user_preference_vector = [1, 5, 5]

# Helper function to convert a perfume's attribute dictionary into a list 
# matching the same attribute order as the user preference vector.
def attribute_dict_to_list(attr_dict):
    return [
        attr_dict["floral"],
        attr_dict["luxurious"],
        attr_dict["opulent"]
    ]

# Function to calculate cosine similarity between two vectors.
def cosine_similarity(vecA, vecB):
    dot_product = sum(a * b for a, b in zip(vecA, vecB))
    normA = math.sqrt(sum(a * a for a in vecA))
    normB = math.sqrt(sum(b * b for b in vecB))
    return dot_product / (normA * normB)

# Step 4: Calculate the similarity for each perfume and track the best match.
best_product = None
highest_similarity = -1

for perfume, attrs in attribute_vectors.items():
    product_vector = attribute_dict_to_list(attrs)
    similarity = cosine_similarity(product_vector, user_preference_vector)
    print(f"{perfume}: similarity = {similarity:.2f}")

    if similarity > highest_similarity:
        highest_similarity = similarity
        best_product = perfume

# Step 5: Print the recommendation.
print(f"\nRecommended product: {best_product} (similarity = {highest_similarity:.2f})")