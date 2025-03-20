from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example dictionary: keys are perfume names, values are their descriptions.
perfume_data = {
    "Chanel No. 5": "A timeless classic with floral aldehydes, jasmine, and sandalwood.",
    "Dior J'adore": "Luxurious feminine fragrance with ylang-ylang, damascus rose, and jasmine.",
    "Gucci Bloom": "Evocative bouquet of white flowers including jasmine, tuberose, and Rangoon creeper.",
    "Tom Ford Black Orchid": "Deep opulent scent with black truffle, bergamot, black orchid, and black plum.",
    "Jo Malone Peony & Blush Suede": "Charming blend of red apple, peony, and jasmine with hints of suede."
}

# Step 1: Convert the perfume descriptions into TF-IDF vectors.
#TfidfVectorizer class is used to convert the perfume descriptions into TF-IDF vectors.
# This vector represents the importance of each word
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(perfume_data.values())

# Step 2: Compute the cosine similarity matrix.
# Cosin Similarity function computes a cosine similarity score between each pair of vectors 
# A high cosine similarity means that two descriptions are very similar , in terms of the words they use
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map indices to perfume names.
perfume_names = list(perfume_data.keys())

# For example, to get recommendations for "Chanel No. 5":
target_index = perfume_names.index("Chanel No. 5")
similarities = list(enumerate(cosine_sim[target_index]))

# Sort the perfumes based on similarity score (excluding the query itself).
similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
print("Recommendations for 'Chanel No. 5':")
for idx, score in similarities[1:]:
    print(f"{perfume_names[idx]}: similarity = {score:.2f}")

"""
A limitation of this is that it is based on the words used in the descriptions,and only directly compares words , rather than 
what they actually mean. For example, if two perfumes are described as "floral" and "flowery", they may not be considered similar, eventhough
they are used to describe the same concept.
"""