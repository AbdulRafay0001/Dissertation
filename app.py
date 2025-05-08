# app.py
from fastapi import FastAPI, Form, HTTPException
import math, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Helper: classical cosine similarity
def cosine(u, v):
    dot = sum(a*b for a, b in zip(u, v))
    na  = math.sqrt(sum(a*a for a in u))
    nb  = math.sqrt(sum(b*b for b in v))
    return dot/(na*nb) if na and nb else 0.0

# 1) Modular attribute-based recommendation
@app.post("/recommend")
def recommend(
    attributes: str = Form(...),
    attribute_vectors: str = Form(...),
    user_vector: str = Form(...)
):
    try:
        vecs = json.loads(attribute_vectors)
        user = json.loads(user_vector)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON in payload: {e}")
    try:
        attrs = json.loads(attributes)
    except json.JSONDecodeError:
        attrs = attributes.strip().split()

    best_item, best_score = None, -1.0
    for item, vec_dict in vecs.items():
        try:
            v = [vec_dict[a] for a in attrs]
        except KeyError as e:
            raise HTTPException(400, f"Missing attribute {e} in item {item}")
        score = cosine(user, v)
        if score > best_score:
            best_item, best_score = item, score

    return {"recommendation": best_item, "score": round(best_score, 3)}

# 2) TF–IDF user-profile recommendation
@app.post("/recommend_tfidf_user")
def recommend_tfidf_user(
    database: str          = Form(...),  # JSON string of {item: description, …}
    attributes: str        = Form(...),  # '["fresh","woody","spicy"]' or 'fresh woody spicy'
    user_vector: str       = Form(...)   # '[3,5,4]' or '3 5 4'
):
    try:
        data = json.loads(database)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Bad JSON in database: {e}")
    try:
        attrs = json.loads(attributes)
    except:
        attrs = attributes.strip().split()
    try:
        user = json.loads(user_vector)
    except:
        user = [float(x) for x in user_vector.strip().split()]

    names = list(data.keys())
    docs  = list(data.values())
    vectorizer   = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)
    idf_lookup   = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    vocab        = vectorizer.vocabulary_

    D = len(vocab)
    profile = [0.0]*D
    for attr, score in zip(attrs, user):
        key = attr.lower()
        idx = vocab.get(key)
        if idx is not None:
            profile[idx] = score * idf_lookup[key]

    sims = []
    for i, name in enumerate(names):
        item_vec = tfidf_matrix[i].toarray()[0].tolist()
        sims.append((name, cosine(profile, item_vec)))

    top_k = 5
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
    return {"recommendations": [{"item": nm, "score": round(sc, 3)} for nm, sc in sims]}

# 3) Two-stage user-refined pipeline
@app.post("/recommend_user_refined")
def recommend_user_refined(
    database:           str = Form(...),  # JSON string of { item_name: description, … }
    attribute_vectors:  str = Form(...),  # JSON string of { item_name: {attr: weight,…}, … }
    attributes:         str = Form(...),  # '["attr1","attr2",...]' or 'attr1 attr2 ...'
    user_vector:        str = Form(...)   # '[r1,r2,...]' or 'r1 r2 ...'
):
    try:
        db   = json.loads(database)
        vecs = json.loads(attribute_vectors)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}")
    try:
        attrs = json.loads(attributes)
    except:
        attrs = attributes.strip().split()
    try:
        user = json.loads(user_vector)
    except:
        user = [float(x) for x in user_vector.strip().split()]

    def cos(u, v):
        dot = sum(a*b for a,b in zip(u,v))
        nu  = math.sqrt(sum(a*a for a in u))
        nv  = math.sqrt(sum(b*b for b in v))
        return dot/(nu*nv) if nu and nv else 0.0

    scores = []
    for item, vec_dict in vecs.items():
        try:
            v = [vec_dict[a] for a in attrs]
        except KeyError:
            continue
        scores.append((item, cos(user, v)))

    top_k = 3
    topN = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    items = [itm for itm,_ in topN]
    refined = { itm: db[itm] for itm in items if itm in db }
    return {"top_items": items, "top_scores": [round(sc,3) for _,sc in topN], "refined_database": json.dumps(refined)}
