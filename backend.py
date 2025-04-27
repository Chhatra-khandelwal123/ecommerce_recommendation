from fastapi import FastAPI, Query
import pandas as pd
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

def extract_main_category(category_tree):
    try:
        categories = ast.literal_eval(category_tree)
        return categories[0].split(" >> ")[0]
    except:
        return "Unknown"
    
app = FastAPI()

# Load dataset
df = pd.read_csv("/home/hhatrahandelwal/recommended_system/flipkart_com-ecommerce_sample.csv")
df['retail_price'] = df['retail_price'].fillna(999)

df['discounted_price'] = df['discounted_price'].fillna(499)
brand_counts = df['brand'].dropna().value_counts()
na_indices = df[df['brand'].isna()].index  # Get the indices of NaN values
sampled_brands = pd.Series(brand_counts.index).sample(n=len(na_indices), replace=True, weights=brand_counts.values).values
df.loc[na_indices, 'brand'] = sampled_brands
df_cleaned = df.dropna()
df=df_cleaned



df["main_category"] = df["product_category_tree"].apply(extract_main_category)


df["combined_text"] = df["product_name"] + " " + df["main_category"] + " " + df["description"]
df["combined_text"] = df["combined_text"].fillna("")


vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])


index = faiss.IndexFlatL2(tfidf_matrix.shape[1])
index.add(tfidf_matrix.toarray())


initial_products = df[["product_name"]].head(5).to_dict(orient="records")

@app.get("/")
def home():
    return {"message": "E-commerce Recommendation System API"}

@app.get("/get_initial_products")
def get_initial_products():
    return {"products": initial_products}

@app.get("/recommend")
def recommend(product_name: str = Query(..., description="Enter product name")):
    """Recommends 5 similar products based on input product"""
    if product_name not in df["product_name"].values:
        return {"message": "Product not found!"}

    # Get index of product
    product_index = df[df["product_name"] == product_name].index[0]

    # Compute similarity
    similarity_scores = list(enumerate(cosine_similarity(tfidf_matrix[product_index], tfidf_matrix)[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_products = [df.iloc[i[0]]["product_name"] for i in similarity_scores[1:6]]  # Top 5 similar

    return {"recommended_products": similar_products}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
