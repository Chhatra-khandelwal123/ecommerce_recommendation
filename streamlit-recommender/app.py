import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import os
from scipy.sparse import coo_matrix

import streamlit as st


train_data = pd.read_csv('marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv', sep='\t')
# print(train_data.columns)
train_data = train_data[['Uniq Id','Product Id', 'Product Rating', 'Product Reviews Count', 'Product Category', 'Product Brand', 'Product Name', 'Product Image Url', 'Product Description', 'Product Tags']]

# print(train_data.shape)
# print(train_data.isnull().sum())
train_data['Product Rating'].fillna(0, inplace=True)
train_data['Product Reviews Count'].fillna(0, inplace=True)
train_data['Product Category'].fillna('', inplace=True)
train_data['Product Brand'].fillna('', inplace=True)
train_data['Product Description'].fillna('', inplace=True)

# print(train_data.isnull().sum())
# print(train_data.duplicated().sum())
column_name_mapping = {
    'Uniq Id': 'ID',
    'Product Id': 'ProdID',
    'Product Rating': 'Rating',
    'Product Reviews Count': 'ReviewCount',
    'Product Category': 'Category',
    'Product Brand': 'Brand',
    'Product Name': 'Name',
    'Product Image Url': 'ImageURL',
    'Product Description': 'Description',
    'Product Tags': 'Tags',
    'Product Contents': 'Contents'
}
train_data.rename(columns=column_name_mapping, inplace=True)


train_data['ID'] = train_data['ID'].str.extract(r'(\d+)').astype(float)
train_data['ProdID'] = train_data['ProdID'].str.extract(r'(\d+)').astype(float)

###############################################EDA (Exploratory Data Analysis)#############################################

num_users = train_data['ID'].nunique()
num_items = train_data['ProdID'].nunique()
num_ratings = train_data['Rating'].nunique()
print(f"Number of unique users: {num_users}")
print(f"Number of unique items: {num_items}")
print(f"Number of unique ratings: {num_ratings}")
# heatmap_data = train_data.pivot_table('ID', 'Rating')


# plt.figure(figsize=(8, 6))
# sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='coolwarm', cbar=True)
# plt.title('Heatmap of User Ratings')
# plt.xlabel('Ratings')
# plt.ylabel('User ID')
# plt.show()

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

def clean_and_extract_tags(text):
    doc = nlp(text.lower())
    tags = [token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
    return ', '.join(tags)

columns_to_extract_tags_from = ['Category', 'Brand', 'Description']

for column in columns_to_extract_tags_from:
    train_data[column] = train_data[column].apply(clean_and_extract_tags)

train_data['Tags'] = train_data[columns_to_extract_tags_from].apply(lambda row: ', '.join(row), axis=1)
#Rating Base Recommendations System
average_ratings = train_data.groupby(['Name','ReviewCount','Brand','ImageURL'])['Rating'].mean().reset_index()
top_rated_items = average_ratings.sort_values(by='Rating', ascending=False)

rating_base_recommendation = top_rated_items.head(10)
rating_base_recommendation['Rating'] = rating_base_recommendation['Rating'].astype(int)
rating_base_recommendation['ReviewCount'] = rating_base_recommendation['ReviewCount'].astype(int)
# print("Rating Base Recommendation System: (Trending Products)")
rating_base_recommendation[['Name','Rating','ReviewCount','Brand','ImageURL']] = rating_base_recommendation[['Name','Rating','ReviewCount','Brand','ImageURL']]
rating_base_recommendation



rating_base_recommendation=rating_base_recommendation[['Name']]
def content_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        return pd.DataFrame({"Message": [f"Item '{item_name}' not found."]})

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name']]
    
    return recommended_items_details

# Collaborative Filtering Recommendation
def collaborative_recommendations(train_data, target_user_id, top_n=10):
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    
    if target_user_id not in user_item_matrix.index:
        return pd.DataFrame({"Message": [f"User {target_user_id} not found."]})

    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]

    similar_users_indices = user_similarities.argsort()[::-1][1:]
    recommended_items = []

    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    recommended_items_details = train_data[train_data['ProdID'].isin(recommended_items)][['Name']]
    
    return recommended_items_details

# Hybrid Recommendation
def hybrid_based_recommendations(train_data, target_user_id, item_name, top_n=10):
    content_based_rec = content_recommendations(train_data, item_name, top_n)
    collaborative_filtering_rec = collaborative_recommendations(train_data, target_user_id, top_n)
    
    if content_based_rec.empty or collaborative_filtering_rec.empty:
        return pd.DataFrame({"Message": ["No recommendations found."]})

    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()
    
    return hybrid_rec.head(10)

# Streamlit UI
st.title("E-Commerce Recommendation System")

st.subheader("Trending Products")
st.write(rating_base_recommendation)

# Content-Based Recommendation UI
st.subheader("Content-Based Recommendations")
item_name = st.text_input("Enter product name:")
if st.button("Get Recommendations", key='content'):
    recs = content_recommendations(train_data, item_name)
    st.write(recs)

# Collaborative Filtering UI
st.subheader("Collaborative Filtering Recommendations")
target_user_id = st.number_input("Enter User ID:", min_value=1, step=1)
if st.button("Get Recommendations", key='collaborative'):
    recs = collaborative_recommendations(train_data, target_user_id)
    st.write(recs)

# Hybrid Recommendation UI
st.subheader("Hybrid Recommendations")
target_user_id_hybrid = st.number_input("Enter User ID (Hybrid):", min_value=1, step=1)
item_name_hybrid = st.text_input("Enter product name (Hybrid):")
if st.button("Get Recommendations", key='hybrid'):
    recs = hybrid_based_recommendations(train_data, target_user_id_hybrid, item_name_hybrid)
    st.write(recs)