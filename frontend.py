import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("🛒 E-commerce Product Recommendation")

# Load initial products
if "products" not in st.session_state:
    response = requests.get(f"{API_URL}/get_initial_products").json()
    st.session_state.products = response["products"]

st.subheader("🔝 Top 5 Trending Products")
for product in st.session_state.products:
    st.write(f"🛍 {product['product_name']}")

st.subheader("🔍 Search for a Product")
search_query = st.text_input("Enter a product name:")

if search_query:
    response = requests.get(f"{API_URL}/recommend", params={"product_name": search_query}).json()
    if "recommended_products" in response:
        st.subheader("✨ Recommended Products")
        for rec in response["recommended_products"]:
            st.write(f"✅ {rec}")
    else:
        st.write("⚠ Product not found. Try another name!")
