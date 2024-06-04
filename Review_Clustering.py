#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Download NLTK stopwords
nltk.download('stopwords')

# Set up NLTK stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    
    # Tokenize the text and remove stopwords
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def preprocess_new_text(new_text):
    preprocessed_text = preprocess_text(new_text)
    return preprocessed_text

def main():
    st.title("Review Clustering")

    # Directory containing your text files
    data_dir = 'C:/Users/Admin/Documents/Data science msc/IR/cw/q2'

    # List to store preprocessed text
    preprocessed_texts = []

    # Iterate over each file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Preprocess the text
            preprocessed_text = preprocess_text(text)
            preprocessed_texts.append(preprocessed_text)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_texts)

    # Apply KMeans clustering
    num_clusters = 3  # Adjust based on your dataset
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Map cluster labels to meaningful names
    cluster_mapping = {0: 'restaurant', 1: 'room', 2: 'service'}
    labels_mapped = [cluster_mapping[label] for label in labels]

    # Add the cluster labels to the preprocessed_texts list
    preprocessed_texts_with_labels = pd.DataFrame({'Text': preprocessed_texts, 'Cluster': labels_mapped})

    # Create a table showing the count of each cluster
    cluster_counts = preprocessed_texts_with_labels['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']

    # Display the table
    st.subheader("Cluster Counts:")
    st.write(cluster_counts)

    # Silhouette Score
    silhouette_avg = silhouette_score(X, labels)
    st.subheader(f"Silhouette Score: {silhouette_avg}")

    # Display the cluster labels in the app
    st.subheader("Cluster Labels:")
    for cluster, label in cluster_mapping.items():
        st.write(f"Cluster {cluster}: {label}")

    # New text input
    st.subheader("Enter New Review:")
    new_text = st.text_area("Type here:", "Everything was really Good and the service was excellent")

    # Add a button to proceed
    if st.button("Cluster New Review"):
        # Preprocess the new text
        preprocessed_new_text = preprocess_new_text(new_text)

        # Transform the new text using the pre-trained TF-IDF vectorizer
        new_text_vectorized = vectorizer.transform([preprocessed_new_text])

        # Predict the cluster using KMeans model
        predicted_cluster = kmeans.predict(new_text_vectorized)

        # Display the predicted cluster for the new text
        st.subheader(f"The new review belongs to Cluster: {cluster_mapping[predicted_cluster[0]]}")

if __name__ == "__main__":
    main()

