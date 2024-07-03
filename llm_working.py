import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(page_title="How an LLM Works", layout="wide")

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stMarkdown {
        font-size: 18px;
    }
    .stTextInput > div > div > input {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .stTextArea > div > div > textarea {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .stPlotlyChart {
        background-color: #2E2E2E;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("How an LLM Works: From Input to Embedding")

# User input
user_input = st.text_area("Enter your query (multiple sentences for better visualization):", 
                          "Hello world what a beautiful world. The sky is blue. Birds are singing.")

if user_input:
    st.write("Processing your query...")

    # Tokenization
    def tokenize(text):
        tokens = text.split()
        st.subheader("1. Tokenization")
        st.latex(r"\text{Tokenization: } T(text) = [t_1, t_2, ..., t_n]")
        st.write(f"Tokens: {tokens}")
        return tokens

    tokens = tokenize(user_input)

    # Simple Encoding (for demonstration)
    def encode(tokens):
        st.subheader("2. Simple Encoding (Demonstration)")
        st.latex(r"\text{Encoding: } E(t_i) = hash(t_i) \mod 1000")
        encoded = [hash(token) % 1000 for token in tokens]
        st.write(f"Encoded: {encoded}")
        return encoded

    encoded = encode(tokens)

    # TF-IDF Vectorization
    st.subheader("3. TF-IDF Vectorization")
    st.write("Now, let's use TF-IDF vectorization to create more meaningful embeddings.")
    
    # Split input into sentences for better visualization
    sentences = user_input.split('.')
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    st.latex(r"\text{TF-IDF}(t, d, D) = tf(t, d) \cdot idf(t, D)")
    st.latex(r"\text{where } tf(t, d) = \frac{\text{frequency of term t in document d}}{\text{total terms in document d}}")
    st.latex(r"\text{and } idf(t, D) = \log \frac{\text{total documents in corpus D}}{\text{documents containing term t}}")

    st.write("TF-IDF Matrix:")
    st.write(tfidf_matrix.toarray())

    # Visualize TF-IDF
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(tfidf_matrix.toarray(), aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(vectorizer.get_feature_names_out())))
    ax.set_xticklabels(vectorizer.get_feature_names_out(), rotation=45, ha='right')
    ax.set_yticks(range(len(sentences)))
    ax.set_yticklabels([f"Sentence {i+1}" for i in range(len(sentences))])
    ax.set_title("TF-IDF Heatmap")
    plt.colorbar(im)
    st.pyplot(fig)

    # Dimensionality Reduction for Visualization
    st.subheader("4. Dimensionality Reduction (PCA)")
    st.write("To visualize high-dimensional embeddings, we use PCA to reduce dimensions.")
    
    if tfidf_matrix.shape[0] > 1:
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(tfidf_matrix.toarray())

        st.latex(r"\text{PCA: } X' = X W")
        st.write("Where X is the original data matrix and W is the projection matrix.")

        fig, ax = plt.subplots()
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1])
        for i, txt in enumerate(sentences):
            ax.annotate(f"Sentence {i+1}", (embedding_2d[i, 0], embedding_2d[i, 1]))
        ax.set_title("2D Projection of Sentence Embeddings")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)
    else:
        st.write("Not enough data for PCA visualization. Please enter multiple sentences.")

    # Simulated LLM Processing
    st.subheader("5. Simulated LLM Processing")
    st.write("This is a simplified representation of how an LLM might process the input.")

    # Attention Mechanism (simplified)
    attention_weights = np.random.rand(len(tokens))
    st.latex(r"\text{Attention: } A(e_i) = softmax(W \cdot e_i + b)")
    
    fig, ax = plt.subplots()
    ax.bar(tokens, attention_weights)
    ax.set_title("Simulated Attention Weights")
    ax.set_ylabel("Weight")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Simulated Layer Processing
    st.write("Processing through neural network layers:")
    for i in range(3):
        st.latex(f"Layer {i+1}: h_{{i+1}} = f(W_{{i+1}} \cdot h_i + b_{{i+1}})")
        st.progress(0.33 * (i+1))

    # Final Response
    st.subheader("6. LLM Response")
    response = f"Generated response for: {user_input}"
    st.write(response)

    st.write("Note: This is a simplified demonstration. Real LLMs involve much more complex processes and billions of parameters.")

else:
    st.write("Please enter a query to see how an LLM processes it.")
