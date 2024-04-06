import pandas as pd
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset of books
books_data = pd.read_csv('books.csv')

# Preprocess the book descriptions
def preprocess(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Stemming
    porter = PorterStemmer()
    stemmed_tokens = [porter.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

books_data['processed_description'] = books_data['description'].apply(preprocess)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(books_data['processed_description'])

# Define a function to recommend books based on user preferences
def recommend_books(author=None, genre=None, n=5):
    if author:
        author_books = books_data[books_data['author'].apply(lambda x: author.lower() in x.lower())]
    else:
        author_books = books_data.copy()
    if genre:
        genre_books = books_data[books_data['genre'].apply(lambda x: genre.lower() in x.lower())]
    else:
        genre_books = books_data.copy()
    
    recommended_books = pd.concat([author_books, genre_books]).drop_duplicates().head(n)
    return recommended_books[['title', 'author', 'description']]

# Define a function to interact with the user and recommend books
def chatbot():
    print("Welcome to the PAASHA ~ Book Recommendation Chatbot!")
    print("Please answer the following questions to help us recommend books for you.")
    
    favorite_author = input("Who is your favorite author? (Enter 'None' if you don't have one): ")
    preferred_genre = input("What genre of books do you prefer to read? (Enter 'None' if you don't have a preference): ")
    
    recommended_books = recommend_books(author=favorite_author, genre=preferred_genre)
    
    print("\nRecommended Books:")
    print(recommended_books)

# Run the chatbot
chatbot()
