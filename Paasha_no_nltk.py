import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset of books
books_data = pd.read_csv('books.csv')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books_data['description'])

# Define a function to recommend books based on user preferences
def recommend_books(author=None, genre=None, n=5):
    if author:
        author_books = books_data[books_data['author'].str.contains(author, case=False)]
    else:
        author_books = books_data.copy()
    if genre:
        genre_books = books_data[books_data['genre'].str.contains(genre, case=False)]
    else:
        genre_books = books_data.copy()
    
    recommended_books = pd.concat([author_books, genre_books]).drop_duplicates().head(n)
    return recommended_books[['title', 'author', 'description']]

# Define a function to interact with the user and recommend books
def chatbot():
    print("Welcome to the Book Recommendation Chatbot!")
    print("Please answer the following questions to help us recommend books for you.")
    
    favorite_author = input("Who is your favorite author? (Enter 'None' if you don't have one): ")
    preferred_genre = input("What genre of books do you prefer to read? (Enter 'None' if you don't have a preference): ")
    
    recommended_books = recommend_books(author=favorite_author, genre=preferred_genre)
    
    print("\nRecommended Books:")
    print(recommended_books)

# Run the chatbot
chatbot()
