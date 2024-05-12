import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.optimizers import Adam

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
# Assuming 'ratings.csv' has columns: userId, movieId, rating
def load_data():
    """
    Load and preprocess data.
    Returns processed train and test datasets.
    """
    data = pd.read_csv('ratings.csv')
    user_ids = data['userId'].unique()
    movie_ids = data['movieId'].unique()

    # Create user and movie mappings to continuous indices
    user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    
    # Encode users and movies in the dataset
    data['user'] = data['userId'].map(user_to_user_encoded)
    data['movie'] = data['movieId'].map(movie_to_movie_encoded)

    # Number of users and movies
    num_users = len(user_to_user_encoded)
    num_movies = len(movie_to_movie_encoded)

    # Split data into train and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    return train, test, num_users, num_movies

# Build the model
def RecommenderV1(num_users, num_movies, embedding_size=50):
    """
    Model builder function.
    Args:
    num_users: Total number of users
    num_movies: Total number of movies
    embedding_size: Number of dimensions for embeddings

    Returns:
    A Keras model instance.
    """
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    # Embedding layers
    user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
    movie_embedding = Embedding(num_movies, embedding_size, name='movie_embedding')(movie_input)

    # Multiply the outputs of the embedding layers
    x = Dot(axes=1)([user_embedding, movie_embedding])

    # Bias layers
    user_bias = Embedding(num_users, 1, name='user_bias')(user_input)
    movie_bias = Embedding(num_movies, 1, name='movie_bias')(movie_input)

    # Add biases to the dot product
    x = Add()([x, user_bias, movie_bias])
    x = Flatten()(x)

    # Create model
    model = Model(inputs=[user_input, movie_input], outputs=x)
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')

    return model

# Main execution function
def main():
    """
    Main function to execute data loading, model training, and evaluation.
    """
    # Load data
    train, test, num_users, num_movies = load_data()
    
    # Initialize and compile the model
    model = RecommenderV1(num_users, num_movies)
    
    # Train the model
    model.fit(
        [train['user'], train['movie']],
        train['rating'],
        batch_size=32,
        epochs=5,
        verbose=1,
        validation_split=0.1
    )

    # Predict and evaluate the model
    predictions = model.predict([test['user'], test['movie']])
    error = mean_squared_error(test['rating'], predictions)
    print(f"Test MSE: {error}")

if __name__ == "__main__":
    main()
