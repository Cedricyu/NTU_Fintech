import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already done
nltk.download('stopwords')

# Initialize the stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Normalize text
    text = text.lower()  # Convert to lowercase
    
    # Modify regex to keep Chinese characters
    text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff]', '', text)  # Keep Chinese characters

    tokens = text.split()  # Split into words
    
    # Remove stopwords and apply stemming (only for English words)
    tokens = [
        stemmer.stem(word) for word in tokens 
        if word not in stop_words or not word.isascii()  # Keep non-ASCII words (like Chinese)
    ]
    return ' '.join(tokens)
