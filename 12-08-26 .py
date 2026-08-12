
import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Load IMDb dataset
df = pd.read_csv("IMDB_Dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Reviews:")
print(df.head())

# 2. Check missing values and duplicates
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Reviews:", df['review'].duplicated().sum())

# Remove missing and duplicate reviews
df = df.dropna(subset=['review'])
df = df.drop_duplicates(subset=['review'])

# NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Cleaning function
def clean_review(text):

    # 3. Convert to lowercase
    text = text.lower()

    # 4. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 6. Tokenize
    words = word_tokenize(text)

    # 5. Remove stopwords
    words = [word for word in words if word not in stop_words]

    # 7. Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


# 8. Store cleaned reviews
df['cleaned_review'] = df['review'].apply(clean_review)

# 9. Compare original and cleaned reviews
print("\nOriginal vs Cleaned Reviews:")
print(df[['review', 'cleaned_review']].head())

# 10. Export preprocessed dataset
df.to_csv("IMDB_Preprocessed.csv", index=False)

print("\nPreprocessed dataset saved as IMDB_Preprocessed.csv")
