import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download stopwords and punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv('/Users/hamzakhan/Downloads/Movie Review Sentiment Analysis/movie_review.csv')

# Verify dataset columns
print("Dataset columns:", df.columns)
print(df.head())

# Define the column names
review_column = 'text'  # Column containing review text
sentiment_column = 'tag'  # Column containing sentiment labels ('pos' or 'neg')

# Preprocess text function
def preprocess_text(text):
    # Convert to lower case
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_review'] = df[review_column].apply(preprocess_text)

# Convert text to numerical representation using ( Bag-of-Words )
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])

# Convert labels to numerical values
y = df[sentiment_column].apply(lambda x: 1 if x == 'pos' else 0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict with Naive Bayes model
y_pred_nb = nb_model.predict(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Predict with Logistic Regression model
y_pred_lr = lr_model.predict(X_test)

# Evaluate Naive Bayes model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

print("Naive Bayes Model")
print(f'Accuracy: {accuracy_nb:.2f}')
print('Confusion Matrix:')
print(conf_matrix_nb)

# Evaluate Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

print("\nLogistic Regression Model")
print(f'Accuracy: {accuracy_lr:.2f}')
print('Confusion Matrix:')
print(conf_matrix_lr)
