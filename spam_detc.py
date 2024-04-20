import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
import nltk
import pyttsx3
import wave 
nltk.download('punkt')

class TextClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()   

# Download NLTK punkt data
nltk.download('punkt')

# Function to train the classifier
def train_classifier(data, column_name):
    # Drop rows with missing values
    data = data.dropna(subset=[column_name, 'spam'])
    
    # Preprocess the data
    data[column_name] = data[column_name].apply(lambda x: str(x).lower())
    data[column_name] = data[column_name].apply(lambda x: word_tokenize(x))
    data[column_name] = data[column_name].apply(lambda x: ' '.join(x))
    
    # Split the data into training and testing sets
    X = data[column_name]
    y = data['spam']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert the text data into numerical vectors
    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    # Convert the count data into TF-IDF vectors
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    
    # Train the classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    
    # Make predictions on the test data
    y_pred = classifier.predict(X_test_tfidf)
    
    # Print the classification report
    print(f"{column_name.capitalize()} Database Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print the confusion matrix
    print(f"{column_name.capitalize()} Database Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Load the Truecaller database
truecaller_data = pd.read_csv('truecaller_database.csv')

# Train the classifier for calls
train_classifier(truecaller_data, 'description')

# Text to speech function for calls
def text_to_speech_calls(text, output_file):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_file)
    engine.runAndWait()

# Create WAV file for "Spam call detected"
text_to_speech_calls("Spam call detected", "spam_call_detected.wav")

# Load the Messages database
messages_data = pd.read_csv('message_database.csv', encoding='latin1')

# Train the classifier for messages
train_classifier(messages_data, 'description')

# Text to speech function for messages
def text_to_speech_messages(text, output_file):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_file)
    engine.runAndWait()

# Create WAV file for "Spam message detected"
text_to_speech_messages("Spam message detected", "spam_message_detected.wav")

# Load the Email database
email_data = pd.read_csv('email_database.csv', encoding='ISO-8859-1')

# Train the classifier for email
train_classifier(email_data, 'subject')

# Text to speech function for email
def text_to_speech_email(text, output_file):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_file)
    engine.runAndWait()
