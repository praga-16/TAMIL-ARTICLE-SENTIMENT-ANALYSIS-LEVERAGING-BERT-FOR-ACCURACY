import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import BertTokenizer, BertModel
import torch
import joblib
from flask import Flask, request, jsonify, render_template
import re
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Step 1: Load the Dataset
df = pd.read_csv(r'C:\Users\PRAGA\OneDrive\Desktop\New folder\tamil_sentiment_dataset_100.csv')


# Step 2: Data Preprocessing
df.dropna(subset=['Tamil Sentence'], inplace=True)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

def tokenize(sentences):
    return tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Step 3: Feature Extraction using BERT
class BERTFeatureExtractor:
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-multilingual-uncased')

    def extract_features(self, sentences):
        with torch.no_grad():
            inputs = tokenize(sentences)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy()

feature_extractor = BERTFeatureExtractor()

# Extract features from the dataset
features = feature_extractor.extract_features(df['Tamil Sentence'].tolist())

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform y_train to numerical labels
y_train_encoded = label_encoder.fit_transform(df['Sentiment'])

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    features, y_train_encoded, test_size=0.2, random_state=42
)

# Define models and hyperparameters
models = {
    'logisticRegression': (LogisticRegression(max_iter=1000), {'C': [1, 10], 'solver': ['liblinear']}),
    'SVC': (SVC(probability=True), {'C': [1, 10], 'kernel': ['linear', 'rbf']})
}

# Training function for models
def train_model(model, param_grid):
    grid = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

# Train models in parallel
best_models = {}
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(train_model, model, param_grid): name for name, (model, param_grid) in models.items()}
    for future in futures:
        name = futures[future]
        try:
            best_models[name] = future.result()
        except Exception as e:
            print(f"Error training {name}: {e}")

# Save the trained models
joblib.dump(best_models, 'sentiment_models.pkl')

# Load the saved models
sentiment_models = joblib.load('sentiment_models.pkl')

# Evaluate models on the test set and print detailed accuracy metrics
for name, model in sentiment_models.items():
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.2f}")

    print(f"\n{name} Classification Report:")

    unique_labels = sorted(list(set(y_test)))
    target_names = label_encoder.classes_

    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=[target_names[i] for i in unique_labels], zero_division=0))

    print(f"\n{name} Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {name}")
    plt.show()

# Function to predict sentiment percentage
def predict_sentiment_percentage(sentence):
    features = feature_extractor.extract_features([sentence])

    total_positive = 0
    total_negative = 0

    for name, model in sentiment_models.items():
        probabilities = model.predict_proba(features)[0]
        # Get the index of the 'Positive' class
        positive_class_index = list(label_encoder.classes_).index('Positive')
        total_positive += probabilities[positive_class_index]
        total_negative += 1 - probabilities[positive_class_index]

    total = total_positive + total_negative
    positive_percentage = (total_positive / total) * 100
    negative_percentage = (total_negative / total) * 100

    return positive_percentage, negative_percentage

# Function to create and save the plot
def create_plot(positive_percentage, negative_percentage):
    labels = ['Positive', 'Negative']
    sizes = [positive_percentage, negative_percentage]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, sizes, color=['blue', 'red'])
    plt.ylabel('Percentage')
    plt.title('Sentiment Analysis Results')
    plt.ylim(0, 100)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return plot_url

# Flask route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for handling sentence predictions
@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']

    tamil_pattern = re.compile(r'[\u0B80-\u0BFF]+')

    if not tamil_pattern.search(sentence):
        return jsonify({'error': "Please enter a valid Tamil sentence."})

    positive, negative = predict_sentiment_percentage(sentence)

    plot_url = create_plot(positive, negative)

    return jsonify({
        'positive': f"{positive:.2f}%",
        'negative': f"{negative:.2f}%",
        'plot_url': plot_url
    })

if __name__ == '__main__':
    app.run(debug=True)