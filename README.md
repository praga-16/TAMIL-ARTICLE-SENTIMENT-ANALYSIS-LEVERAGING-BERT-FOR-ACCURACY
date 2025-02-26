# TAMIL-ARTICLE-SENTIMENT-ANALYSIS-LEVERAGING-BERT-FOR-ACCURACY
Sentiment analysis for Tamil articles presents unique challenges due to the language's complexity and the scarcity of resources. BERT (Bidirectional Encoder Representations from Transformers), particularly its multilingual variants, offers a powerful solution by understanding and analyzing Tamil text through its deep contextual learning. While SpaCy is a popular NLP library known for 1efficient tokenization and other preprocessing tasks, it lacks native support for Tamil. However, integrating SpaCy with a pre-trained multilingual BERT model can create a robust pipeline for sentiment analysis. 
In this approach, SpaCy can be used for basic text preprocessing, while BERT handles the sentiment prediction, outputting labels like "Positive" or "Negative" along with confidence scores. This combination is particularly useful for applications such as media monitoring, customer feedback analysis, and social media sentiment tracking in Tamil. 
Despite the challenges, including the need for custom tokenization and the complexity of Tamil's morphology, leveraging BERT and SpaCy together provides a strong framework for understanding the sentiment in Tamil texts, offering valuable insights across various domains.
The objective is to develop a robust sentiment analysis system for Tamil articles. Sentiment analysis involves determining the sentiment expressed in a text, typically categorizing it as positive, negative, or neutral. While existing sentiment analysis tools are highly effective for languages like English, there is a significant gap when it comes to analyzing sentiments in Tamil due to its unique linguistic structure and nuances.

## ARCHITECTURE DIAGRAM
![image](https://github.com/user-attachments/assets/ff96ea4e-8d5a-4186-b367-bc1ac11a5dae)

## code
```
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
df = pd.read_csv('dataset.csv')

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
    app.run(debug=True).


```



### index.html:
```
<!DOCTYPE html>
<html lang="ta">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tamil Sentiment Analysis</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #ff9a9e, #fad0c4);
            font-family: 'Arial', sans-serif;
        }
        .container {
            max-width: 600px;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            margin-top: 50px;
        }
        h1 {
            text-align: center;
            font-weight: bold;
            color: #333;
        }
        textarea {
            width: 100%;
            border: 2px solid #ff758c;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
        button {
            background: #ff758c;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background: #ff416c;
        }
        #result {
            text-align: center;
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
        }
        img {
            width: 100%;
            border-radius: 8px;
            margin-top: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
    </style>
</head>
<body>

<div class="container">
    <h1>தமிழ் உணர்வு பகுப்பாய்வு</h1>
    <form id="sentimentForm">
        <label for="sentence" class="form-label">தமிழ் வாக்கியத்தை உள்ளிடவும்:</label>
        <textarea id="sentence" name="sentence" rows="4" class="form-control"></textarea>
        <br>
        <button type="button" onclick="predictSentiment()">உணர்வு கணிக்க</button>
    </form>
    <div id="result"></div>
</div>

<script>
    function predictSentiment() {
        const sentence = document.getElementById('sentence').value;
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `sentence=${encodeURIComponent(sentence)}`,
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('result').innerHTML = `<p style="color:red;">❌ ${data.error}</p>`;
            } else {
                document.getElementById('result').innerHTML = `
                    <p style="color:green;">✅ நேர்மறை: ${data.positive}%</p>
                    <p style="color:red;">❌ எதிர்மறை: ${data.negative}%</p>
                    <img src="data:image/png;base64,${data.plot_url}" alt="Sentiment Plot">
                `;
            }
        });
    }
</script>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

## home page
![image](https://github.com/user-attachments/assets/e7965f08-4327-43b4-a668-218e956472f6)

## Positive input and output

![image](https://github.com/user-attachments/assets/79e3f744-4184-4b59-b07e-3ddad5dfbac2)
## report and scores
![image](https://github.com/user-attachments/assets/ea3b20d4-2bcf-42a3-bec0-2ea81c95a1b6)

## CONCLUSION

In conclusion, our Tamil sentiment analysis system, powered by BERT, represents a significant advancement in natural language processing for Tamil text. By leveraging the capabilities of transformer-based deep learning, the system efficiently classifies Tamil sentences into sentiment categories such as positive, negative, and neutral, along with domain-based categorization. The integration of context-aware deep learning techniques ensures that the system captures the intricate linguistic nuances of Tamil, making it highly accurate and reliable for sentiment analysis tasks. Our experimental evaluations demonstrate high classification accuracy, proving the model's robustness in handling diverse Tamil text inputs.
