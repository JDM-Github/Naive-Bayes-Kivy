import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('MovieReviewTrainingDatabase.csv')
df['review'] = df['review'].fillna('Unknown')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

full_comments_file = 'FullComments.csv'
output_file = 'SentimentAnalysisResults.csv'

full_df = pd.read_csv(full_comments_file)
full_df['comment'] = full_df['comment'].fillna('Unknown')
predicted_sentiments = []

for comment in full_df['comment']:
    transformed_comment = vectorizer.transform([comment])
    prediction_probabilities = model.predict_proba(transformed_comment)
    neutral_prob = prediction_probabilities[0][0] < 0.55 and prediction_probabilities[0][1] < 0.55
    
    if neutral_prob:
        sentiment = 'Neutral'
    else:
        prediction = model.predict(transformed_comment)
        sentiment = encoder.inverse_transform(prediction)[0]
    
    predicted_sentiments.append(sentiment)

full_df['Predicted_Sentiment'] = predicted_sentiments
positive_comments = full_df[full_df['Predicted_Sentiment'] == 'Positive']
negative_comments = full_df[full_df['Predicted_Sentiment'] == 'Negative']
neutral_comments = full_df[full_df['Predicted_Sentiment'] == 'Neutral']

# Display and save results
print("\nPositive Comments:")
print(positive_comments['comment'].tolist())

print("\nNegative Comments:")
print(negative_comments['comment'].tolist())

print("\nNeutral Comments:")
print(neutral_comments['comment'].tolist())

# Save results to a CSV file
full_df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")
