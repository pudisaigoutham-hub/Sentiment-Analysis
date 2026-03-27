import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('IMDB Dataset.csv', quoting=3, on_bad_lines='skip').head(5000)
df.dropna(inplace=True)

text_col = 'review' if 'review' in df.columns else df.columns[0]
label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]

df['target'] = df[label_col].astype(str).str.lower().map({'positive': 1, 'negative': 0})
df = df.dropna(subset=['target'])

tfidf = TfidfVectorizer(stop_words='english', max_features=2500)
X = tfidf.fit_transform(df[text_col].astype(str))
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB().fit(X_train, y_train)

print(f"--- Task 2: Results for {len(df)} Reviews ---")
print(classification_report(y_test, model.predict(X_test)))

df[label_col].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('IMDb Sentiment Distribution')
plt.ylabel('Count')
plt.show()