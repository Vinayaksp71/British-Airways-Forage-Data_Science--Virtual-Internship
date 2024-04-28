import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read the scraped review data into a DataFrame
csv_file = "C:/Users/91948/Desktop/British Airways/scraped_reviews.csv"
df = pd.read_csv(csv_file)

# Clean the data
cleaned_reviews = df['Review Text'].str.replace('[^a-zA-Z\s]', '').str.lower()

# Sentiment Analysis
df['sentiment'] = cleaned_reviews.apply(lambda x: TextBlob(x).sentiment.polarity)

# Categorize sentiments into positive, negative, and neutral based on thresholds
positive_threshold = 0.1
negative_threshold = -0.1
neutral_threshold = 0.1
df['sentiment_category'] = pd.cut(df['sentiment'],
                                  bins=[-float("inf"), negative_threshold, positive_threshold, float("inf")],
                                  labels=['negative', 'neutral', 'positive'])

# Print number of positive, negative, and neutral reviews
print("From cleaning 1000 reviews given by the customers, we obtained")
print("Number of Positive Reviews:", (df['sentiment_category'] == 'positive').sum())
print("Number of Negative Reviews:", (df['sentiment_category'] == 'negative').sum())
print("Number of Neutral Reviews:", (df['sentiment_category'] == 'neutral').sum())

# Topic Modeling (LDA)
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(cleaned_reviews)

lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X)

# Top words for each topic
feature_names = vectorizer.get_feature_names_out()
top_words_per_topic = {}
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[:-10 - 1:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    top_words_per_topic[f"Topic {topic_idx + 1}"] = top_words

# Save the sentiment data in the specified format
output_folder = "C:/Users/91948/Desktop/British Airways"
output_file = os.path.join(output_folder, "analysis_results.txt")

# Display top words for each topic
with open(output_file, 'w') as f:
    f.write("Top words for each topic:\n")
    for topic, words in top_words_per_topic.items():
        f.write(f"{topic}: {', '.join(words[:5])}\n")

# Generate Word Cloud
all_text = ' '.join(cleaned_reviews)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display Word Cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.show()

# Save Word Cloud image
wordcloud_file = os.path.join(output_folder, "wordcloud.png")
wordcloud.to_file(wordcloud_file)
print(f"Word cloud image saved to {wordcloud_file}")

# Sentiment Pie Chart
sentiment_counts = df['sentiment_category'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Sentiment Categories')
plt.axis('equal')
plt.show()

# Save sentiment pie chart
sentiment_pie_file = os.path.join(output_folder, "sentiment_pie_chart.png")
plt.savefig(sentiment_pie_file)
print(f"Sentiment pie chart saved to {sentiment_pie_file}")
