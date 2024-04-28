import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = "C:\\Users\\91948\\Desktop\\British Airways\\2.customer_booking.csv"
try:
    df = pd.read_csv(data_path, encoding='latin1')
    print("CSV file loaded successfully.")
except UnicodeDecodeError as e:
    print(f"Error: Unable to decode the CSV file. {e}")
    exit()

# Preprocessing
# Convert categorical variables into numerical using LabelEncoder
label_encoders = {}
for column in df.select_dtypes(include=['object']):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Splitting the dataset into features and target variable
X = df.drop('booking_complete', axis=1)  # Features
y = df['booking_complete']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predictions on the testing set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred)

# Feature importance
feature_importance = pd.Series(rf_classifier.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(5).index.tolist()

# Print the results
print("We have trained the data set with Random forest classifier model and received:")
print(f"ACCURACY: {accuracy:.2f}")
print(f"AUC score: {auc_score:.3f}")
print("\nTop 5 features which influence Customer buying behavior:")
for feature in top_features:
    print(feature)

print("\nAnd plot top features that can drive successful flight booking:")

# Calculate Mutual Information Scores
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns)

# Plot Mutual Information Scores
def plot_mi_scores(features, scores):
    # Sort the features and scores by scores
    sorted_indices = np.argsort(scores)
    features = [features[i] for i in sorted_indices]
    scores = np.sort(scores)

    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(features)), scores)
    plt.yticks(np.arange(len(features)), features)
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Feature")
    plt.title("Top Features Driving Successful Flight Booking")
    plt.grid(axis='x')
    plt.show()

top_scores = [mi_scores[feature] for feature in top_features]
plot_mi_scores(top_features, top_scores)
