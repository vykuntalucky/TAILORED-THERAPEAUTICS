import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re  # For punctuation removal
from nltk.corpus import stopwords  # For stop words
import nltk
from sklearn.model_selection import train_test_split
import xgboost as xgb  # Import XGBoost
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder
from collections import Counter  # Import Counter
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Download required NLTK data if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Handle non-string values (e.g., NaN)

    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text


def recommend_medicine(symptoms, smoking, drinking, genetics, df, threshold=0.2):  # Added threshold
    df['Cleaned_Symptoms'] = df['Symptoms'].apply(preprocess_text)
    df['Cleaned_Smoking'] = df['Smoking'].apply(preprocess_text)
    df['Cleaned_Drinking'] = df['Drinking'].apply(preprocess_text)
    df['Cleaned_Genetics'] = df['Genetics'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Cleaned_Symptoms'] + ' ' + df['Cleaned_Smoking'] + ' ' + df['Cleaned_Drinking'] + ' ' + df['Cleaned_Genetics'])
    
    user_input = preprocess_text(f"{symptoms} {smoking} {drinking} {genetics}")
    user_input_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)
    
    best_match_idx = similarities.argmax()
    best_match_similarity = similarities[0, best_match_idx]  # Get the similarity score
    best_match = df.iloc[best_match_idx]
    
    if best_match_similarity < threshold:
        return None  # No good match found

    recommendation = {
        "Disease": best_match['Possible_Disease'],
        "Recommended Medicine": best_match['Recommended_Medicine'],
        "Dosage": best_match['Dosage'],
        "Contraindications": best_match['Contraindications'],
        "Alternative Medicine": best_match['Alternative_Medicine'],
        "Similarity Score": best_match_similarity
    }
    
    return recommendation

@st.cache_resource  # Changed from st.cache to st.cache_resource (for objects like models)
def train_and_evaluate_model(df):
    """Trains an XGBoost model and returns the trained model and vectorizer."""
    df['Cleaned_Combined'] = df['Symptoms'].apply(preprocess_text) + ' ' + df['Smoking'].apply(preprocess_text) + ' ' + df['Drinking'].apply(preprocess_text) + ' ' + df['Genetics'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Cleaned_Combined'])
    y = df['Possible_Disease']

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Identify classes with fewer than 2 samples
    counts = Counter(y_encoded)
    rare_classes = [cls for cls, count in counts.items() if count < 2]

    # Remove rows with rare classes
    rows_to_keep = ~pd.Series(y_encoded).isin(rare_classes)
    X_filtered = X[rows_to_keep.to_numpy()]
    y_filtered = y_encoded[rows_to_keep.to_numpy()]

    # Relabel the filtered data so classes are contiguous from 0
    label_encoder_filtered = LabelEncoder()
    y_filtered_encoded = label_encoder_filtered.fit_transform(y_filtered)


    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered_encoded, test_size=0.2, random_state=42, stratify=y_filtered_encoded)  # Add stratify

    # Initialize XGBoost classifier with specified parameters
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # Specify multiclass classification objective
        num_class=len(label_encoder_filtered.classes_),  # Number of classes to predict
        n_estimators=100,  # Number of boosting rounds
        learning_rate=0.1,  # Step size shrinkage to prevent overfitting
        max_depth=3,  # Maximum depth of a tree
        subsample=0.8,  # Subsample ratio of the training instance
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        random_state=42,  # Random seed for reproducibility
        eval_metric='mlogloss'  # Metric used for multiclass log loss evaluation
    )
    model.fit(X_train, y_train)

    # Evaluate the model and print metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    st.subheader("Model Evaluation Metrics:")
    st.write(f"Accuracy: {accuracy}")
    st.text("Classification Report:\n" + classification_rep)
    st.write("Confusion Matrix:")
    st.dataframe(pd.DataFrame(confusion_mat))


    return model, vectorizer, label_encoder, label_encoder_filtered # Return label encoder and filtered label encoder

# Streamlit UI
st.title("Personalized Medical Recommendation System")

# Disclaimer
st.markdown(
    """
    **Disclaimer:** This system is for informational purposes only and does not constitute medical advice. 
    Always consult with a qualified healthcare professional for diagnosis and treatment.
    """
)

data_path = "data2.csv"
df = load_data(data_path)

# Train the Model (Run only once when the app starts)
model, vectorizer, label_encoder, label_encoder_filtered = train_and_evaluate_model(df) # Get LabelEncoder and mapping

# Symptom selection with autocompletion (example)
all_symptoms = df['Symptoms'].str.split(', ', expand=True).stack().unique()  # Extract all unique symptoms
symptoms = st.multiselect("Select your symptoms:", options=all_symptoms)
symptoms_str = ', '.join(symptoms)  # Convert list to comma-separated string

smoking = st.selectbox("Do you smoke?", ["Never", "Occasionally", "Frequently"])
drinking = st.selectbox("Do you drink?", ["Never", "Occasionally", "Frequently"])
genetics = st.selectbox("Genetic predisposition", ["Low", "Medium", "High"])


if st.button("Get Recommendation"):
    if symptoms_str: # Use the joined string
        # --- Cosine Similarity Recommendation ---
        result = recommend_medicine(symptoms_str, smoking, drinking, genetics, df) # Pass the string

        st.subheader("Recommendations:")

        # Cosine Similarity Result
        st.subheader("Cosine Similarity Recommendation:")
        if result:
            for key, value in result.items():
                st.write(f"**{key}:** {value}")
        else:
            st.warning("No suitable recommendation found based on the provided information (Cosine Similarity).")

    else:
        st.warning("Please select your symptoms.")