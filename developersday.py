import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# --- 1. Φόρτωση και Καθαρισμός ---
df = pd.read_csv(r"C:\Users\konos\Documents\techno\Programming\developers\customer_reviews.csv")
df.dropna(subset=['review_text', 'sentiment'], inplace=True)
target_classes = sorted(df['sentiment'].unique())

# --- 2. Συνάρτηση Καθαρισμού Κειμένου ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['review_text_cleaned'] = df['review_text'].apply(clean_text)

# --- 3. Διαχωρισμός Δεδομένων (Μόνο τα χρήσιμα Features) ---
numeric_features = ['stars', 'shipping_days']
text_feature = 'review_text_cleaned'

# Επιλέγουμε ΜΟΝΟ τις στήλες που θέλουμε
X = df[['review_text_cleaned', 'stars', 'shipping_days']] 
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- 4. Δημιουργία Preprocessor (Χωρίς κατηγορίες) ---
text_transformer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_feature),
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='drop',
    sparse_threshold=0  # <-- Η κρίσιμη διόρθωση για το bug!
)

# --- 5. Δημιουργία και Εκπαίδευση του Τελικού Pipeline ---
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', HistGradientBoostingClassifier(
        random_state=42,
        class_weight='balanced'
    ))
])

# Εκπαίδευση του μοντέλου
pipeline.fit(X_train, y_train)

# --- 6. ΑΞΙΟΛΟΓΗΣΗ ΚΑΙ ΑΠΟΤΕΛΕΣΜΑΤΑ ---
y_pred = pipeline.predict(X_test)

# Υπολογισμός μετρικών
macro_f1 = f1_score(y_test, y_pred, average='macro', labels=target_classes)
cm = confusion_matrix(y_test, y_pred, labels=target_classes)
cm_df = pd.DataFrame(cm,
                     index=[f"Actual: {l}" for l in target_classes],
                     columns=[f"Predicted: {l}" for l in target_classes])
report = classification_report(y_test, y_pred, labels=target_classes, digits=4)

# --- Εκτύπωση ΜΟΝΟ των Αποτελεσμάτων ---

print("\n--- Classification Report ---")
print(report)

# *** ΑΥΤΗ Η ΓΡΑΜΜΗ ΑΛΛΑΞΕ ***
print("\n--- Confusion Matrix ---")
print(cm_df)
## changed this line all good 
print("\n--------------------------------------------------")
# *** ΑΥΤΗ Η ΓΡΑΜΜΗ ΑΛΛΑΞΕ ***
print(f"  Main Metric (Macro F1-Score): {macro_f1:.4f}")
print("--------------------------------------------------")