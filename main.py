import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import re

print("="*70)
print("LOADING ENRON DATASET")
print("="*70)

data = pd.read_csv('enron_spam_data.csv')
print(f"Loaded {data.shape[0]} emails")
print(data['Spam/Ham'].value_counts())

data['label'] = data['Spam/Ham']
data['message'] = data['Subject'].fillna('') + " " + data['Message'].fillna('')

# Clean messages: remove special characters, keep only letters, numbers, and spaces
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Keep only letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['message'] = data['message'].apply(clean_text)
data = data[['label', 'message']]

# Remove empty messages after cleaning
data = data[data['message'].str.len() > 0]

ham = data[data['label'] == 'ham']
spam = data[data['label'] == 'spam']
min_samples = min(ham.shape[0], spam.shape[0])
ham = ham.sample(min_samples, random_state=42)
spam = spam.sample(min_samples, random_state=42)
data = pd.concat([ham, spam], axis=0, ignore_index=True)

data['label_binary'] = (data['label'] == 'spam').astype(int)

print(f"\nBalanced dataset: {data.shape[0]} messages ({data['label'].value_counts()['ham']} ham, {data['label'].value_counts()['spam']} spam)")

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], 
    data['label'], 
    test_size=0.2, 
    random_state=42, 
    shuffle=True, 
    stratify=data['label']
)

X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    data['message'], 
    data['label_binary'], 
    test_size=0.2, 
    random_state=42, 
    shuffle=True, 
    stratify=data['label_binary']
)

print("\n" + "="*70)
print("TRAINING ALL THREE MODELS WITH FILTERED VOCABULARY")
print("="*70)

# Custom CountVectorizer with filtering
vectorizer_params = {
    'min_df': 2,           # Word must appear in at least 2 documents
    'max_df': 0.95,        # Word must appear in less than 95% of documents
    'max_features': 1000,  # Keep only top 1000 features
    'ngram_range': (1, 1), # Only unigrams
    'token_pattern': r'\b[a-z]{2,}\b'  # Only words with 2+ letters
}

rf_model = Pipeline([
    ('vectorizer', CountVectorizer(**vectorizer_params)),
    ('classifier', RandomForestClassifier(
        n_estimators=3,
        max_depth=5,
        random_state=42
    ))
])
print("Training Random Forest...")
rf_model.fit(X_train, y_train)
print("Random Forest trained")

xgb_model = Pipeline([
    ('vectorizer', CountVectorizer(**vectorizer_params)),
    ('classifier', xgb.XGBClassifier(
        n_estimators=3,
        max_depth=3,
        learning_rate=0.3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])
print("Training XGBoost...")
xgb_model.fit(X_train_binary, y_train_binary)
print("XGBoost trained")

lr_model = Pipeline([
    ('vectorizer', CountVectorizer(**vectorizer_params)),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])
print("Training Logistic Regression...")
lr_model.fit(X_train, y_train)
print("Logistic Regression trained")

# Print vocabulary size
print(f"\nVocabulary size: {len(rf_model.named_steps['vectorizer'].get_feature_names_out())} words")

test_messages = [
    'WINNER! You have won $1000. Click here now!',
    'Hey mom, can you pick me up after school?',
    'FREE money! Call now to claim your prize!',
    'Thanks for the notes from yesterday meeting'
]

for msg_idx, message in enumerate(test_messages):
    print("\n" + "="*70)
    print(f"MESSAGE {msg_idx + 1}: {message}")
    print("="*70)
    
    # Clean the test message
    cleaned_message = clean_text(message)
    print(f"CLEANED: {cleaned_message}")
    
    print("\nRANDOM FOREST")
    print("-" * 70)
    
    rf_vectorizer = rf_model.named_steps['vectorizer']
    rf_classifier = rf_model.named_steps['classifier']
    X_transformed_rf = rf_vectorizer.transform([cleaned_message])
    
    print("\nIndividual Tree Votes:")
    rf_votes = []
    for tree_idx, tree_estimator in enumerate(rf_classifier.estimators_):
        tree_pred = tree_estimator.predict(X_transformed_rf)[0]
        rf_votes.append(tree_pred)
        print(f"  Tree {tree_idx + 1}: {str(tree_pred).upper()}")
    
    unique, counts = np.unique(rf_votes, return_counts=True)
    vote_dict = dict(zip(unique, counts))
    
    print("\nVoting Summary:")
    for label in unique:
        count = vote_dict[label]
        percentage = (count / len(rf_votes)) * 100
        bar = "#" * int(percentage / 10)
        print(f"  {str(label).upper()}: {count}/3 trees ({percentage:.1f}%) {bar}")
    
    rf_final = rf_model.predict([cleaned_message])[0]
    rf_proba = rf_model.predict_proba([cleaned_message])[0]
    print(f"\nFINAL: {str(rf_final).upper()} | ham={rf_proba[0]:.3f}, spam={rf_proba[1]:.3f}")
    
    print("\n" + "-" * 70)
    print("XGBOOST")
    print("-" * 70)
    
    xgb_vectorizer = xgb_model.named_steps['vectorizer']
    xgb_classifier = xgb_model.named_steps['classifier']
    X_transformed_xgb = xgb_vectorizer.transform([cleaned_message])
    
    print("\nSequential Tree Contributions:")
    prev_spam_score = 0.5
    for n_trees in range(1, 4):
        pred_proba = xgb_classifier.predict_proba(X_transformed_xgb, iteration_range=(0, n_trees))[0]
        spam_score = pred_proba[1]
        adjustment = spam_score - prev_spam_score if n_trees > 1 else spam_score - 0.5
        
        bar_length = int(abs(adjustment) * 100)
        bar = ">" * bar_length if adjustment > 0 else "<" * bar_length
        
        if n_trees == 1:
            print(f"  Tree {n_trees}: spam={spam_score:.4f}")
        else:
            print(f"  Tree {n_trees}: {prev_spam_score:.4f} -> {spam_score:.4f} ({adjustment:+.4f}) {bar}")
        
        prev_spam_score = spam_score
    
    xgb_final_proba = xgb_classifier.predict_proba(X_transformed_xgb)[0]
    xgb_final = 'spam' if xgb_final_proba[1] > 0.5 else 'ham'
    print(f"\nFINAL: {xgb_final.upper()} | ham={xgb_final_proba[0]:.3f}, spam={xgb_final_proba[1]:.3f}")
    
    print("\n" + "-" * 70)
    print("LOGISTIC REGRESSION")
    print("-" * 70)
    
    lr_vectorizer = lr_model.named_steps['vectorizer']
    lr_classifier = lr_model.named_steps['classifier']
    X_transformed_lr = lr_vectorizer.transform([cleaned_message])
    
    feature_names = lr_vectorizer.get_feature_names_out()
    coefficients = lr_classifier.coef_[0]
    
    words_in_message = cleaned_message.split()
    print(f"\nCleaned words in message: {words_in_message}")
    print(f"Total word count: {len(words_in_message)}")
    
    word_indices = X_transformed_lr.nonzero()[1]
    recognized_words = [feature_names[idx] for idx in word_indices]
    
    print(f"\nWords RECOGNIZED by model: {recognized_words}")
    print(f"Recognized count: {len(recognized_words)}")
    
    print(f"\nAll words RECOGNIZED and their contributions:")
    if len(word_indices) > 0:
        word_contributions = []
        for idx in word_indices:
            word = feature_names[idx]
            coef = coefficients[idx]
            word_contributions.append((word, coef))
        
        word_contributions.sort(key=lambda x: x[1], reverse=True)
        
        total_contribution = 0
        for word, coef in word_contributions:
            direction = "SPAM" if coef > 0 else "HAM"
            bar_length = int(abs(coef) * 20)
            bar = "█" * bar_length
            print(f"  '{word:15s}' {coef:+.4f} -> {direction:4s} {bar}")
            total_contribution += coef
        
        intercept = lr_classifier.intercept_[0]
        decision_score = total_contribution + intercept
        
        print(f"\nCalculation:")
        print(f"  Sum of word contributions: {total_contribution:+.4f}")
        print(f"  Intercept (baseline bias):  {intercept:+.4f}")
        print(f"  {'='*40}")
        print(f"  Raw decision score:         {decision_score:+.4f}")
        print(f"  Threshold:                   0.0000")
        print(f"  Decision: {'SPAM' if decision_score > 0 else 'HAM'} (score {'>' if decision_score > 0 else '<'} 0)")
    else:
        print("  No recognized words in message")
        decision_score = lr_classifier.intercept_[0]
        print(f"\nCalculation:")
        print(f"  Only intercept: {decision_score:+.4f}")
    
    lr_final = lr_model.predict([cleaned_message])[0]
    lr_proba = lr_model.predict_proba([cleaned_message])[0]
    
    print(f"\nFINAL: {str(lr_final).upper()} | ham={lr_proba[0]:.3f}, spam={lr_proba[1]:.3f}")

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)

for message in test_messages:
    cleaned_message = clean_text(message)
    
    rf_pred = rf_model.predict([cleaned_message])[0]
    rf_conf = rf_model.predict_proba([cleaned_message])[0][1]
    
    xgb_proba = xgb_classifier.predict_proba(xgb_model.named_steps['vectorizer'].transform([cleaned_message]))[0]
    xgb_pred = 'spam' if xgb_proba[1] > 0.5 else 'ham'
    xgb_conf = xgb_proba[1]
    
    lr_pred = lr_model.predict([cleaned_message])[0]
    lr_conf = lr_model.predict_proba([cleaned_message])[0][1]
    
    print(f"\n{message[:50]}")
    print(f"  RF:  {str(rf_pred).upper():4s} ({rf_conf:.2f})")
    print(f"  XGB: {xgb_pred.upper():4s} ({xgb_conf:.2f})")
    print(f"  LR:  {str(lr_pred).upper():4s} ({lr_conf:.2f})")


import joblib

# Save the models
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

joblib.dump(rf_model, 'rf_model.pkl')
print("✓ Random Forest saved as 'rf_model.pkl'")

joblib.dump(xgb_model, 'xgb_model.pkl')
print("✓ XGBoost saved as 'xgb_model.pkl'")

joblib.dump(lr_model, 'lr_model.pkl')
print("✓ Logistic Regression saved as 'lr_model.pkl'")

print("\nAll models saved successfully!")