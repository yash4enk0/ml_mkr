import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models (cache them so they're only loaded once)
@st.cache_resource
def load_models():
    rf_model = joblib.load('rf_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    lr_model = joblib.load('lr_model.pkl')
    return rf_model, xgb_model, lr_model

# Load dataset
@st.cache_data
def load_dataset():
    data = pd.read_csv('enron_spam_data.csv')
    data['message'] = data['Subject'].fillna('') + " " + data['Message'].fillna('')
    return data

st.title("Spam Detection with Three Models")

# Load models and data
rf_model, xgb_model, lr_model = load_models()
data = load_dataset()

# Center the input and buttons
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    input_col, dice_col, submit_col = st.columns([5, 1, 1])
    with input_col:
        user_input = st.text_input("Enter a message", 
                                    label_visibility="collapsed", 
                                    placeholder="Type a message to check...",
                                    key="message_input")
    with dice_col:
        if st.button("🎲"):
            # Select random message from dataset
            random_message = data['message'].sample(1).iloc[0]
            st.session_state.message_input = random_message
            st.rerun()
    
    with submit_col:
        submit_button = st.button("▶️")

st.divider()

# Three columns for model results
if user_input and submit_button:
    # Left: Logistic Regression
    # Center: Random Forest  
    # Right: XGBoost
    lr_col, rf_col, xgb_col = st.columns(3)
    
    # LOGISTIC REGRESSION (Left)
    with lr_col:
        st.subheader("📊 Logistic Regression")
        lr_vectorizer = lr_model.named_steps['vectorizer']
        lr_classifier = lr_model.named_steps['classifier']
        X_transformed = lr_vectorizer.transform([user_input])
        
        feature_names = lr_vectorizer.get_feature_names_out()
        coefficients = lr_classifier.coef_[0]
        word_indices = X_transformed.nonzero()[1]
        
        lr_pred = lr_model.predict([user_input])[0]
        lr_proba = lr_model.predict_proba([user_input])[0]
        lr_pred_str = str(lr_pred).upper() if isinstance(lr_pred, str) else ('SPAM' if lr_pred == 1 else 'HAM')
        
        if len(word_indices) > 0:
            word_contributions = []
            total_contribution = 0
            for idx in word_indices:
                word = feature_names[idx]
                coef = coefficients[idx]
                word_contributions.append((word, coef))
                total_contribution += coef
            
            word_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get top 5 contributions
            top_words = word_contributions[:5]
            
            intercept = lr_classifier.intercept_[0]
            decision_score = total_contribution + intercept
            
            # Build LaTeX visualization
            latex_lines = []
            latex_lines.append(r"\begin{array}{rcl}")
            
            for i, (word, coef) in enumerate(top_words):
                if i < len(top_words) - 1:
                    latex_lines.append(f'\\text{{"{word}"}} & \\rightarrow & {coef:+.2f} \\\\')
                else:
                    # Last line with big brace
                    latex_lines.append(f'\\text{{"{word}"}} & \\rightarrow & {coef:+.2f}')
            
            latex_lines.append(r"\end{array}")
            latex_lines.append(f"\\Bigg\\}} \\text{{{lr_pred_str}}} ({lr_proba[1]:.2f})")
            
            latex_str = " ".join(latex_lines)
            st.latex(latex_str)
        else:
            st.write("No recognized words")
            st.latex(f"\\text{{No features}} \\rightarrow \\text{{{lr_pred_str}}} ({lr_proba[1]:.2f})")
    
    # RANDOM FOREST (Center)
    with rf_col:
        st.subheader("🌲 Random Forest")
        rf_pred = rf_model.predict([user_input])[0]
        rf_proba = rf_model.predict_proba([user_input])[0]
        
        rf_vectorizer = rf_model.named_steps['vectorizer']
        rf_classifier = rf_model.named_steps['classifier']
        X_transformed = rf_vectorizer.transform([user_input])
        
        rf_votes = []
        for i, tree in enumerate(rf_classifier.estimators_):
            tree_pred = tree.predict(X_transformed)[0]
            rf_votes.append(tree_pred)
        
        # Count spam votes
        spam_votes = sum([1 if (v == 'spam' or v == 1) else 0 for v in rf_votes])
        rf_pred_str = str(rf_pred).upper() if isinstance(rf_pred, str) else ('SPAM' if rf_pred == 1 else 'HAM')
        
        # Build LaTeX visualization
        latex_lines = [r"\begin{array}{rcl}"]
        
        for i, vote in enumerate(rf_votes):
            vote_val = 1 if (vote == 'spam' or vote == 1) else 0
            if i < len(rf_votes) - 1:
                latex_lines.append(f"\\text{{Tree{i+1}}} & \\rightarrow & {vote_val} \\\\")
            else:
                latex_lines.append(f"\\text{{Tree{i+1}}} & \\rightarrow & {vote_val}")
        
        latex_lines.append(r"\end{array}")
        latex_lines.append(f"\\Bigg\\}} \\text{{{rf_pred_str}}} ({spam_votes}/{len(rf_votes)})")
        
        latex_str = " ".join(latex_lines)
        st.latex(latex_str)
    
    # XGBOOST (Right)
    with xgb_col:
        st.subheader("⚡ XGBoost")
        xgb_vectorizer = xgb_model.named_steps['vectorizer']
        xgb_classifier = xgb_model.named_steps['classifier']
        X_transformed = xgb_vectorizer.transform([user_input])
        
        # Get sequential predictions
        tree_scores = []
        for n in range(1, 4):
            proba = xgb_classifier.predict_proba(X_transformed, iteration_range=(0, n))[0]
            spam_score = proba[1]
            tree_scores.append(spam_score)
        
        xgb_proba = xgb_classifier.predict_proba(X_transformed)[0]
        xgb_pred = 'SPAM' if xgb_proba[1] > 0.5 else 'HAM'
        
        # Build LaTeX visualization
        latex_parts = []
        for i, score in enumerate(tree_scores):
            pred_label = 'SPAM' if score > 0.5 else 'HAM'
            latex_parts.append(f"\\text{{Tree{i+1}}} \\rightarrow \\text{{{pred_label}}} ({score:.2f})")
            if i < len(tree_scores) - 1:
                latex_parts.append(r"\downarrow")
        
        # Final prediction
        latex_parts.append(r"\downarrow")
        latex_parts.append(f"\\text{{{xgb_pred}}} ({xgb_proba[1]:.2f})")
        
        # Join with line breaks
        latex_str = r" \\ ".join(latex_parts)
        st.latex(latex_str)

else:
    st.info("👆 Enter a message or click 🎲 for a random message, then click ▶️ to classify!")