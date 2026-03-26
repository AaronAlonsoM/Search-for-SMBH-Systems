import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# ==========================================
# 1. STRICT PLOT STYLE (Matched to your paper)
# ==========================================
def set_plot_style():
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.grid'] = False
    plt.rcParams['font.size'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['xtick.labelsize'] = 24  
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['xtick.major.width'] = 2.0
    plt.rcParams['ytick.major.width'] = 2.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['ytick.major.size'] = 8.0

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_prep_data(filepath='data/ML_training_set.csv'):
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop any rows where feature extraction might have failed or yielded infinity
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Usable samples after cleaning: {len(df)}")
    
    # Define features (X) and labels (y)
    # Exclude metadata columns ('label', 'injected_alpha', 'base_mask')
    exclude_cols = ['label', 'injected_alpha', 'base_mask']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['label']
    
    return X, y, feature_cols

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================
def plot_learning_curve(estimator, X, y):
    """Plots the learning curve to prove dataset size is sufficient."""
    print("Calculating learning curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(train_sizes, train_mean, 'o-', color="tab:red", label="Training score", linewidth=3)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="tab:red")
    
    ax.plot(train_sizes, test_mean, 'o-', color="tab:blue", label="Cross-validation score", linewidth=3)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="tab:blue")
    
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.set_xlim(3900,40100)
    
    plt.tight_layout()
    plt.savefig('ML_fig_1_learning_curve.pdf', bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names):
    """Plots the Gini importance of each feature."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Homogenize and clean feature names
    sorted_features = []
    for i in indices:
        clean_name = feature_names[i].replace('_', ' ').capitalize()
        for acronym in ['Ftp', 'Lsp', 'Fwhm', 'Snr']:
            clean_name = clean_name.replace(acronym, acronym.upper())
        sorted_features.append(clean_name)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(x=importances[indices], y=sorted_features, color="tab:blue", ax=ax)
    
    ax.set_xlabel("Relative Importance (Gini)")
    ax.set_ylabel("Feature")
    
    plt.tight_layout()
    plt.savefig('ML_fig_2_feature_importances.pdf', bbox_inches='tight')
    plt.close()

def plot_performance(y_test, y_pred, y_prob):
    """Plots Confusion Matrix and ROC Curve side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Noise', 'QPO'], yticklabels=['Noise', 'QPO'], 
                ax=axes[0], annot_kws={"size": 28})
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title('Confusion Matrix')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    axes[1].plot(fpr, tpr, color='tab:red', lw=4, label=f'ROC curve (AUC = {auc_score:.3f})')
    axes[1].plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Receiver Operating Characteristic')
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('ML_fig_3_performance.pdf', bbox_inches='tight')
    plt.close()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    set_plot_style()
    
    # 1. Load Data
    X, y, feature_names = load_and_prep_data('data/ML_training_set.csv')
    
    # 2. Split into Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # 3. Initialize and Train Model (NOW WITH REGULARIZATION)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight='balanced')
    
    print("\nTraining Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # 4. Evaluate Model
    print("\nEvaluating on Test Set:")
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=['Noise (0)', 'QPO (1)']))
    print("-" * 50)
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # 5. Generate Paper Figures (PASSING THE REGULARIZED MODEL)
    print("\nGenerating Figures...")
    plot_learning_curve(RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1), X, y)
    plot_feature_importance(rf_model, feature_names)
    plot_performance(y_test, y_pred, y_prob)
    
    # 6. Save the fully trained model
    model_path = 'data/qpo_rf_model.joblib'
    joblib.dump(rf_model, model_path)
    print(f"\nModel saved successfully to {model_path}")

if __name__ == '__main__':
    main()
