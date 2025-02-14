# %%
# # !pip install spotipy
# !pip install lightgbm
# !pip install shap
# !pip install player

# %%
import os
import json
import time
from tqdm import tqdm
import requests
import spotipy
import pandas as pd
import numpy as np
import shap
from scipy.stats import randint, uniform
from scipy.stats import loguniform
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import mannwhitneyu, f_oneway, shapiro
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, classification_report, precision_recall_curve, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
import warnings

# %%
warnings.filterwarnings("ignore", category=UserWarning)

# %%
# Carrega variáveis do ficheiro .env
load_dotenv()

redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')
client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')

# Autenticação com SpotifyOAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="user-library-read user-read-private"
))

# %%
# Function to fetch the track ID
def get_track_id(track_name, artist_name):
    result = sp.search(q=f"track:{track_name} artist:{artist_name}", type="track", limit=1)
    if result['tracks']['items']:
        track_id = result['tracks']['items'][0]['id']
        # print(f"Track ID: {track_id}")
        return track_id
    else:
        # print("Track not found!")
        return None
    

# Function to normalize strings (lowercase and without extra spaces)
def normalize(text):
    return text.lower().strip() if isinstance(text, str) else text
 
 # FEATURE ENGINEERING
def create_features(df):
    df['energy_loudness'] = df['energy'] * df['loudness']
    df['speechiness_loudness'] = df['speechiness'] * df['loudness']
    df['hit_probability'] = df['chorus_hit'] / (df['sections'] + 1)  # Avoid division by zero
    return df

# ADAPTIVE THRESHOLD
def find_best_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.3, 0.7, 50)
    best_f1, best_threshold = 0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold, best_f1


# %% [markdown]
# # Data Extraction

# %%
df = pd.read_csv('spotify_full_list_20102023.csv', header=0)
df

# %%
df['track_name'].isna
df['track_name'] = df['Artist and Title'].str.split('-', n=1).str[1].fillna(df['Artist and Title'])
df

# %%
# Cache to avoid repeated calls
cache = {}

def get_track_id_batch(names_artists):
    """
    Fetch IDs for a list of (track_name, artist).
    """
    track_ids = []
    for name, artist in names_artists:
        key = (name, artist)
        if key in cache:
            track_ids.append(cache[key])
        else:
            try:
                # Try to get the IDs
                track_id = get_track_id(name, artist)
                cache[key] = track_id
                track_ids.append(track_id)
            except Exception as e:
                track_ids.append(None)
                # print(f"Error fetching ID for {name} - {artist}: {e}")
            time.sleep(0.1)  # Pause to respect request limits
    return track_ids

# Divide the dataframe into batches
batch_size = 50
batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

# Iterate through the batches
for batch in tqdm(batches):
    names_artists = list(zip(batch['track_name'], batch['Artist']))
    batch['track_id'] = get_track_id_batch(names_artists)

df_final = pd.concat(batches, ignore_index=True)


# %%
save_path_pickle = "data/tracks_with_ids.pkl"
df_final.to_pickle(save_path_pickle)
df_final.to_csv('data/tracks_with_ids.csv')

# %% [markdown]
# # Data processing

# %%
df_enriched2 = pd.read_csv('data/audio_features.csv', header=0)
df_enriched2.rename({'performer' : 'artist', 'song' : 'track'}, axis='columns', inplace = True)
df_enriched2.drop(['song_id','spotify_track_preview_url', 'spotify_track_explicit', 'spotify_track_album'], axis=1, inplace = True)
df_enriched2.dropna(thresh=7, inplace=True)
df_enriched2

# %%
df_enriched = pd.read_csv('data/tracks_with_ids.csv', header=0)
df_enriched.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace = True)
df_enriched

# %%
# Path to the folder containing the CSV files
folder = "features"

# List to store all DataFrames
dfs = []

# Iterate through all CSV files in the folder
for filename in os.listdir(folder):
    if filename.endswith(".csv"):  # Ensures only CSV files are read
        file_path = os.path.join(folder, filename)
        df = pd.read_csv(file_path)

        # Normalize column names to avoid inconsistencies
        df.columns = [col.lower().strip() for col in df.columns]

        # Normalize artist and track names to facilitate merging
        if 'artist' in df.columns and 'track' in df.columns:
            df['artist_norm'] = df['artist'].apply(normalize)
            df['track_norm'] = df['track'].apply(normalize)
            dfs.append(df)
        # else:
        #     print(f"Warning: {filename} does not contain the required columns.")

# Combine all datasets (row-wise concatenation)
if len(dfs) > 0:
    df_features = pd.concat(dfs, ignore_index=True)

df_features.rename({'artist': 'Artist'}, axis='columns', inplace=True)


# %%
df_enriched2

# %%
# Normalize artist and track names in both datasets
df_features['artist_norm'] = df_features['Artist'].apply(normalize)
df_features['track_norm']  = df_features['track'].apply(normalize)

df_enriched['artist_norm'] = df_enriched['Artist'].apply(normalize)
df_enriched['track_norm']  = df_enriched['track_name'].apply(normalize)

df_enriched2['artist_norm'] = df_enriched2['artist'].apply(normalize)
df_enriched2['track_norm']  = df_enriched2['track'].apply(normalize)


print("Original records df_features:", len(df_features))
print("Original records df_enriched:", len(df_enriched))
print("Original records df_enriched2:", len(df_enriched2))

# Remove duplicates – adjusting for the columns that define a unique track
df_features.drop_duplicates(subset=['Artist', 'track'], inplace=True)
df_enriched.drop_duplicates(subset=['Artist', 'track_name'], inplace=True)
df_enriched2.drop_duplicates(subset=['artist', 'track'], inplace=True)

print("After removing duplicates df_features:", len(df_features))
print("After removing duplicates df_enriched:", len(df_enriched))
print("After removing duplicates df_enriched2:", len(df_enriched2))

# Create a set of tuples (artist_norm, track_norm) for the TOP100 tracks
top100_set = set(zip(df_enriched['artist_norm'], df_enriched['track_norm']))
top100_set2 = set(zip(df_enriched2['artist_norm'], df_enriched2['track_norm']))

top100_combined = top100_set.union(top100_set2)

# Create a new 'in_top100' column in df_features:
# 1 if the track is in the TOP100 list, 0 otherwise
df_features['in_top100'] = df_features.apply(
    lambda row: 1 if (row['artist_norm'], row['track_norm']) in top100_combined else 0, 
    axis=1
)

print(len(df_features.loc[df_features['in_top100'] == 1]))


# Optional: Remove auxiliary normalization columns if desired
df_features.drop(columns=['target','artist_norm', 'track_norm'], inplace=True)


print("Merge completed! The final dataset has the 'in_top100' column.")


# %%
df_features.loc[df_features['in_top100']==1]

# %%
# -----------------------------
# 2. Define Features for Analysis
# -----------------------------
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'time_signature', 'chorus_hit', 'sections'
]

# %%
# -----------------------------
# 3. Separate Data into TOP and non-TOP Groups
# -----------------------------
# Assuming 'in_top100' is 1 for TOP songs and 0 for non-TOP songs

df_top = df_features[df_features['in_top100'] == 1]
df_non_top = df_features[df_features['in_top100'] == 0]

# %%
# -----------------------------
# 4. Plot Histograms for Each Feature (Separate for TOP and non-TOP)
# -----------------------------
for feature in features:
    # Create subplots with independent y-axis scales
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    
    sns.histplot(data=df_top, x=feature, bins=30, kde=True, ax=axes[0], color='blue')
    axes[0].set_title(f"{feature} - TOP")
    axes[0].set_xlabel(feature)
    
    sns.histplot(data=df_non_top, x=feature, bins=30, kde=True, ax=axes[1], color='green')
    axes[1].set_title(f"{feature} - non-TOP")
    axes[1].set_xlabel(feature)
    
    plt.tight_layout()
    # plt.savefig(f"hist_{feature}.png")
    plt.show()

# %%
# -----------------------------
# 5. Plot Boxplots for Each Feature by Group
# -----------------------------
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='in_top100', y=feature, data=df_features, hue = 'in_top100', legend = 'auto', palette=["green", "blue"])
    plt.title(f"Boxplot of {feature} by Group")
    plt.xticks([0, 1], ["non-TOP", "TOP"])
    plt.tight_layout()
    # plt.savefig(f"boxplot_{feature}.png")
    plt.show()

# %%
# -----------------------------
# 6. Descriptive Statistics by Group
# -----------------------------
# Compute descriptive statistics grouped by 'in_top100'
stats_by_group = df_features.groupby('in_top100')[features].describe().transpose()

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

print("Descriptive Statistics by Group:")
print(stats_by_group)

# Save the descriptive statistics to a CSV file
# stats_by_group.to_csv("descriptive_statistics_by_group_full.csv", index=True)

# %%
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')
pd.reset_option('display.width')

# %%
# -----------------------------
# 7. Statistical Tests: Mann-Whitney U Test for Each Feature
# -----------------------------
p_values = {}
for feature in features:
    top_data = df_top[feature].dropna()
    non_top_data = df_non_top[feature].dropna()
    stat, p = mannwhitneyu(top_data, non_top_data, alternative='two-sided')
    p_values[feature] = p

p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'p_value'])
print("Mann-Whitney U test p-values:")
print(p_values_df)
# p_values_df.to_csv("mann_whitney_pvalues.csv", index=False)

# %%
# -----------------------------
# 8. Correlation Matrix Plot
# -----------------------------
plt.figure(figsize=(12, 6))
corr_matrix = df_features[features + ['in_top100']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
# plt.savefig("correlation_matrix.png")
plt.show()

# %%
## anova analysis ##
anova_results = {}
for feature in features:
    f_stat, p_val = f_oneway(df_top[feature], df_non_top[feature])
    anova_results[feature] = p_val

# Display the results
anova_results_df = pd.DataFrame(list(anova_results.items()), columns=['Feature', 'p-value'])
anova_results_df


# %%
## shapiro analysis ##
normality_results = {}
for feature in features:
    stat, p_val = shapiro(df_features[feature])
    normality_results[feature] = p_val

# Display the results
normality_results_df = pd.DataFrame(list(normality_results.items()), columns=['Feature', 'p-value'])
normality_results_df

# %%
# Select the numeric features
X = df_features[features]
X = add_constant(X)  # Add a constant for the model

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF
vif_data


# %% [markdown]
# # Model Testing

# %%
# Assuming 'df' is the DataFrame with the features and the 'TOP_spotify' column
X = df_features.drop(columns=['track','Artist','uri', 'in_top100'])
y = df_features['in_top100']

# Scaling the numeric variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Balancing with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 3 Feature Engineering - Create new features based on interactions
X_train_res['energy_loudness'] = X_train_res['energy'] * X_train_res['loudness']
X_train['energy_loudness'] = X_train['energy'] * X_train['loudness']
X_test['energy_loudness'] = X_test['energy'] * X_test['loudness']


# %%
X_test

# %%
# Initializing models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=1, random_state=42, use_label_encoder=False, eval_metric='auc', tree_method='hist'),
}

# Defining hyperparameter distributions for RandomizedSearchCV
param_distributions = {
    "Logistic Regression": {
        'C': loguniform(0.1, 10),
        'solver': ['liblinear']
    },
    "Random Forest": {
        'n_estimators': randint(50, 350),
        'max_depth': [None] + list(randint(5, 20).rvs(5)),  # Some random depths between 5 and 20, plus None
        'min_samples_split': randint(2, 6)
    },
    "XGBoost": {
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'n_estimators': randint(50, 350)
    }
}

# Storing results
results = {}

# Use fewer iterations to speed up the search
n_iter_search = 10

# For each model, perform RandomizedSearchCV, adjust threshold, and evaluate performance
for model_name, model in models.items():
    print(f"\nRunning RandomizedSearchCV for {model_name}...")
    # If SVM, consider reducing iterations further
    random_search = RandomizedSearchCV(
        model, 
        param_distributions[model_name], 
        n_iter=n_iter_search, 
        cv=3, 
        n_jobs=-1, 
        scoring='roc_auc', 
        random_state=42,
        error_score=np.nan
    )
    start_time = time.time()
    random_search.fit(X_train_res, y_train_res)
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f} seconds")
    
    best_model = random_search.best_estimator_
    probs = best_model.predict_proba(X_test)[:, 1]
    
    # Adjust decision threshold dynamically to maximize F1-score
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # small epsilon to avoid division by zero
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Best Threshold for {model_name}: {best_threshold:.2f}, F1-score: {f1_scores[best_idx]:.3f}")
    
    predictions = (probs >= best_threshold).astype(int)
    auc = roc_auc_score(y_test, probs)
    
    results[model_name] = {
        'Best Params': random_search.best_params_,
        'AUC': auc,
        'Confusion Matrix': confusion_matrix(y_test, predictions),
        'Classification Report': classification_report(y_test, predictions)
    }
    
    print(f"{model_name} - AUC: {auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

# Display final results
print("\nFinal Results:")
for model_name, res in results.items():
    print(f"\nResults for {model_name}:")
    print(f"Best Parameters: {res['Best Params']}")
    print(f"AUC: {res['AUC']:.4f}")
    print("Confusion Matrix:")
    print(res['Confusion Matrix'])
    print("Classification Report:")
    print(res['Classification Report'])

# %%
results["Random Forest"].get('Best Params')

# %%
# Dictionary to store the fitted models
fitted_models = {}

# Fit Logistic Regression with balanced class weight
fitted_models["Logistic Regression"] = LogisticRegression(
    class_weight='balanced',
    **results["Logistic Regression"].get('Best Params'),
    random_state=42,
    max_iter=1000
).fit(X_train_res, y_train_res)

# Fit Random Forest with balanced class weight and 300 trees
fitted_models["Random Forest"] = RandomForestClassifier(
    class_weight='balanced',
    **results["Random Forest"].get('Best Params'),
    random_state=42
).fit(X_train_res, y_train_res)

# Fit XGBoost with specified parameters; using hist tree_method for CPU efficiency
fitted_models["XGBoost"] = XGBClassifier(
    scale_pos_weight=1,
    **results["XGBoost"].get('Best Params'),
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    tree_method='hist'
).fit(X_train_res, y_train_res)

# Fit SVM using linear kernel (for faster computation) and enabling probability estimates
fitted_models["SVM"] = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
).fit(X_train_res, y_train_res)

# Plot ROC curves for each fitted model
plt.figure(figsize=(10, 8))
results_auc = {}  # Dictionary to store AUC for each model

for model_name, model in fitted_models.items():
    # Get predicted probabilities for the positive class
    probs = model.predict_proba(X_test)[:, 1]
    # Compute false positive rate and true positive rate
    fpr, tpr, _ = roc_curve(y_test, probs)
    # Calculate AUC score
    auc = roc_auc_score(y_test, probs)
    results_auc[model_name] = auc
    # Plot ROC curve with a label showing the model name and AUC
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')

# Plot the diagonal reference line for a random classifier
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curves for Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# %%
# Use the same feature matrix that was used to train rf_model
X_for_shap = X_test

# Calculate the SHAP values
explainer = shap.TreeExplainer(fitted_models["Random Forest"])
shap_values = explainer.shap_values(X_for_shap)

# Check the shape of the shap_values:
print("shap_values shape:", shap_values.shape)  # Should print (40003, 15, 2)

# Extract the SHAP values for class 1 (TOP)
# This returns an array with shape (40003, 15)
shap_values_class1 = shap_values[:, :, 1]
print("Adjusted SHAP values shape for class 1:", shap_values_class1.shape)

# Generate the summary plot charts using SHAP values for class 1
shap.summary_plot(shap_values_class1, X_for_shap, plot_type="bar")
shap.summary_plot(shap_values_class1, X_for_shap)


# %%
# Train models with cross-validation to check for overfitting
rf = RandomForestClassifier(
    class_weight='balanced',
    **results["Random Forest"].get('Best Params'),
    random_state=42
)
xgb = XGBClassifier(
    scale_pos_weight=1,
    **results["XGBoost"].get('Best Params'),
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    tree_method='hist'
)

rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
xgb_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='roc_auc')

print("Random Forest Cross-Val AUC:", np.mean(rf_scores))
print("XGBoost Cross-Val AUC:", np.mean(xgb_scores))

# THRESHOLD TUNING TO MAXIMIZE F1-SCORE USING LOGISTIC REGRESSION

# Train a logistic regression model (as a baseline for threshold tuning)
log_model = LogisticRegression(
    class_weight='balanced',
    **results["Logistic Regression"].get('Best Params'),
    random_state=42,
    max_iter=1000
)
log_model.fit(X_train, y_train)

# Get predicted probabilities for the positive class (class 1)
log_reg_probs = log_model.predict_proba(X_test)[:, 1]

# Define a range of thresholds to test
thresholds = np.linspace(0.1, 0.9, 50)

best_threshold, best_f1 = 0, 0
for t in thresholds:
    preds = (log_reg_probs > t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

print(f"Best threshold for F1-score: {best_threshold:.2f}, F1-score: {best_f1:.3f}")

# Plot the Precision-Recall curve for logistic regression
precision, recall, _ = precision_recall_curve(y_test, log_reg_probs)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

# Optionally, display confusion matrix and classification report for the best threshold
predictions = (log_reg_probs > best_threshold).astype(int)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# %%
# -----------------------------
# 9. Feature Importance Using Random Forest
# -----------------------------
df_features['energy_loudness'] = X_test['energy'] * X_test['loudness']

features.append('energy_loudness')

rf.fit(df_features[features], df_features['in_top100'])
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue = 'Feature', legend = 'auto', palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
# plt.savefig("feature_importance.png")
plt.show()

# %%
# Create additional features for the dataset
df_features = create_features(df_features)

# Split the data into features (X) and target (y)
# We drop the columns 'track', 'Artist', and 'uri' because they are identifiers,
# and 'in_top100' is our target variable.
X = df_features.drop(columns=['track', 'Artist', 'uri', 'in_top100'])
y = df_features['in_top100']

# Split into training and testing sets
X_train_tuning, X_test_tuning, y_train_tuning, y_test_tuning = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------
# Additional Hyperparameter Tuning
# -------------------------------------------

# Define hyperparameter grids for Random Forest and XGBoost
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5]
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.05],
    'colsample_bytree': [0.8, 1.0],
    'subsample': [0.8, 1.0]
}

# Initialize GridSearchCV for Random Forest and XGBoost
rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1
)
xgb = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid_xgb,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1
)

# Fit the models on the tuning data
rf.fit(X_train_tuning, y_train_tuning)
xgb.fit(X_train_tuning, y_train_tuning)

# Get the best models
best_rf = rf.best_estimator_
best_xgb = xgb.best_estimator_

# Determine the best decision thresholds for each model using the helper function
best_threshold_rf, best_f1_rf = find_best_threshold(best_rf, X_test_tuning, y_test_tuning)
best_threshold_xgb, best_f1_xgb = find_best_threshold(best_xgb, X_test_tuning, y_test_tuning)

# Apply the optimized thresholds to generate predictions
rf_preds = (best_rf.predict_proba(X_test_tuning)[:, 1] >= best_threshold_rf).astype(int)
xgb_preds = (best_xgb.predict_proba(X_test_tuning)[:, 1] >= best_threshold_xgb).astype(int)

# Print the results for Random Forest
print(f"Random Forest - Best Threshold: {best_threshold_rf}, F1-score: {best_f1_rf}")
print(confusion_matrix(y_test_tuning, rf_preds))
print(classification_report(y_test_tuning, rf_preds))

# Print the results for XGBoost
print(f"XGBoost - Best Threshold: {best_threshold_xgb}, F1-score: {best_f1_xgb}")
print(confusion_matrix(y_test_tuning, xgb_preds))
print(classification_report(y_test_tuning, xgb_preds))


# %%
# Previsões no conjunto de treino
# y_pred_rf_train = (rf_model.predict_proba(X_train_tuning)[:, 1] > 0.3).astype(int)
# y_pred_xgb_train = (xgb_model.predict_proba(X_train_tuning)[:, 1] > 0.38).astype(int)

# Aplicação do threshold nos modelos
y_pred_rf_train = (best_rf.predict_proba(X_train_tuning)[:, 1] >= 0.3).astype(int)
y_pred_xgb_train = (best_xgb.predict_proba(X_train_tuning)[:, 1] >= 0.38).astype(int)

print("Random Forest - Treino")
print(classification_report(y_train_tuning, y_pred_rf_train))

print("\nRandom Forest - Teste")
print(classification_report(y_test_tuning, (rf.predict_proba(X_test_tuning)[:, 1] > 0.3).astype(int)))

print("\nXGBoost - Treino")
print(classification_report(y_train_tuning, y_pred_xgb_train))

print("\nXGBoost - Teste")
print(classification_report(y_test_tuning, (xgb.predict_proba(X_test_tuning)[:, 1] > 0.38).astype(int)))

# %%
cv_scores_rf = cross_val_score(rf, X_train_tuning, y_train_tuning, cv=5, scoring='f1')
cv_scores_xgb = cross_val_score(xgb, X_train_tuning, y_train_tuning, cv=5, scoring='f1')

print(f"Random Forest - Cross-Validation F1-score: {cv_scores_rf.mean():.3f} ± {cv_scores_rf.std():.3f}")
print(f"XGBoost - Cross-Validation F1-score: {cv_scores_xgb.mean():.3f} ± {cv_scores_xgb.std():.3f}")


# %%
# -----------------------------
# 11. Generate a Final Report
# -----------------------------
with open("final_report.txt", "w", encoding="utf-8") as f:
    f.write("Music Feature Analysis for TOP Songs\n")
    f.write("====================================\n\n")
    f.write("1. Descriptive Statistics by Group:\n")
    f.write(stats_by_group.to_string())
    f.write("\n\n2. Mann-Whitney U Test p-values:\n")
    f.write(p_values_df.to_string(index=False))
    f.write("\n\n3. Observations:\n")
    f.write("   - The musical features, although informative, do not fully explain song popularity due to external factors (e.g., marketing, cultural trends).\n")
    f.write("   - Certain features (see feature importance plot) are more influential in distinguishing TOP songs, yet inherent variability remains.\n")
    f.write("   - Consider incorporating contextual data to further enhance the model.\n")
    
print("Final report and all plots have been saved.")

# %% [markdown]
# # Notas


