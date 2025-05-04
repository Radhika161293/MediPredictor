import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

def preprocess_liver(df):
    df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})
    df["Gender"] = df["Gender"].map({'Male': 1, 'Female': 0})
    imputer = SimpleImputer(strategy='mean')
    X = df.drop("Dataset", axis=1)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y = df["Dataset"]
    return StandardScaler().fit_transform(X), y

def preprocess_kidney(df):
    df.drop("id", axis=1, inplace=True)
    df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})
    df = df.dropna()

    # Keep only relevant features matching Streamlit form
    selected_cols = ["age", "bp", "sg", "al", "su", "pcv", "wc", "rc", "htn", "dm", "appet", "ane"]
    df = df[selected_cols + ["classification"]]

    # Encode categorical
    for col in ["htn", "dm", "appet", "ane"]:
        df[col] = df[col].map({"yes": 1, "no": 0, "good": 1, "poor": 0})

    X = df.drop("classification", axis=1)
    y = df["classification"]
    return StandardScaler().fit_transform(X), y

def preprocess_parkinsons(df):
    df = df.drop("name", axis=1)
    X = df.drop("status", axis=1)
    y = df["status"]
    return StandardScaler().fit_transform(X), y

def train_and_save_models(X, y, disease):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"{disease}_{name}.pkl")
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{disease} - {name} Accuracy:", round(acc, 4))

if __name__ == "__main__":
    liver = pd.read_csv("liver.csv")
    kidney = pd.read_csv("kidney.csv")
    parkinsons = pd.read_csv("parkinsons.csv")
    
    X, y = preprocess_liver(liver)
    train_and_save_models(X, y, "liver")

    X, y = preprocess_kidney(kidney)
    train_and_save_models(X, y, "kidney")

    X, y = preprocess_parkinsons(parkinsons)
    train_and_save_models(X, y, "parkinsons")
