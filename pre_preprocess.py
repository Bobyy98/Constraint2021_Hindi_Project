import pandas as pd

# ---------- Helper: map label to binary ----------
def map_label(label):
    if pd.isna(label):
        return None   # keep None for test set (no labels)
    return 1 if "fake" in str(label).lower() else 0

# ---------- Convert train ----------
train = pd.read_excel("Constraint_Hindi_Train.xlsx")
train_clean = pd.DataFrame({
    "id": train["Unique ID"],
    "text": train["Post"],
    "label": train["Labels Set"].apply(map_label)
})
train_clean.to_csv("train_clean.csv", index=False, encoding="utf-8")

# ---------- Convert validation ----------
val = pd.read_excel("Constraint_Hindi_Valid.xlsx")
val_clean = pd.DataFrame({
    "id": val["Unique ID"],
    "text": val["Post"],
    "label": val["Labels Set"].apply(map_label)
})
val_clean.to_csv("val_clean.csv", index=False, encoding="utf-8")

# ---------- Convert test ----------
test = pd.read_excel("Constraint_Hindi_Test.xlsx")
test_clean = pd.DataFrame({
    "id": test["Unique ID"],
    "text": test["Post"]
    # no labels in test
})
test_clean.to_csv("test_clean.csv", index=False, encoding="utf-8")

print("✅ Done: train_clean.csv, val_clean.csv, test_clean.csv created")
