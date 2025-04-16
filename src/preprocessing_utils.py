# preprocessing_utils.py

def clean_categories(X):
    return X.applymap(lambda x: x.lower().replace(" ", "_") if isinstance(x, str) else x)