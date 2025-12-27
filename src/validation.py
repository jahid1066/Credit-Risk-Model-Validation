from sklearn.model_selection import StratifiedKFold, cross_val_score

def cross_validate(models, X, y):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=cv, scoring='roc_auc'
        )
        results[name] = scores.mean()

    return results
