import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc(models, X_test, y_test):
    plt.figure(figsize=(8,6))

    for name, model in models.items():
        probs = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=name)

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()
