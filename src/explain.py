"""SHAP explainability for tree-based models."""


def tree_shap_values(model, X):
    import shap

    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)
