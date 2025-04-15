# Common function to train and evaluate a model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name="Model"):
    print(f"🔍 {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("-" * 60)