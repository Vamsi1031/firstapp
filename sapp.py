import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


st.title("📈 Decision Tree Classifier on Uploaded Dataset")

# 1. Upload File
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    # 2. Read File
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        df=df.head(10)
    else:
        df = pd.read_excel(uploaded_file)
        df=df.head(10)

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # 3. Select target column
    target = st.selectbox("Select the target column (what to predict):", df.columns)

    if st.button("Train Decision Tree"):
        try:
            X = df.drop(columns=[target])
            y = df[target]

            # Handle non-numeric data (optional)
            X = pd.get_dummies(X)

            # 4. Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 5. Train model
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

            # 6. Predict & show results
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"Model trained successfully! ✅ Accuracy: {acc:.2f}")

            st.subheader("📊 Predictions on Test Set")
            result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.dataframe(result_df.reset_index(drop=True))

            # ✅ 7. Visualize Decision Tree
            st.subheader("🌳 Decision Tree Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in model.classes_], filled=True, ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
