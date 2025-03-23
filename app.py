import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the XGBoost model
def load_model(model_path="xgb_model.pkl"):
    """
    Load the trained XGBoost model from pickle file
    """
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file not found! Please check the file path.")
        return None

# Define feature names used in training
FEATURE_NAMES = [
    "Payment_of_Min_Amount",
    "Credit_Mix",
    "Outstanding_Debt"
]

# Define encodings used in training
PAYMENT_MAPPING = {
    "No": 0,
    "Yes": 1
}

CREDIT_MIX_MAPPING = {
    "Unknown": 0,
    "Bad": 1,
    "Standard": 2,
    "Good": 3
}

# Define credit score categories
CREDIT_CATEGORIES = {
    0: "Poor",
    1: "Standard",
    2: "Good"
}

def get_category_color(category):
    """
    Returns appropriate color for each category
    """
    colors = {
        "Poor": "red",
        "Standard": "orange",
        "Good": "green"
    }
    return colors.get(category, "blue")

def main():
    # Page configuration
    st.set_page_config(
        page_title="Credit Score Prediction",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Main title and description
    st.title("📊 Credit Score Prediction with XGBoost")
    
    # Add feature encodings section as shown in the image
    st.markdown("## Feature Encodings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Payment Mapping:**")
        st.write("No: 0")
        st.write("Yes: 1")
    
    with col2:
        st.markdown("**Credit Mix Mapping:**")
        st.write("Unknown: 0")
        st.write("Bad: 1")
        st.write("Standard: 2")
        st.write("Good: 3")
    
    with col3:
        st.markdown("**Credit Score Mapping:**")
        st.write("Poor: 0")
        st.write("Standard: 1")
        st.write("Good: 2")
    
    st.markdown("---")
    
    st.write("Please enter the following information to predict your credit score:")

    # Create input form
    with st.form("prediction_form"):
        # Input fields
        user_input = {}
        for feature in FEATURE_NAMES:
            user_input[feature] = st.number_input(
                f"Enter {feature.replace('_', ' ')}",
                min_value=0,
                help=f"Enter value for {feature.replace('_', ' ')}"
            )

        # Submit button
        submit_button = st.form_submit_button("Predict Credit Score")

    # Make prediction when form is submitted
    if submit_button:
        # Load model
        model = load_model()

        if model:
            # Prepare input data
            input_data = np.array([[user_input[f] for f in FEATURE_NAMES]])

            try:
                # Make prediction
                prediction = model.predict(input_data)[0]

                # Get category
                category = CREDIT_CATEGORIES[prediction]
                category_color = get_category_color(category)

                # Display result with styling
                st.markdown(f"### Prediction Result")
                st.markdown(f"<h2 style='color: {category_color};'>Credit Score Category: {category}</h2>",
                          unsafe_allow_html=True)

                # Additional context based on category
                if category == "Good":
                    st.balloons()
                    st.write("🌟 Excellent! You have a good credit score.")
                elif category == "Standard":
                    st.write("✨ Your credit score is standard. There's room for improvement!")
                else:
                    st.write("💡 Consider taking steps to improve your credit score.")

                # Display feature importance info
                st.markdown("### Your Input Summary")
                for feature, value in user_input.items():
                    st.write(f"**{feature.replace('_', ' ')}:** {value}")

            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# None (this script only reads from 'xgb_model.pkl')