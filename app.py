from preprocess import preprocess_transform
import __main__ as main
import streamlit as st
import pickle

# ensure unpickling works regardless of whether this file is run as a script or imported
main.preprocess_transform = preprocess_transform

# load pipeline
model = pickle.load(open("pipeline.pkl", "rb"))

st.set_page_config(
    page_title=" Email Spam Classifier — Bharat Solanki",
    layout="centered",
)

st.title("📧 Email Spam Classifier")

st.markdown("**Built by Bharat Solanki**")

st.markdown(
    """ 
    Enter an email snippet and the model will predict whether it is **Spam** or **Not Spam**.

    - The model is a scikit-learn pipeline that applies text preprocessing + TF-IDF + Random Forest.
    - Use the sidebar to toggle extra details.
    """
)

# Sidebar controls
st.sidebar.header("Samples")

st.sidebar.markdown("---")
samples = {
    "Friendly reminder": "Hi there! Just checking in on the report I sent last week.",
    "Promotion": "You have been selected for a free trial! Click here to claim your reward.",
    "Account alert": "Your account has been accessed from a new device. If this wasn't you, reset your password now.",
}
sample_choice = st.sidebar.selectbox("Pick a sample", ["(none)", *samples.keys()])

st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Bharat Solanki**")

# Prediction form
with st.form(key="spam_form"):
    message = st.text_area("Enter Email Text", value=samples.get(sample_choice, ""))
    submit = st.form_submit_button("Predict")

if submit:
    if not message.strip():
        st.warning("Please enter some text to classify.")
    else:
        prediction = model.predict([message])[0]

        # show probabilities (always attempt to show if available)
        prob_text = ""
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba([message])[0]
                spam_prob = float(probs[1])
                not_spam_prob = float(probs[0])
                prob_text = f"Spam: {spam_prob:.1%} | Not Spam: {not_spam_prob:.1%}"
            except Exception:
                prob_text = "(Probability not available)"

        else:
            prob_text = "(Probability not available)"

        if prediction == 1:
            st.error(f"🚨 This is Spam – {prob_text}")
        else:
            st.success(f"✅ This is Not Spam – {prob_text}")
