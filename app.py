from preprocess import preprocess_transform
import streamlit as st
import pickle



preprocess_transform
   


# load pipeline
model = pickle.load(open("pipeline.pkl", "rb"))

st.title("📧 Email Spam Classifier")

st.write("Enter an email message to check whether it is Spam or Not Spam.")

# user input
message = st.text_area("Enter Email Text")

if st.button("Predict"):
    
    prediction = model.predict([message])[0]

    if prediction == 1:
        st.error("🚨 This is Spam")
    else:
        st.success("✅ This is Not Spam")