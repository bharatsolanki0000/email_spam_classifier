# Email Spam Classifier

🌐 **Try it live:** https://emailspamclassifierbybharatsolanki.streamlit.app/

🚀 **A fast, browser-based spam detector powered by scikit-learn and Streamlit.**

This project classifies freeform email text as **Spam** or **Not Spam** using a pre-trained pipeline that includes text preprocessing, TF-IDF feature extraction, and a Random Forest classifier.

---

## ✨ Highlights

- ✅ **Instant predictions** in a nice Streamlit UI
- 🧠 Custom preprocessing pipeline (tokenization, stopword removal, stemming)
- 📊 Always shows prediction probabilities (spam vs not spam)
- 🧪 Includes sample phrases so you can test in a click
- 👤 Branded with the author name to make it clear who built it

---

## 🧰 Quick Start

1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the browser link shown by Streamlit (typically `http://localhost:8501`).

---

## 🗂 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI + model prediction logic |
| `preprocess.py` | Text preprocessing (tokenize, stopwords, stem) used by the pipeline |
| `pipeline.pkl` | Pretrained scikit-learn pipeline (processed text → TF-IDF → RandomForest) |
| `requirements.txt` | Python dependencies needed to run the app |

---

## 🛠 How It Works (Under the Hood)

1. **User inputs email text** into the Streamlit UI.
2. `preprocess.py` cleans the text:
   - lowercases
   - tokenizes (regex-based)
   - removes stopwords and punctuation
   - stems words with PorterStemmer
3. Processed text is passed through the sklearn pipeline in `pipeline.pkl`.
4. The model returns a **spam/not-spam label** and a **probability score**.

---

## 🧩 Customize / Extend

Want to retrain the model or swap in a different classifier?
- Replace `pipeline.pkl` with your own sklearn pipeline file.
- Keep `preprocess.py` in sync with how you preprocess training data.

---

## 🙋‍♂️ Author

**Bharat Solanki** — this project is built, owned, and maintained by me.
