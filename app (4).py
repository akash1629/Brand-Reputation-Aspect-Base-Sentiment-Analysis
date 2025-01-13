import streamlit as st
import pandas as pd
import altair as alt
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def get_sentiment_score(text, tokenizer, model):
    if not text.strip():
        return "neutral", 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).tolist()[0]
    idx = probs.index(max(probs))
    sentiments = ["negative","neutral","positive"]
    return sentiments[idx], float(probs[idx])

def extract_aspects(text, threshold=0.5):
    if not text.strip():
        return []
    aspect_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    aspects = ["Pricing","Design","Customer Service","Battery Life","Features","Performance"]
    result = aspect_classifier(text, aspects, multi_class=True)
    aspects_detected = []
    for label, score in zip(result["labels"], result["scores"]):
        if score >= threshold:
            aspects_detected.append(label)
    return aspects_detected

def analyze_text(text):
    # We re-initialize model here for a self-contained example (inefficient!)
    # In a real scenario, you'd do it once outside.
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    aspects = extract_aspects(text)
    sentiment_label, sentiment_conf = get_sentiment_score(text, tokenizer, model)
    return {
        "text": text,
        "aspects": aspects,
        "sentiment_label": sentiment_label,
        "sentiment_confidence": sentiment_conf
    }

st.title("Aspect-Based Sentiment Analysis Demo")

st.markdown("Zero-shot classification (BART) for aspects, RoBERTa for sentiment.")

st.subheader("Quick Test")
user_input = st.text_area("Enter text:")
if st.button("Analyze"):
    if user_input.strip():
        result = analyze_text(user_input)
        st.write("**Aspects**:", result["aspects"])
        st.write("**Sentiment**:", result["sentiment_label"],
                 f"(confidence={result['sentiment_confidence']:.2f})")
    else:
        st.warning("Please enter some text above.")

st.write("---")

st.subheader("Batch CSV Analysis")
csv_file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])
if csv_file:
    df = pd.read_csv(csv_file, encoding="latin-1", nrows=1000)
    if "text" not in df.columns:
        st.error("No 'text' column found in CSV!")
    else:
        keyword = st.text_input("Keyword filter (optional):")
        limit = st.number_input("Limit rows", value=20, min_value=1, max_value=500)
        if st.button("Process CSV"):
            if keyword.strip():
                df = df[df["text"].str.lower().str.contains(keyword.lower())]
            df = df.head(limit)

            results = []
            for _, row in df.iterrows():
                txt = str(row["text"])
                results.append(analyze_text(txt))
            result_df = pd.DataFrame(results)
            st.dataframe(result_df)

            all_aspects = []
            for asp_list in result_df["aspects"]:
                all_aspects.extend(asp_list)

            if all_aspects:
                import collections
                aspect_counts = collections.Counter(all_aspects)
                aspect_df = pd.DataFrame(aspect_counts.items(), columns=["aspect","count"])
                chart = alt.Chart(aspect_df).mark_bar().encode(
                    x="aspect",
                    y="count",
                    tooltip=["aspect","count"]
                )
                st.altair_chart(chart, use_container_width=True)

            sentiment_counts = result_df["sentiment_label"].value_counts()
            sentiment_df = pd.DataFrame({
                "sentiment": sentiment_counts.index,
                "count": sentiment_counts.values
            })
            chart2 = alt.Chart(sentiment_df).mark_bar().encode(
                x="sentiment",
                y="count",
                tooltip=["sentiment","count"]
            )
            st.altair_chart(chart2, use_container_width=True)
