import streamlit as st
import pandas as pd
import altair as alt
import torch
import time
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px

# App configuration
st.set_page_config(
    page_title="Advanced Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Cache models to avoid reloading
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

@st.cache_resource
def load_aspect_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Analysis functions
def get_sentiment_score(text, tokenizer, model):
    if not text or not isinstance(text, str) or not text.strip():
        return "neutral", 0.5
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).tolist()[0]
        idx = probs.index(max(probs))
        sentiments = ["negative", "neutral", "positive"]
        return sentiments[idx], float(probs[idx])
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return "neutral", 0.5

def extract_aspects(text, aspect_classifier, threshold=0.5, custom_aspects=None):
    if not text or not isinstance(text, str) or not text.strip():
        return []
    
    try:
        # Default aspects
        aspects = ["Pricing", "Design", "Customer Service", "Battery Life", "Features", "Performance"]
        
        # Use custom aspects if provided
        if custom_aspects and custom_aspects.strip():
            custom_list = [a.strip() for a in custom_aspects.split(",") if a.strip()]
            if custom_list:
                aspects = custom_list
        
        result = aspect_classifier(text, aspects, multi_class=True)
        
        aspects_detected = []
        for label, score in zip(result["labels"], result["scores"]):
            if score >= threshold:
                aspects_detected.append({"aspect": label, "confidence": score})
        
        return aspects_detected
    except Exception as e:
        st.error(f"Error in aspect extraction: {str(e)}")
        return []

def analyze_text_item(item, tokenizer, model, aspect_classifier, aspect_threshold, custom_aspects):
    """Analyze a single text item - for parallel processing"""
    text = item["text"] if isinstance(item, dict) else str(item)
    
    if not isinstance(text, str):
        text = str(text)
    
    aspects = extract_aspects(text, aspect_classifier, aspect_threshold, custom_aspects)
    sentiment_label, sentiment_conf = get_sentiment_score(text, tokenizer, model)
    
    result = {
        "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for display
        "full_text": text,
        "aspects": aspects,
        "sentiment_label": sentiment_label,
        "sentiment_confidence": sentiment_conf
    }
    
    # Add aspect-specific sentiment indicators
    aspect_sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    aspect_sentiments[sentiment_label] = len(aspects)
    
    for aspect_name in set(a["aspect"] for a in aspects):
        result[f"has_{aspect_name.lower().replace(' ', '_')}"] = True
    
    return result

def analyze_batch(texts, tokenizer, model, aspect_classifier, aspect_threshold=0.5, custom_aspects=None, max_workers=5):
    """Analyze a batch of texts with progress indication"""
    results = []
    total = len(texts)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    # Function to update progress
    def update_progress(future_done):
        nonlocal completed
        completed += 1
        progress_bar.progress(completed / total)
        elapsed = time.time() - start_time
        remaining = (total - completed) * (elapsed / completed) if completed > 0 else 0
        status_text.text(f"Processed {completed}/{total} items. Est. time remaining: {remaining:.1f}s")
    
    # Process in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in texts:
            future = executor.submit(
                analyze_text_item, 
                item, 
                tokenizer, 
                model, 
                aspect_classifier, 
                aspect_threshold,
                custom_aspects
            )
            future.add_done_callback(lambda f: update_progress(f))
            futures.append(future)
        
        # Collect results
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                st.error(f"Error processing item: {str(e)}")
    
    progress_bar.progress(1.0)
    status_text.text(f"Completed analysis of {total} items in {time.time() - start_time:.2f} seconds")
    
    return results

# UI Components
def render_sidebar():
    st.sidebar.title("Settings")
    
    with st.sidebar.expander("Model Settings", expanded=False):
        aspect_threshold = st.slider(
            "Aspect Detection Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5,
            step=0.05,
            help="Higher values require stronger evidence for aspect detection"
        )
        
        custom_aspects = st.text_area(
            "Custom Aspects (comma separated)",
            placeholder="Price, Quality, Service, etc.",
            help="Leave empty to use default aspects"
        )
        
        max_workers = st.slider(
            "Parallel Workers", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Higher values may speed up batch processing but use more resources"
        )
    
    with st.sidebar.expander("Visualization Settings", expanded=False):
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Pie Chart", "Scatter Plot"],
            index=0
        )
        
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Blues", "Reds", "Greens", "Viridis", "Plasma"],
            index=0
        )
    
    with st.sidebar.expander("CSV Import Settings", expanded=False):
        csv_encoding = st.selectbox(
            "CSV Encoding",
            ["utf-8", "latin-1", "iso-8859-1", "cp1252"],
            index=0
        )
        
        text_column = st.text_input(
            "Text Column Name", 
            value="text",
            help="Column name containing text to analyze"
        )
    
    return {
        "aspect_threshold": aspect_threshold,
        "custom_aspects": custom_aspects,
        "max_workers": max_workers,
        "chart_type": chart_type,
        "color_scheme": color_scheme,
        "csv_encoding": csv_encoding,
        "text_column": text_column
    }

def render_single_analysis_ui(tokenizer, model, aspect_classifier, settings):
    st.header("Single Text Analysis", divider="gray")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="The new phone has an excellent camera and impressive battery life, although it's a bit expensive."
        )
    
    with col2:
        st.write("Example Topics:")
        if st.button("Product Review"):
            user_input = "This laptop has amazing performance but the battery life is disappointing. Customer service was helpful though."
            st.session_state.user_input = user_input
        if st.button("Restaurant Review"):
            user_input = "The food was delicious with reasonable prices, but the service was slow and the ambiance could be improved."
            st.session_state.user_input = user_input
        if st.button("App Feedback"):
            user_input = "The user interface is intuitive but it crashes frequently. The developers respond quickly to support tickets."
            st.session_state.user_input = user_input
    
    if "user_input" in st.session_state:
        user_input = st.session_state.user_input
        
    analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)
    
    if analyze_button and user_input.strip():
        with st.spinner("Analyzing text..."):
            result = analyze_text_item(
                {"text": user_input}, 
                tokenizer, 
                model, 
                aspect_classifier, 
                settings["aspect_threshold"],
                settings["custom_aspects"]
            )
            
            display_single_result(result)

def display_single_result(result):
    st.subheader("Analysis Results", divider="gray")
    
    # Display sentiment with color coding
    sentiment_colors = {
        "positive": "green",
        "neutral": "blue",
        "negative": "red"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Sentiment
        <div style="padding: 10px; background-color: {sentiment_colors[result['sentiment_label']]}30; 
                    border-left: 5px solid {sentiment_colors[result['sentiment_label']]}; border-radius: 5px;">
            <h4 style="color: {sentiment_colors[result['sentiment_label']]};">{result['sentiment_label'].upper()}</h4>
            <p>Confidence: {result['sentiment_confidence']:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if result['aspects']:
            st.markdown("### Detected Aspects")
            for aspect in result['aspects']:
                st.markdown(f"""
                <div style="margin-bottom: 5px; padding: 5px 10px; background-color: #f0f2f6; border-radius: 3px;">
                    {aspect['aspect']} <span style="float: right; color: #666;">{aspect['confidence']:.2%}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific aspects were detected with the current threshold.")
    
    # Show a visual representation
    if result['aspects']:
        data = pd.DataFrame([{
            'Aspect': a['aspect'], 
            'Confidence': a['confidence']
        } for a in result['aspects']])
        
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('Aspect', sort='-y'),
            y='Confidence',
            color=alt.Color('Aspect', legend=None),
            tooltip=['Aspect', 'Confidence']
        ).properties(height=200)
        
        st.altair_chart(chart, use_container_width=True)

def render_batch_analysis_ui(tokenizer, model, aspect_classifier, settings):
    st.header("Batch Analysis", divider="gray")
    
    source_tab, results_tab, charts_tab = st.tabs(["Data Source", "Analysis Results", "Visualizations"])
    
    with source_tab:
        st.subheader("Upload CSV Data")
        
        csv_file = st.file_uploader(
            "Upload CSV with text data", 
            type=["csv"], 
            help=f"Must contain a '{settings['text_column']}' column"
        )
        
        if csv_file:
            try:
                # Display sample of uploaded data
                df = pd.read_csv(csv_file, encoding=settings["csv_encoding"])
                
                if settings["text_column"] not in df.columns:
                    st.error(f"No '{settings['text_column']}' column found in CSV! Available columns: {', '.join(df.columns)}")
                    return
                
                st.success(f"CSV loaded successfully. Found {len(df)} rows.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    keyword_filter = st.text_input(
                        "Filter by keyword (optional):",
                        help="Only include rows containing this text"
                    )
                
                with col2:
                    limit = st.number_input(
                        "Limit rows (for performance)", 
                        value=min(50, len(df)),
                        min_value=1, 
                        max_value=1000
                    )
                
                # Preview data
                st.subheader("Data Preview")
                st.dataframe(df.head(5), use_container_width=True)
                
                # Process button
                if st.button("Run Batch Analysis", type="primary", use_container_width=True):
                    # Apply filters
                    if keyword_filter.strip():
                        df = df[df[settings["text_column"]].astype(str).str.lower().str.contains(keyword_filter.lower())]
                        st.info(f"Filtered to {len(df)} rows containing '{keyword_filter}'")
                    
                    df = df.head(limit)
                    
                    # Extract text items for processing
                    text_items = df[[settings["text_column"]]].rename(columns={settings["text_column"]: "text"}).to_dict('records')
                    
                    with st.spinner("Processing batch data..."):
                        results = analyze_batch(
                            text_items,
                            tokenizer,
                            model,
                            aspect_classifier,
                            settings["aspect_threshold"],
                            settings["custom_aspects"],
                            settings["max_workers"]
                        )
                        
                        # Store results in session state
                        st.session_state.batch_results = results
                        
                        # Automatically switch to results tab
                        st.session_state.active_tab = "results"
                        st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    with results_tab:
        if "batch_results" in st.session_state and st.session_state.batch_results:
            results = st.session_state.batch_results
            
            # Convert to DataFrame for display
            result_df = pd.DataFrame([{
                "text": r["text"],
                "sentiment": r["sentiment_label"],
                "confidence": r["sentiment_confidence"],
                "aspects": ", ".join([a["aspect"] for a in r["aspects"]]) if r["aspects"] else "None"
            } for r in results])
            
            st.subheader(f"Results for {len(result_df)} Items")
            
            # Add download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results as CSV",
                csv,
                "sentiment_analysis_results.csv",
                "text/csv",
                key='download-csv'
            )
            
            # Display results table
            st.dataframe(result_df, use_container_width=True)
            
            # Show detailed view for selected item
            st.subheader("Detailed View")
            selected_idx = st.selectbox(
                "Select item for detailed view:",
                range(len(results)),
                format_func=lambda i: f"Item {i+1}: {results[i]['text']}"
            )
            
            if selected_idx is not None:
                display_single_result(results[selected_idx])
        else:
            st.info("Please run a batch analysis to see results here.")
    
    with charts_tab:
        if "batch_results" in st.session_state and st.session_state.batch_results:
            results = st.session_state.batch_results
            
            st.subheader("Analysis Visualizations")
            
            # Create sentiment distribution chart
            sentiment_counts = pd.DataFrame([{"sentiment": r["sentiment_label"]} for r in results])
            sentiment_counts = sentiment_counts["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            
            # Set consistent colors
            sentiment_colors = {
                "positive": "#2ecc71",
                "neutral": "#3498db",
                "negative": "#e74c3c"
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Distribution")
                
                if settings["chart_type"] == "Pie Chart":
                    fig = px.pie(
                        sentiment_counts, 
                        names="Sentiment", 
                        values="Count",
                        color="Sentiment",
                        color_discrete_map=sentiment_colors
                    )
                else:  # Default to bar
                    fig = px.bar(
                        sentiment_counts, 
                        x="Sentiment", 
                        y="Count",
                        color="Sentiment",
                        color_discrete_map=sentiment_colors
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Create aspect distribution chart
            all_aspects = []
            for r in results:
                for a in r["aspects"]:
                    all_aspects.append({"aspect": a["aspect"], "confidence": a["confidence"]})
            
            if all_aspects:
                aspect_df = pd.DataFrame(all_aspects)
                aspect_counts = aspect_df["aspect"].value_counts().reset_index()
                aspect_counts.columns = ["Aspect", "Count"]
                
                with col2:
                    st.subheader("Aspect Distribution")
                    
                    if settings["chart_type"] == "Pie Chart":
                        fig = px.pie(
                            aspect_counts, 
                            names="Aspect", 
                            values="Count",
                            color="Aspect",
                            color_discrete_sequence=px.colors.sequential.Blues
                        )
                    else:  # Default to bar
                        fig = px.bar(
                            aspect_counts.sort_values("Count", ascending=False), 
                            x="Aspect", 
                            y="Count",
                            color="Aspect",
                            color_discrete_sequence=px.colors.sequential.Blues
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Aspect by sentiment chart
                aspect_sentiment = []
                for r in results:
                    sentiment = r["sentiment_label"]
                    for a in r["aspects"]:
                        aspect_sentiment.append({
                            "aspect": a["aspect"], 
                            "sentiment": sentiment
                        })
                
                if aspect_sentiment:
                    aspect_sentiment_df = pd.DataFrame(aspect_sentiment)
                    aspect_sentiment_counts = aspect_sentiment_df.groupby(
                        ["aspect", "sentiment"]
                    ).size().reset_index(name="count")
                    
                    st.subheader("Aspects by Sentiment")
                    
                    fig = px.bar(
                        aspect_sentiment_counts,
                        x="aspect",
                        y="count",
                        color="sentiment",
                        barmode="group",
                        color_discrete_map=sentiment_colors
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show correlation of confidence
                    if "confidence" in aspect_df.columns:
                        st.subheader("Aspect Detection Confidence")
                        
                        fig = px.box(
                            aspect_df,
                            x="aspect",
                            y="confidence",
                            color="aspect",
                            points="all"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.info("No aspects were detected with the current threshold.")
        else:
            st.info("Please run a batch analysis to see charts here.")

def main():
    st.title("📊 Advanced Aspect-Based Sentiment Analysis")
    
    st.markdown("""
    This app analyzes text to identify sentiment and relevant aspects. Upload a CSV or try the single text analysis.
    
    - **Sentiment Analysis**: Uses RoBERTa to classify text as positive, neutral, or negative
    - **Aspect Detection**: Uses BART zero-shot classification to identify relevant aspects
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        tokenizer, model = load_sentiment_model()
        aspect_classifier = load_aspect_model()
    
    # Get settings from sidebar
    settings = render_sidebar()
    
    # Create tabs for single and batch analysis
    single_tab, batch_tab, about_tab = st.tabs(["Single Analysis", "Batch Analysis", "About"])
    
    with single_tab:
        render_single_analysis_ui(tokenizer, model, aspect_classifier, settings)
    
    with batch_tab:
        render_batch_analysis_ui(tokenizer, model, aspect_classifier, settings)
    
    with about_tab:
        st.markdown("""
        ## About This App
        
        This advanced aspect-based sentiment analysis tool combines:
        
        - **RoBERTa** for sentiment classification
        - **BART** for zero-shot aspect detection
        - **Parallel processing** for efficient batch analysis
        - **Interactive visualizations** for result interpretation
        
        ### How It Works
        
        1. Text is analyzed for overall sentiment (positive, neutral, negative)
        2. The same text is analyzed for relevant aspects using zero-shot classification
        3. Results are displayed with confidence scores and visualizations
        
        ### Performance Notes
        
        - First-time loading may take longer as models are downloaded
        - Processing large batches can be resource-intensive
        - Adjust the aspect threshold to control sensitivity of aspect detection
        
        ### Models Used
        
        - Sentiment: `cardiffnlp/twitter-roberta-base-sentiment`
        - Aspects: `facebook/bart-large-mnli`
        """)

if __name__ == "__main__":
    main()
