import streamlit as st

from TextExtraction import extract_text_from_pdf
from EmotionDetection import extract_emotional_words, clean_text
from NER import perform_ner
from TTS import synthesize_speech_chunk, combine_mp3_files, detect_emotion, split_text

def main():
    st.set_page_config(page_title="PDF Text Analysis", layout="wide")
    st.title("PDF Text Extraction and Analysis")

    st.write(
        """
        Upload a PDF file to :
        - Perform Named Entity Recognition (NER) to identify PERSON and LOCATION entities
        - Detect emotional words categorized into various types
        - Convert text to speech 
        """
    )

    st.sidebar.title("Settings")
    chunk_size = st.sidebar.number_input("Chunk Size for NER", min_value=1000, max_value=5000, value=2000, step=500, help="Enter the chunk size for processing text (in characters).")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf", help="Only PDF files are supported.")
    
    processing_placeholder = st.sidebar.empty()

    if uploaded_file is not None:
        processing_placeholder.text("Processing...")

        with st.spinner("Extracting and analyzing the text..."):
            try:
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    cleaned_text = clean_text(text)

                    per_entities, loc_entities = perform_ner(cleaned_text, chunk_size=chunk_size)

                    emotional_words = extract_emotional_words(cleaned_text)

                    with st.expander("**Named Entity Recognition Results**"):
                        if per_entities:
                            per_entities_str = ', '.join(per_entities)
                            st.write(f"**PERSON Entities :** {per_entities_str}")
                        else:
                            st.write("No PERSON entities found.")
                        
                        if loc_entities:
                            loc_entities_str = ', '.join(loc_entities)
                            st.write(f"**LOCATION Entities :** {loc_entities_str}")
                        else:
                            st.write("No LOCATION entities found.")
                    
                    with st.expander("**Emotional Words**"):
                        emotional_found = False
                        for category, words in emotional_words.items():
                            if words:
                                st.write(f"**{category} :** {', '.join(words)}")
                                emotional_found = True
                        
                        if not emotional_found:
                            st.write("No emotional words found.")

                    emotion = detect_emotion(cleaned_text)

                    chunks = split_text(cleaned_text)
                    chunk_files = []
                    
                    for i, chunk in enumerate(chunks, start=1):
                        chunk_file = synthesize_speech_chunk(chunk, emotion, i)
                        if chunk_file:
                            chunk_files.append(chunk_file)
                    
                    if chunk_files:
                        combined_output_file = "combined_output.mp3"
                        combine_mp3_files(chunk_files, combined_output_file)

                    with st.expander("**Text-to-Speech Results**"):
                        st.audio(combined_output_file, format='audio/mp3')

            except Exception as e:
                st.error(f"An error occurred: {e}")
                return

            processing_placeholder.empty()

if __name__ == "__main__":
    main()
