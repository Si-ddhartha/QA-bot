# Q&A Bot with PDF Integration
A Question and Answer (Q&A) bot built using Streamlit, LangChain, Google Generative AI embeddings, and Chroma, enabling users to ask questions related to the content of uploaded PDF files.

## Features
- Upload PDF files and interactively ask questions based on the document's content.
- Utilizes Google Generative AI embeddings for document and query embeddings.
- Stores and retrieves document chunks using Chroma for efficient similarity search.
- Simple, intuitive UI using Streamlit.

## Setup Guide
1. Clone the repository:
   ```
   git clone <repo-url>
   cd <folder>
   ```
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up your **Google API key**:
   - Create a .env file in the project directory.
   - Add your key: GOOGLE_API_KEY=<your_google_api_key>
5. Run the application:
   ```
   streamlit run QA_bot.py
   ```

## Deployed Link
https://qna-bot1.streamlit.app/
