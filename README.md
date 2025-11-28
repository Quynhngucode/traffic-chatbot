# Vietnam Traffic Law Chatbot (RAG & LangGraph)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.1-green)](https://python.langchain.com/)
[![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange)](https://deepmind.google/technologies/gemini/)

This project builds an AI assistant designed to look up and answer inquiries regarding Vietnam's Road Traffic Law. The system utilizes **RAG (Retrieval-Augmented Generation)** techniques to extract information from official legal documents (PDF files stored in the `data` directory), ensuring accurate answers backed by specific legal grounds.

## Project Structure

```text
traffic-law-chatbot/
├── data/                  # Contains legal document data (Knowledge Base)
│   ├── 36-2024-qh15.pdf   # Law on Road Traffic Order and Safety      
│   └── ...                # Other legal PDF documents  
├── src/                   # Main source code (RAG logic, Evaluation...)
├── .gitignore             # Files excluded from Git
├── chatbot_env.yml        # Conda environment configuration (Dependencies)
├── app.py                 # Main application entry point    
└── README.md              # Project documentation
```
## Installation & Setup
### 1. Clone the repository
  ```bash
  git clone [https://github.com/username/traffic-law-chatbot.git](https://github.com/username/traffic-law-chatbot.git)
  cd traffic-law-chatbot
  ```

### Environment Setup (Using Conda)
The project uses `chatbot_env.yml` to manage dependencies. Run the following command to create the environment:
  ```bash
  conda env create -f chatbot_env.yml
  ```

Once created, activate the environment:

  ```bash
  conda activate chatbot_env
  ```

### 3. Configure API Keys
Create a `.env` file in the root directory and populate it with the following information (replace with your actual keys):

  ```env
  # Google Gemini API (Bắt buộc)
  GOOGLE_API_KEY=AIzaSy...
  
  # LangSmith (Tùy chọn - Để tracking và debug)
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=lsv2_pt_...
  LANGCHAIN_PROJECT=traffic-law-chatbot
  ```

### 4. CRun the Application
To start the Chatbot, run the following command:

  ```bash
  python app.py
  ```
