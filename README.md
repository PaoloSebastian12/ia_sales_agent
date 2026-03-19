# 🚀 AI Sales & Compliance Intelligence Agent

This repository features an advanced Multi-Engine AI Agent designed to bridge the gap between complex sales engineering, customer history, and legal compliance. By combining Hierarchical RAG with autonomous routing, the system provides high-fidelity, evidence-based insights for enterprise decision-making.


## 🧠 Core Technical Features

* **Hierarchical Auto-merging Retrieval:** Unlike standard RAG, this system uses a parent-child recursive structure:

Leaf Node Retrieval: The engine searches for small, precise "leaf" nodes (512 tokens) to ensure high semantic matches.

Recursive Merging: If enough child nodes from the same context are retrieved, the engine automatically "merges" them into their larger parent context. This solves the "lost in the middle" problem and provides the LLM with complete paragraphs or sections for better reasoning.
* **Intelligent Multi-Engine Routing:** The core is a RouterQueryEngine that acts as a decision-making agent. It dynamically evaluates user intent to select the most relevant specialized engine:

Technical Engine: Analyzes PDF manuals and AWS/Architecture specs.

Legal & Compliance Engine: Processes complex TOS, Google Cloud contracts, and liability terms.

CRM/Email Engine: A specialized CSV retriever that tracks customer feedback and previous interactions.
* **RAGAS Evaluation Framework:** Integrated quality assurance that measures **Faithfulness** (hallucination mitigation) and **Answer Relevancy** in real-time using NLP metrics.
* **Streamlit UI:** A clean, production-ready interface for interactive querying, system status monitoring, and metric visualization.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```
### 🛠️ Install Dependencies
    Ensure you have Python 3.9+ installed, then run:
    pip install -r requirements.txt

### 3. Environment Configuration
    Create a .env file in the root directory and add your Google API credentials:
    GOOGLE_API_KEY=your_gemini_api_key_here

### 🚀 Execution
    You can try the AI Sales & Compliance Intelligence Agent live here:
    👉 [Live Demo - Try the App here!]()

## 📊 Project Architecture

    main.py: The primary entry point. Manages the Streamlit state, UI components, and the orchestration of the RAG engine.

    src/engine.py: The core logic for index construction, hierarchical node parsing, and the multi-tool routing agent.

    src/evaluator.py: Implementation of the RAGAS framework for automated response auditing and performance metrics.

## 📁 Suggested Data Structure (Demo Mode):

To test the system immediately, organize your local files as follows:

    /
    ├── main.py
    ├── requirements.txt
    ├── .env
    ├── src/
    │   ├── engine.py
    │   ├── evaluator.py
    │   └── LOGO.png
    └── data/
        ├── technical/   <-- Place technical PDFs here
        ├── contracts/   <-- Place legal/TOS PDFs here
        └── emails/
            └── emails.csv <-- Place customer interaction history here
