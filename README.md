# FX-Agents: A Multi-Agent Framework for Exchange Rate Forecasting

![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)![License](https://img.shields.io/badge/License-MIT-green)![Status](https://img.shields.io/badge/status-in%20progress-orange)

## Overview

FX-Agents is a novel, multi-agent paradigm designed to revolutionize exchange rate forecasting. Traditional forecasting methods often suffer from limited data scope, manual and inefficient data processing, subjective feature selection, and a lack of automated optimization. FX-Agents systematically overcomes these challenges by decomposing the forecasting workflow into four specialized, coordinated agents. This framework leverages intelligent automation to provide a more robust, transparent, and efficient solution for financial market prediction.

Based on LangGraph, FX-Agents implements a modular and collaborative system where each agent handles a specific stage of the forecasting process, from data perception to final prediction, ensuring a seamless and adaptable workflow.

## Framework

![FX-Agents Framework](./fig1.png)
*Figure 1: The FX-Agents Framework*

The core of FX-Agents is a multi-agent system where each agent has a distinct role, designed to address a specific limitation of traditional forecasting processes. This collaborative structure enhances adaptability and efficiency in complex forecasting tasks.

## Key Features

-   **🧩 Modular Agent-based Design:** Based on LangGraph, the system is divided into four specialized agents for a clear and maintainable workflow.
-   **Perception Agent (PA1):** Automates the collection of multimodal data from diverse sources, including APIs for structured data and web scraping for unstructured information, overcoming the narrow scope of traditional methods.
-   **Planning Agent (PA2):** Replaces cumbersome manual workflows by automating complex data preprocessing and feature engineering for both structured and unstructured data.
-   **Decision Agent (DA):** Implements a dynamic, evidence-based feature selection mechanism using Retrieval-Augmented Generation (RAG) to replace subjective expert judgment, ensuring transparency and robustness.
-   **Forecasting Agent (FA):** Automates model training, hyperparameter optimization (with Optuna), and provides interpretable results (with SHAP), overcoming the inefficiencies of manual model handling.

## Project Structure

```
FX-Agent/
├── data/                # Raw data and knowledge base
├── model/               # Pre-trained models and related resources
├── src/                 # Main source code
│   ├── agents/          # Implementation of PA1, PA2, DA, FA
│   ├── core/            # Core workflow graph and state management
│   ├── models/          # Forecasting models (RNN, LSTM, etc.)
│   ├── rag/             # RAG components (retriever, knowledge base)
│   ├── tools/           # Tools used by agents (data fetchers, scrapers)
│   └── utils/           # Utility scripts (data processing, optimizers)
├── .env.example         # Environment variable template
├── main.py              # Main entry point to run the workflow
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/FX-Agent.git
    cd FX-Agent
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Copy the `.env.example` file to a new file named `.env` and fill in the necessary API keys and configurations.
    ```bash
    cp .env.example .env
    ```

## How to Run

To start the entire forecasting workflow, run the main script from the root directory:

```bash
python main.py
```

The framework will execute the agent-based workflow, and the final forecast results and analysis will be generated as per the configuration.

## The FX-Agents Workflow in Detail

### Perception Agent (PA1)

The Perception Agent (PA1) is the framework's automated, multimodal data perception and acquisition module. It addresses the inefficiency of acquiring heterogeneous, multi-source data. PA1 uses a dual-track strategy:
-   **Function Calling:** For sources with stable interfaces like APIs (e.g., FRED, Yahoo Finance), PA1 uses function calls to fetch structured data.
-   **Model Context Protocol (MCP):** For unstructured sources without official APIs (e.g., news sites, government statements), PA1 uses the MCP standard to interact with scraping services like Firecrawl, which returns clean, "AI-Ready" data.
This approach decouples the agent's reasoning from the data execution details, ensuring efficient, broad, and robust data acquisition.

### Planning Agent (PA2)

The Planning Agent (PA2) handles the complex data processing stage. It receives multimodal data from PA1 and employs a hybrid strategy:
-   **For structured data:** It uses deterministic tools for cleaning, resampling, handling missing values, and normalization.
-   **For unstructured text:** It leverages its core LLM to perform intelligent semantic transformation, such as converting news text into quantitative sentiment scores using specialized internal tools.
PA2 consolidates all processed information into a unified, structured candidate feature set for the next stage.

### Decision Agent (DA)

The Decision Agent (DA) introduces an evidence-based evaluation framework for intelligent feature selection, replacing opaque or subjective traditional methods. Inspired by SELF-RAG, the DA operates in a three-stage workflow:
1.  **Evidence Retrieval:** For each candidate feature, the DA generates a query and retrieves relevant evidence from a curated, domain-specific knowledge base (containing academic literature, official reports, etc.).
2.  **Prompt-driven Evaluation with Self-Critique:** The DA uses the retrieved evidence to evaluate the feature across three dimensions: *Relevance*, *Supportiveness*, and *Utility*. This process is transparent, generating both scores and qualitative justifications.
3.  **Iterative Refinement:** If a feature scores low, the DA doesn't immediately discard it. Instead, it reflects on the reasoning, refines its query, and re-attempts retrieval to find better evidence, ensuring a rigorous and flexible selection process.

### Forecasting Agent (FA)

The Forecasting Agent (FA) is the final orchestrator, responsible for generating forecasts and evaluating performance. Upon receiving the high-quality feature set from the DA, the FA:
1.  **Automates Hyperparameter Optimization:** It uses Optuna to efficiently search for the best hyperparameters for various forecasting models (e.g., RNN, LSTM, Transformer).
2.  **Trains and Forecasts:** It trains the optimal model on the selected features to generate the final exchange rate forecast.
3.  **Ensures Interpretability:** It integrates model explanation techniques like SHAP to quantify the contribution of each feature. It then uses an LLM to translate these technical insights into accessible, natural language reports, explaining the key drivers behind the forecast.



