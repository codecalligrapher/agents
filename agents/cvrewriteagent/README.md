# CVRewrite Agent Documentation

## Overview
The `CVRewriteAgent` is designed to automate the process of rewriting CVs (resumes) using natural language processing and machine learning. It takes a LaTeX-formatted CV as input, along with a job description, and produces an optimized version of the CV tailored for the specified role.

## Features
- Automated CV formatting and optimization.
- Customizable analysis based on job descriptions.
- Support for handling LaTeX formatted CVs.
- Integration with Gradio for a user-friendly web interface.

## Installation
To use the `CVRewriteAgent`, you need to have Python installed. Additionally, install the required dependencies using pip:

```bash
pip install gradio requests
```

Ensure that the Ollama server is running on localhost:11434 as this agent relies on it for processing.

## Usage

### Running the Gradio Frontend
To start the Gradio frontend for interacting with the `CVRewriteAgent`, run the following command in your terminal:

```bash
python agents/cvrewriteagent/agent.py
```

This will launch a web interface where you can upload a .tex CV file and paste a job description. The agent will analyze the CV, provide recommendations, and output an optimized version of the CV.
