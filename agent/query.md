Goal:
You are building a synthetic data generation pipeline. This pipeline will take as input a dataset of questions and will generate and store 2 types of synthetic data using LLM models.
1. Semantically similar questions. 
2. Similar looking but different intent questions

Source files for the pipeline should live in data/synthetic_data_generator. You should use Google's Gemini model for the LLM calls.

Dataset to use (Customizable): Defaults to "data/raw/small_final_questions.csv"
Dataser has header format (qid,question), where qid represents a unique question id, "question" column contains the actual question text.

Input: Dataset path, output path

Configuration:
GEMINI_API_KEY - Primary API Key for LLM calls
Model list - List of models to use for LLM calls.
LLM_RPM_LIMIT - Requests per minute per model limit for LLM calls. Pipeline should honor this and try to stay below the per minute limit for each model.

Tech Stack:
Use python to implement this pipeline. 
Use LangGraph for orchestration of LLM calls and various flows.

Pipeline Initialization: Upon initialization the pipeline creates the output file if it doesn't exist. It initializes all LLM models that nodes can use for LLM calls.

Parallelization behaviour:
Pipeline will use all the input models, to parallely generate synthetic data. However a single question_id should be processed by exactly one model.
The synthetic data would be stored in the output file while ensuring there is no race condition to write. Acquire appropriate locks in output file before writing.
When processing a question the node should split into two LLM queries, one for two semantically similar questions and another for three similar looking but different intent questions.

Output data format:
The output file should have the following format
{id, original_question, generated_question, is_semantically_similar}

Output log:
Create a log file for ids that fail the question generation.