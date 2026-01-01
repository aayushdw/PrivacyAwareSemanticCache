I want to build a production grade multi-stage embedding model evaluation pipeline.
This pipeline's output final output is a LoRA based fine-tuned embedding model which will be used for semantic caching.

Below are the details. I am pondering over a few questions which I have added below for which I want you input.
I want to ensure production best practices for these.
Also help me identify are other key design decision that I should be thinking more about.

# Requirements
The pipeline would:
1. Use a model registry to keep information about candidate models.
(How should this be kept?)

2. Each candidate model would go through an evaluation phase. 
The evaluation will tune threshold (threshold above which statements are considered similar) over a small training dataset. Then, performance would be evaluated over a small test dataset.
(Where should the performance stats be kept?)
(Threshold tuning would be subject to some constraint.
Ex: Maximize accuracy ensuring precision is greather than 80 percent. i.e. we need support for custom performance metrics)

3. Models will be ranked based on performance and top-N models would move on to the next stage. (N would be small, at most 3)

4. Shortlisted models will go through a LoRA training stage (this can be a pipeline in itself) over a larger dataset.
(How to store the LoRA weights of the models)

5. Ensure proper logging in every stage using MLFlow and prefect and build visualization of the pipeline execution and results.

# Tech Stack
- Python
- MLFlow
- Prefect

# Device on hand
- M4 Macbook Air

