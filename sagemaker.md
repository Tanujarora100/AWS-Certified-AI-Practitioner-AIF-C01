
# SageMaker

### Sage Maker Studio

- Team Collaboration
- Automated Workflows
- Deploy ML models everything at one place.

### Sage-maker Data wrangler:

- Prepare the data to train the model such as tabular or image data
- Data preparation, transformations on the data, feature engineering everything is done here.
- `SQL SUPPORTED`
- `Data Quality Feature Supported`

![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%2012.png)

### What are ML Features?

- Features are the inputs to ML Models used during training of those models and used for inference.

![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%2013.png)

### Sage-Maker Feature Store

- Ingest features from a variety of sources
- Ability to define the data transformation and logic in the feature directly using feature store
- can publish directly from sage maker data wrangler to the feature store also.

### Sage Maker Clarify:

- Evaluate Foundational Models: does comparison of models
- gives some tasks to these models and then clarify evaluates the models
- Evaluate human factor such as friendliness or humour
part of sage maker studio
- We can use our own team to evaluate these models⇒ Involves humans also in clarify

**Model Explainability in Clarify:**

- Set of tools that explain how these model works internally
- Understand the prediction logic of these models
- Helps in increasing the trust and understanding of the model.

![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%2014.png)

- Can detect human biases in our own datasets and models
- Specify the input features and biases in our datasets which will be automatically detected

### Sage Maker Ground Truth:
- Made for RLHF-Reinforcement learning from human feedback
- used for model review
- Reinforced learning where human feedback is included in the `reward` function.

![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%2015.png)

- Labelling Data In Your Inputs: `Use Sagemaker ground truth plus`

### SageMaker Governance

#### Sagemaker Model Cards:

- Essential Model Information, risk ratings, training details.
- This is used for creating `ML Documentation`.

### Sagemaker Model Dashboard:
- Repo for all models and their insights+information
- Can be access from the sagemaker console
- help you find models which violate thresholds you have set for data quality, inference etc.

### Sagemaker Model Monitor
- monitors the quality of deployed models, can be scheduled or always. We get alerts if they violate the quality standards

### Sagemaker Model Registry: 
- Central repo to track manage and version machine learning models like git. We can have different version for models, Manage Model Approval status of a model to take the approvals before being deployed.

### Sagemaker Pipelines: 
- Automate the process of building, deploying and training a machine learning model. eg: mlops
- This is a CI-CD Pipeline for sagemaker.

#### Supported Steps:
- processing-feature enggineering
- Training
- Tuning for hyperparameter Training
- AutoML-Auto training
- Model: Create or register a model
- Quality Check
- Clarify Check: Perform drifts check against baselines.

![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%2016.png)

### Sage maker Role Manager:

- Define roles within sagemaker like IAM.
- Example: Data Engineer, Data Scientist etc

### Sagemaker Jumpstart:
- pre trained foundation models or computer vision models directly on the sagemaker similar to bedrock
- Collection is larger compare to bedrock it has huggingface,databricks, meta, stability AI etc.
- Full control on the deployment of the model.

![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%2017.png)

### Sagemaker Canvas: 
- Build ML models using a visual interface.
- `no coding required for this.`
- powered by sagemaker autopilot. 
- part of sagemaker studio and does something calls AUTO-MLOPS
- MLFLOW: Open source tool which helps ML teams manage the entire machine learning lifecycle.

### ML Flow Tracking servers:
- Used to track runs and experiments

![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%2018.png)

![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%2019.png)