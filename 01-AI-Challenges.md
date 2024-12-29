# Responsible AI
- Making sure AI systems are transparent and trustworthy
- Throughout the AI lifecycle: design, development, deployment,
monitoring, evaluation
- Ensure that confidentiality, integrity, and availability are maintained
- On organizational data and information assets and infrastructure

## Governance & Compliance
### Governance
- Ensure to add value and manage risk in the operation of business
- Clear policies, guidelines, and oversight mechanisms to ensure AI systems align with legal and regulatory requirements
## Compliance
- Ensure adherence to regulations and guidelines
- Sensitive domains such as healthcare, finance, and legal applications

## AWS Services- Responsible AI
- Amazon Bedrock: human or automatic model evaluation
- Guardrails for Amazon Bedrock
- Filter content, redact PII, enhanced safety and privacy…
- SageMaker Clarify
- Bias detection (ex: data skewed towards middle-aged people)
- SageMaker Data Wrangler: fix bias by balancing dataset
- SageMaker Model Monitor: quality analysis in production
- Amazon Augmented AI (A2I): human review of ML predictions
## AWS AI Service Cards
- Form of responsible AI Documentation
![alt text](image-12.png)

## Interpretability
- The degree to which a human can understand the cause of a decision
- Access into the system so that a human can interpret the model’s output
- Answer “why and how”
- High transparency => High interpretability=> Poor Performance
- Linear regression is simple to interpret but performs poorly on the other hand neural networks are performant but difficult to interpret as they are complex.
![alt text](image-13.png)
![alt text](image-14.png)
![alt text](image-15.png)
## Challenges of AI
### Toxicity
- Challenges
    - Defining what constitutes “toxicity” can be a
    challenge
    - Boundary between restricting toxic content and censorship
- Mitigation:
    - Use guardrail models to detect and filter out
unwanted content.
### Hallucinations
- Challenges
    - Assertions or claims that sound true, but are incorrect
    - This is due to the next-word probability sampling employed by LLM
- Mitigation:
    - Educate users that content generated by
    the model must be checked
    - Ensure verification of content with independent sources
### Prompt Misuses
- Poisoning
- Intentional introduction of malicious or biased data into the training dataset of a model
- Leads to the model producing biased, offensive, or harmful outputs (intentionally or unintentionally)
### Hijacking and Prompt Injection
- Influencing the outputs by embedding specific
instructions within the prompts themselves
- Hijack the model's behavior and make it produce outputs that align with the attacker's intentions (e.g., generating misinformation or running malicious code)
### Jailbreaking
- AI models are typically trained with certain ethical and safety constraints in place to prevent misuse or harmful outputs (e.g., filtering out
offensive content, restricting access to sensitive information…)
- Circumvent the constraints and safety measures implemented in a generative model to gain unauthorized access or functionality

![alt text](image-16.png)
## AWS COMPLIANCE AND GOVERNANCE
![alt text](image-17.png)
![alt text](image-18.png)
 - Policies – principles, guidelines, and responsible AI considerations
 -  Data management, model training, output validation, safety, and human oversight
 - Intellectual property, bias mitigation, and privacy protection
 - Review Cadence – combination of technical, legal, and responsible AI review
 - Clear timeline: monthly, quarterly, annually…
- Non-technical reviews on policies, responsible AI principles, regulatory requirements

## MLOps
- Make sure models aren’t just developed but also deployed, monitored, retrained systematically and repeatedly
- Extension of DevOps to deploy code regularly
### Key Principles:
- Version control: data, code, models could be rolled back if necessary
- Automation: of all stages, including data ingestion, pre-processing, training, etc…
- Continuous Integration: test models consistently
- Continuous Delivery: of model in productions
- Continuous Monitoring
![alt text](image-19.png)