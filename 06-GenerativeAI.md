
## Basics of Cloud

AWS has 3 pricing fundamentals, following the pay-as-you-go pricing model

- Compute: Pay for compute time
- Storage: Pay for data stored in the Cloud
    - Data transfer OUT of the Cloud
    - Data transfer IN is free

## Generative AI

Gen AI is a subset of deep learning

It can generate new data based on the data it was trained on.

![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image.png)

- Foundational Model: To Generate data we must need a foundational model
    - They are trained on wide variety of data and these models take millions to train.
    - GPT4.0 is the foundational model behind CHATGPT
    - Examples:
        - Anthropic
        - Perplexity
        - OpenAI
        - Meta
    - Some are open source like Meta and Google Bert, some are under commercial licence like OPEN AI
- LLM: Types of AI but designed to generate coherent human like text.
    - Example is chatgpt
    - They are trained on TEXT data , very large corpus of data.
    - Tasks they can perform are:
        - Translation
        - Summarization
    - How it works?
        - We give it a promt⇒model will leverage all the existing data it was trained on to find the answer.
        - Answers are non deterministic for everyone which means the answer for the  same prompt for one person can be different from person B even if the prompt is exactly same as Person A.
        - Why is that?
            - LLM generates a list of potential words alongside probabilities.
            - Algorithm then selects a word from that list.
            
            ![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%201.png)
            
- Gen AI-Images:
    - Diffusion Model:
        - We add noise to the images that we train the model on and then we keep adding the noise till the AI is trained on it.
        - Then once we give a prompt for an image, it removes the noise from the training data and then gives us the image, which is not the exact image it was trained on but by using the knowledge of that particular data it regenerates new images.(Backward diffusion model)
- Amazon Bedrock:
    - GEN AI Service on AWS
    - Fully managed service and you keep control of your data to train the model.
    - Unified API to use wide array of GEN AI Models
    - Models on bedrock:
        - cohere
        - meta
        - A121Labs
        - stability.ai
    - Bedrock will create a copy of the LLM and none of your data will never be sent back to these providers and your data is always protected.
    
    ![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%202.png)
    
    - Base Foundation Models:
        - Depends on performance, compliance, requirements, model sizes , latency of the model.
        - Multimodal: Wide combination of inputs(Audio, Video,Text)
        - Amazon Titan:
            - One of the foundation model directly from AWS
            - **Does not take an image** as a prompt only text prompts are supported.
            - Can do Images, Text and be customised with your own data
            - Video support is still pending
        - Smaller Models are cost effective.
        - [Stability.AI](http://Stability.AI) is specific for image generation.
        - [Claude.AI](http://Claude.AI) gives the most amount of tokens for context windows and can intake a bigger input compared to Amazon titan or LLAMA 2.
        
        ![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%203.png)
        
- Fine tuning a model:
    - Adapt a copy of foundation model with your own data
    - Training data must be:
        - Adhere to specific format
        - Stored in amazon s3
        - You must use a `provisioned throughput`
        - `Not all models can be fine tuned` in amazon bedrock.
    - Used when we need to train the model on a particular field.
    - Labeled examples that gives promt-response pairs.
    
    **Continued-Pre Training:**
    
    - Provide `unlabelled data` to continue train the foundational model
    - Also called `domain-adaptation fine tuning` to make a model expert in a specific domain
    - Input is just alot of information that we feed the model
    - Example: Feeding the entire aws documentation to make the model an expert on AWS.
    
    Single Turn Messaging
    
    Fully managed service for developers/data scientist
    
    It is difficult to do all the process at one place but sagemaker is one stop solution for that.
    
    Build and train machine learning models
    
    Deploy those models and monitor the performance of the models
    
    Supervised Algorithms:
    
    Linear Regression and classifications
    
    KNN Classifications
    
    Unsupervised Algorithms:
    
    Principal Component Analysis
    
    K Means-Find grouping in the data
    
    anomaly detection
    
    Image processing
    
    Textual Algorithms: NLP
    
    ![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%204.png)
    
    Automatic Model Tuning:
    
    Define the objective of the tuning
    
    It will automatically choose the hyperparameters ranges, maximum runtimes of a tuning job, stop conditions etc.
    
    Deployment
    
    One click deployment, automatic scaling given to this
    
    Managed solution
    
    Real time:
    
    One prediction at a time
    
    Serverless:
    
    More latency due to cold starts
    
    ![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%205.png)
    
    Asynchonrous
    
    For large payload sizes upto 1 gb
    
    Long processing times
    
    `It is near real time not real time`
    
    Both request and response are stored in S3
    
    Processing time is maximum one hour.
    
    Batch Processing
    
    Predictions for entire data sets⇒ Multiple predictions can be given as the entire data set is given as an input
    
    Processing time is maximum one hour.
    
    `Latency is higher as the entire dataset is given.`
    
    Concurrent processing of data⇒ BEST CHOICE.
    
    100MB per batch(mini batch)→ Per invocation.
    
    ![image.png](AWS%20Certified%20AI%20Practitioner%2014e898cf4c9e80b79415dc48856fae87/image%206.png)
    
## Terms Used:
- Accuracy measures the proportion of correct predictions made by the model.
    - It is calculated as:
Accuracy = Number of correct predictions/ Total number of predictions 
    - Accuracy is particularly useful in `classification tasks when the classes are balanced (i.e., when each class has a roughly equal number of examples)`.
Sure! I'll explain these terms in simpler language, especially focusing on why they are not suitable for evaluating an image classification model, like the one you're working with for plant disease detection.

### **A. R-squared score**
- **What it is**: R-squared is a measure used to evaluate how well a model explains a set of data in **regression tasks** (predicting continuous values, like price or temperature). 
    - It shows how much of the variation in the output (e.g., house prices) can be explained by the model.
- **Why it's not suitable for classification**: In **classification** (like predicting plant diseases), the output is **discrete categories** (e.g., "Healthy," "Diseased"), not continuous values. 

### **C. Root Mean Squared Error (RMSE)**
- **What it is**: RMSE is another evaluation metric used for **regression tasks**. It measures the average `difference between predicted values and actual values`. For example, if you're predicting a person's weight (in kg), RMSE would tell you how far off your predictions are from the true weights on average.
- **Why it's not suitable for classification**: Like R-squared, **RMSE** applies to **regression**, not classification. 

### **D. Learning Rate**
- **What it is**: The **learning rate** is a setting you use during **model training**. It controls how big a change the model makes to itself after each step in the training process. Think of it as how "fast" the model learns from the data. If the learning rate is too high, the model could learn too quickly and make mistakes. If it's too low, the model might take too long to learn.
- **Why it's not suitable for evaluation**: The **learning rate** is a **training setting**, not a way to measure the model’s performance. It has nothing to do with whether the model is correctly classifying plant diseases or not. You use it to help train the model, but once the model is trained, you want an **evaluation metric** (like **accuracy**) to see how well it's performing.
## Algorithms
### A. Decision Trees
What it is: A Decision Tree is a model that makes decisions based on asking a series of questions about the features of the data (like gene characteristics). It’s like a flowchart, where each node represents a question (e.g., "Is gene expression level > X?"), and the branches represent the answers (yes or no).
How it works: The tree splits the data into subsets based on the features that best separate the categories (like healthy vs. diseased genes). It does this until it reaches a decision (the final classification of the gene into one of the 20 categories).
Interpretability: Decision trees are highly interpretable because you can easily trace through the tree to see how a decision was made. For example, you can see exactly which feature (gene characteristic) led to which classification.
### B. Linear Regression
- What it is: Linear Regression is used to predict continuous values (e.g., predicting a person's weight or income) based on the relationship between input features and the output.
- How it works: It finds the line that best fits the data points. In a gene context, it would try to draw a straight line that predicts a continuous output, like gene expression levels. However, linear regression is not suitable for classification tasks where the output is a category (like "disease type").
- Interpretability: Linear regression can be somewhat interpretable since you can look at the coefficients of the model to see the impact of each feature on the predicted value. But it doesn’t work for categorical classification problems.
### C. Logistic Regression
- What it is: Logistic Regression is used for binary classification tasks (e.g., classifying something as "yes" or "no"). It works by estimating the probability that an input belongs to a particular class.
- How it works: It uses the input features (gene characteristics) to calculate the probability of a certain category, and based on that probability, it classifies the input into one of the two possible classes. For multi-class problems (like your 20 categories), you’d use Multinomial Logistic Regression.
- Interpretability: Logistic regression is fairly interpretable because you can look at the coefficients to see how each gene characteristic affects the probability of being in a particular category. But like linear regression, it's better suited for simpler problems or smaller sets of categories.
### D. Neural Networks
- What it is: Neural Networks are complex models inspired by the human brain, and they are particularly good at learning patterns in large datasets. They consist of layers of interconnected nodes (neurons) that process the input data through complex transformations.
- How it works: Neural networks can learn very non-linear relationships between gene characteristics and the output categories. They are powerful for complex tasks like image recognition or speech recognition. However, neural networks are often called "black-box" models because it’s hard to interpret exactly how they arrive at their predictions.
- Interpretability: Neural networks are not very interpretable by design, meaning it’s difficult to understand how individual gene features influence the model’s predictions. You would need additional tools (like LIME or SHAP) to interpret the model’s behavior.
