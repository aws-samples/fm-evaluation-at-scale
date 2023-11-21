# Operationalize LLM Evaluation at Scale leveraging Amazon SageMaker Clarify and MLOps services

This solution streamlines the customer journey by facilitating the evaluation process of Large Language Models at scale, offering an automated and efficient approach 
that enhances the overall experience and ensures a smoother path to decision-making.  
The implementation is made by combining Amazon SageMaker Clarify and Amazon SageMaker Pipelines.
We provide pipeline implementations that automates different steps of an evaluation process such as data preprocess, model deploy, model evaluation,
best model selection, resources cleanup. Model fine-tuning is also supported as a preliminary step.  
Each pipeline implementation will work together with a *.yaml* file. This file provide a way to dynamically change the models under evaluation.
At the same time the configuration provide a quick way to define different parameters tied with the model like fine-tuning parameters, deployment parameters,
inference parameters and evaluation parameters.  
Editing the yaml file will allow you to reuse the pipeline code without modification.

You will find two folders in the solution:
### sm_pipelines
This folder contains starting examples that could be run with Amazon SageMaker Jumpstart models.
- *pipeline.py* : runs together with *pipeline.yaml* . This example implements evaluation on a *meta-textgeneration-llama-2-7b-f* 
model
- *pipeline_finetuning.py* : runs with *pipeline_finetuning.yaml* . This example implements fine-tuning and evaluation on a 
- *pipeline_scale.py* : runs with *pipeline_scale.yaml* . This pipeline provides a more dynamic approach. You can define any number of SageMaker Jumpstart Models inside the *.yaml* file and
the pipeline will automatically create all the required steps. We provide an example that runs parallel evaluation and model selection between 
*meta-textgeneration-llama-2-7b-f*, *huggingface-llm-falcon-7b-bf16* foundation models and *meta-textgeneration-llama-2-7b* fine-tuned on 
a sample dataset.

### sm_pipelines_advanced
This folder contains a pipeline implementation that allows you to extend model evaluation beyond Amazon SageMaker Jumpstart.
You will find a single fully dynamic pipeline implemented in *pipeline_scale_adv.py* .
In this implementation you can switch between any model that is supported by an implemented provider.
At the moment we support (jumpstart, ...) but we are planning to support other providers in the future. You can also implement your own providers.

We provide different examples of *.yaml* files that can be loaded:
- *evaluation_singlemodel.yaml* provides an example configuration for *meta-textgeneration-llama-2-7b-f*
- *evaluation_multimodels.yaml* provides an example configuration that runs parallel evaluation and model selection between 
*meta-textgeneration-llama-2-7b-f*, *huggingface-llm-falcon-7b-bf16* foundation models and *meta-textgeneration-llama-2-7b* fine-tuned on 
a sample dataset.

You can switch between different *.yaml* configuration by editing the *ConfigParser* line :  
`config = ConfigParser('evaluation_multimodels.yaml').get_config()`

You can implement your own model providers by extending the class *ModelPipelineSteps.py*. By inheriting from this class 
you will need to implement your own *deploy*, *finetune*, *cleanup* steps and your *get_model_runner* method required for running the evaluation step.

Finally, you will need to add your new model provider in the *for* loop inside the file *import_models.py* to allow your model provider
to be loaded automatically by the pipeline.

## Configuring the solution

- This example can only run on either Python 3.8 or Python 3.10. 
Otherwise, you will get an error message prompting you to provide an image_uri when defining a step.

### Running the solution on Amazon SageMaker Studio
TODO: Anything to do ?

### Running the solution on a local IDE
You can launch the pipeline configurators from your local IDE (Pycharm or Visual Studio Code)
- Configure a valid Amazon SageMaker Execution Role in *config.yaml* file.  
`RoleArn: arn:aws:iam::<...>:role/service-role/AmazonSageMaker-ExecutionRole-<...>`
- At the end of each pipeline configure a valid Amazon SageMaker Execution Role in the upsert function.  
`pipeline.upsert(role_arn="arn:aws:iam::<...>:role/service-role/AmazonSageMaker-ExecutionRole-<...>")`

  
  
  
  
  
  
  
  
