import os
import argparse
from datetime import datetime

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.function_step import step
from sagemaker.workflow.step_outputs import get_step

# Import the necessary steps
from steps.preprocess import preprocess
from steps.evaluation import evaluation
from steps.finetune import finetune
from steps.cleanup import cleanup
from steps.deploy_finetuned_model import deploy_finetuned_model
from steps.deploy import deploy
from steps.selection import selection
from steps.register import register

from lib.utils import ConfigParser
from lib.utils import is_finetuning

if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    sagemaker_session = sagemaker.session.Session()

    # Define data location either by providing it as an argument or by using the default bucket
    default_bucket = sagemaker.Session().default_bucket()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input-data-path",
        "--input-data-path",
        dest="input_data_path",
        default=f"s3://{default_bucket}/llm-evaluation-at-scale-example",
        help="The S3 path of the input data",
    )
    parser.add_argument(
        "-config",
        "--config",
        dest="config",
        default="",
        help="The path to .yaml config file",
    )
    args = parser.parse_args()

    # Initialize configuration for data, model, and algorithm
    if args.config:
        config = ConfigParser(args.config).get_config()
    else:
        # Change to switch configuration or use --config argument
        # config = ConfigParser("pipeline_scale_config.yaml").get_config()
        config = ConfigParser("pipeline_scale_hybrid_config.yaml").get_config()

    evaluation_exec_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    pipeline_name = config["pipeline"]["name"]
    dataset_config = config["dataset"]  # Get dataset configuration
    input_data_path = args.input_data_path + "/" + dataset_config["input_data_location"]
    output_data_path = (
        args.input_data_path + "/output_" + pipeline_name + "_" + evaluation_exec_id
    )

    print("Data input location:", input_data_path)
    print("Data output location:", output_data_path)

    algorithms_config = config["algorithms"]  # Get algorithms configuration

    # Construct the steps
    processed_data_path = step(preprocess, name="preprocess")(
        input_data_path, output_data_path
    )

    evaluation_results_all = []

    # Iterate over models array
    for model_config in config["models"]:
        model_id = model_config["model_id"]
        model_version = model_config["model_version"]
        endpoint_name = model_config["endpoint_name"]

        model_deploy_config = model_config["deployment_config"]
        deploy_instance_type = model_deploy_config["instance_type"]
        deploy_num_instances = model_deploy_config["num_instances"]

        if is_finetuning(model_config):
            model_fine_tuning_config = model_config["finetuning"]
            train_data_path = (
                args.input_data_path + "/" + model_fine_tuning_config["train_data_path"]
            )
            validation_data_path = (
                args.input_data_path
                + "/"
                + model_fine_tuning_config["validation_data_path"]
            )

            instruction_tuned = model_fine_tuning_config["parameters"][
                "instruction_tuned"
            ]
            chat_dataset = model_fine_tuning_config["parameters"]["chat_dataset"]
            epoch = model_fine_tuning_config["parameters"]["epoch"]
            max_input_length = model_fine_tuning_config["parameters"][
                "max_input_length"
            ]
            finetune_instance_type = model_fine_tuning_config["parameters"][
                "instance_type"
            ]
            finetune_num_instances = model_fine_tuning_config["parameters"][
                "num_instances"
            ]

            training_job_name = step(
                finetune, name=f"finetune_{model_id}", keep_alive_period_in_seconds=2400
            )(
                model_id,
                endpoint_name,
                train_data_path,
                validation_data_path,
                finetune_num_instances,
                finetune_instance_type,
                instruction_tuned,
                chat_dataset,
                epoch,
                max_input_length,
            )

            endpoint_name = step(deploy_finetuned_model, name=f"deploy_{model_id}")(
                training_job_name,
                model_id,
                endpoint_name,
                deploy_instance_type,
                deploy_num_instances,
            )

        else:
            endpoint_name = step(deploy, name=f"deploy_{model_id}")(
                model_id,
                model_version,
                endpoint_name,
                deploy_instance_type,
                deploy_num_instances,
            )

        evaluation_results = step(
            evaluation,
            name=f"evaluation_{model_id}",
            keep_alive_period_in_seconds=1200,
            pre_execution_commands=[
                "pip install fmeval==0.2.0",
            ],
        )(
            processed_data_path,
            endpoint_name,
            dataset_config,
            model_config,
            algorithms_config,
            output_data_path,
        )
        evaluation_results_all.append(evaluation_results)

    best_model = step(selection, name="model_selection")(*evaluation_results_all)
    model_package_arn = step(register, name="best_model_registration")(
        best_model, "FMEvaluationBestModel", default_bucket
    )

    # Create cleanup steps
    last_pipeline_step = []
    for model_config in config["models"]:
        if model_config["cleanup_endpoint"]:
            model_id = model_config["model_id"]
            endpoint_name = model_config["endpoint_name"]
            cleanup_step = step(cleanup, name=f"cleanup_{model_id}")(
                model_id, endpoint_name
            )
            get_step(cleanup_step).add_depends_on([model_package_arn])
            last_pipeline_step.append(cleanup_step)

    if len(last_pipeline_step) == 0:
        last_pipeline_step.append(model_package_arn)

    # Define the Sagemaker Pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        steps=last_pipeline_step,
    )

    # Build and run the Sagemaker Pipeline
    pipeline.upsert(role_arn=sagemaker.get_execution_role())
    # pipeline.upsert(role_arn="arn:aws:iam::<...>:role/service-role/AmazonSageMaker-ExecutionRole-<...>")

    pipeline.start()
