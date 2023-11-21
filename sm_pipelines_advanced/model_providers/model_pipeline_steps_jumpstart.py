import sagemaker
from model_providers.model_pipeline_steps import ModelPipelineSteps
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker import get_execution_role
from sagemaker import image_uris, model_uris, Model, script_uris
from fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
from lib.utils import endpoint_exists
import boto3


class ModelPipelineStepsJumpStart(ModelPipelineSteps):

    def __init__(self, config):
        super().__init__(config=config)

    @staticmethod
    def deploy_step(model, *args):
        model_id = model.config["model_id"]
        endpoint_name = model.config["endpoint_name"]

        endpoint_exist = endpoint_exists(endpoint_name)

        if endpoint_exist:
            print("Endpoint already exists")
        else:
            my_model = JumpStartModel(model_id=model_id)
            predictor = my_model.deploy(initial_instance_count=model.config["deployment_config"]["num_instances"],
                                        instance_type=model.config["deployment_config"]["instance_type"],
                                        serializer=sagemaker.serializers.JSONSerializer(),
                                        deserializer=sagemaker.deserializers.JSONDeserializer(),
                                        endpoint_name=endpoint_name)

        return {"model_deployed": True}

    @staticmethod
    def finetune_step(model, *args):

        model_id = model.config["model_id"]
        model_version = model.config["model_version"]

        model_fine_tuning_config = model.config["finetuning_config"]
        train_data_path = model_fine_tuning_config["train_data_path"]
        validation_data_path = model_fine_tuning_config["validation_data_path"]
        epoch = model_fine_tuning_config["parameters"]["epoch"]
        max_input_length = model_fine_tuning_config["parameters"]["max_input_length"]
        instance_count = model_fine_tuning_config["parameters"]["num_instances"]
        instance_type = model_fine_tuning_config["parameters"]["instance_type"]
        instruction_tuned = model_fine_tuning_config["parameters"]["instruction_tuned"]
        chat_dataset = model_fine_tuning_config["parameters"]["chat_dataset"]

        estimator = JumpStartEstimator(
            model_id=model_id,
            model_version=model_version,
            instance_count=instance_count,
            instance_type=instance_type,
            environment={"accept_eula": "true"},
            disable_output_compression=True)  # For Llama-2-70b, add instance_type = "ml.g5.48xlarge"

        # By default, instruction tuning is set to false. Thus, to use instruction tuning dataset you use
        estimator.set_hyperparameters(instruction_tuned=instruction_tuned,
                                      epoch=epoch,
                                      max_input_length=max_input_length)
        estimator.fit({"training": train_data_path, "validation": validation_data_path})

        training_job_name = estimator.latest_training_job.name

        return {"training_job_name": training_job_name}

    @staticmethod
    def register(model):
        sagemaker_session = sagemaker.Session()
        role = get_execution_role()

        instance_type, instance_count = "ml.m5.xlarge", 1

        client = boto3.client('sagemaker')

        model_id = model.config['model_id']
        model_version = model.config['model_version']
        print(model_id, model_version)

        base_model_uri = model_uris.retrieve(
            model_id=model_id, model_version=model_version, model_scope="inference"
        )

        #script_uri = script_uris.retrieve(
        #    model_id=model_id, model_version=model_version, script_scope="inference"
        #)

        deploy_image_uri = image_uris.retrieve(
            region=None,
            framework=None,
            image_scope="inference",
            model_id=model_id,
            model_version=model_version,
            instance_type=instance_type,
        )

        model = Model(
            image_uri=deploy_image_uri,
            model_data=base_model_uri,
            # source_dir=script_uri,
            role=sagemaker.get_execution_role(),
            sagemaker_session=sagemaker_session
            # predictor_cls=Predictor,
        )

        modelpackage_inference_specification = {
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": deploy_image_uri,
                        "ModelDataUrl": base_model_uri
                    }
                ],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            }
        }

        create_model_package_input_dict = {
            "ModelPackageGroupName": model["register_config"]["model_package_group_name"],
            "ModelPackageDescription": "Model to detect 3 different types of irises (Setosa, Versicolour, and Virginica)",
            "ModelApprovalStatus": "PendingManualApproval"
        }
        create_model_package_input_dict.update(modelpackage_inference_specification)

        create_model_package_response = client.create_model_package(**create_model_package_input_dict)
        model_package_arn = create_model_package_response["ModelPackageArn"]
        print('ModelPackage Version ARN : {}'.format(model_package_arn))

        return model_package_arn

    @staticmethod
    def deploy_finetuned_step(model, finetune_step_ret, *args):
        model_id = model.config["model_id"]
        endpoint_name = model.config["endpoint_name"]
        training_job_name = finetune_step_ret["training_job_name"]

        endpoint_exist = endpoint_exists(endpoint_name)

        if endpoint_exist:
            print("Endpoint already exists")
        else:
            estimator = JumpStartEstimator.attach(training_job_name, model_id=model_id)
            estimator.logs()
            predictor = estimator.deploy(initial_instance_count=model.config["deployment_config"]["num_instances"],
                                         instance_type=model.config["deployment_config"]["instance_type"],
                                         serializer=sagemaker.serializers.JSONSerializer(),
                                         deserializer=sagemaker.deserializers.JSONDeserializer(),
                                         endpoint_name=endpoint_name)

        return {"model_deployed": True}

    @staticmethod
    def cleanup_step(model, *args):
        client = boto3.client('sagemaker')
        client.delete_endpoint(EndpointName=model.config["endpoint_name"])
        client.delete_endpoint_config(EndpointConfigName=model.config["endpoint_name"])
        return {"cleanup_done": True}

    @staticmethod
    def get_model_runner(model, content_template):

        endpoint_name = model.config["endpoint_name"]

        js_model_runner = JumpStartModelRunner(
            endpoint_name=endpoint_name,
            model_id=model.config["model_id"],
            model_version=model.config["model_version"],
            output=model.config["evaluation_config"]["output"],
            content_template=content_template,
            custom_attributes="accept_eula=true"
        )

        return js_model_runner

