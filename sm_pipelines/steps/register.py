import json

import s3fs as s3fs
import boto3
import sagemaker
from sagemaker import ModelMetrics, MetricsSource
from sagemaker.s3_utils import s3_path_join
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.utils import unique_name_from_base
from sagemaker import image_uris, model_uris, Model, script_uris
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.predictor import Predictor
from sagemaker import get_execution_role
from sagemaker import ModelPackage


def register(
    evaluation,
    model_approval_status,
    model_package_group_name,
    bucket,
):
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()
    model_config = evaluation['model_config']
    eval_result = evaluation['eval_result']

    instance_type, instance_count = "ml.m5.xlarge", 1

    client = boto3.client('sagemaker')

    model_id = model_config['model_id']
    model_version = model_config['model_version']
    print(model_id, model_version)

    base_model_uri = model_uris.retrieve(
    model_id=model_id, model_version=model_version, model_scope="inference"
    )

    script_uri = script_uris.retrieve(
    model_id=model_id, model_version=model_version, script_scope="inference"
    )   
    # model_package_group = ModelPackageGroup(
    # model_package_group_name, 
    # sagemaker_session, 
    # description="Your description"
    # )
    # try:
    #     model_package_group.describe()
    # except:
    #     model_package_group.create()

    # # Step 3: Create a SageMaker Model Package
    # model_package = ModelPackage(
    #     role=role,
    #     model_package_group_name=model_package_group_name,
    #     model_package_name=model_id,
    #     model_package_version=model_version,
    #     # Add other necessary parameters
    # )   

    # model_package.create()
   
    deploy_image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        image_scope="inference",
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
    )

    # # model = Model(
    # #     image_uri=deploy_image_uri,
    # #     model_data=model_uri,
    # #     name="JumpStartRegisterModel",
    # # )

    model = Model(
        image_uri=deploy_image_uri,
        model_data=base_model_uri,
        # source_dir=script_uri,
        role=sagemaker.get_execution_role(),
        sagemaker_session=sagemaker_session
        # predictor_cls=Predictor,
    )
    
    # Upload evaluation report to s3
    # eval_file_name = unique_name_from_base("evaluation")
    # eval_report_s3_uri = s3_path_join(
    #     "s3://",
    #     bucket,
    #     model_package_group_name,
    #     f"evaluation-report/{eval_file_name}.json",
    # )
    # s3_fs = s3fs.S3FileSystem()
    # # TODO: remove hard coding 
    # eval_report_str = json.dumps({"score": .5})
    # with s3_fs.open(eval_report_s3_uri, "wb") as file:
    #     file.write(eval_report_str.encode("utf-8"))

    # # Create model_metrics as per evaluation report in s3
    # model_metrics = ModelMetrics(
    #     model_statistics=MetricsSource(
    #         s3_uri=eval_report_s3_uri,
    #         content_type="application/json",
    #     )
    # )

    # Build the trained model and register it
    # model_builder = ModelBuilder(
    #     model=model,
    #     image_uri=deploy_image_uri,
    #     role_arn=sagemaker.get_execution_role(),
    #     s3_model_data_url=s3_path_join(
    #         "s3://", bucket, model_package_group_name, "model-artifacts"
    #     ),
    # )
    # # Notes: There will be further improvements on the register method,
    # # such as automatically filling in the content_types and response_types parameters.
    # model_package = model_builder.build().register(
    #     content_types=["application/json"],
    #     response_types=["application/json"],
    #     model_package_group_name=model_package_group_name,
    #     approval_status=model_approval_status,
    #     model_metrics=model_metrics,
    # )
    # model = JumpStartModel(model_id=model_id)
    # model_package = model.register(
        
    #     content_types=["application/json"],
    #     response_types=["application/json"],
    #     model_package_group_name=model_package_group_name,
    #     approval_status=model_approval_status,
    #     # model_metrics=model_metrics,
    # )
    modelpackage_inference_specification =  {
        "InferenceSpecification": {
        "Containers": [
            {
                "Image": deploy_image_uri,
                "ModelDataUrl": base_model_uri
            }
        ],
        "SupportedContentTypes": [ "text/csv" ],
        "SupportedResponseMIMETypes": [ "text/csv" ],
    }
    }

    create_model_package_input_dict = {
    "ModelPackageGroupName" : model_package_group_name,
    "ModelPackageDescription" : "Model to detect 3 different types of irises (Setosa, Versicolour, and Virginica)",
    "ModelApprovalStatus" : "PendingManualApproval"
}
    create_model_package_input_dict.update(modelpackage_inference_specification)

    create_model_package_response = client.create_model_package(**create_model_package_input_dict)
    model_package_arn = create_model_package_response["ModelPackageArn"]
    print('ModelPackage Version ARN : {}'.format(model_package_arn))

    return model_package_arn
