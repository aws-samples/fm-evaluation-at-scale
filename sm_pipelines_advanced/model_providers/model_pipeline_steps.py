from abc import ABC, abstractmethod
import boto3


class ModelPipelineSteps(ABC):
    def __init__(self, config):
        self.config = config

    def is_finetuning(self):
        if "finetuning_config" in self.config.keys():
            return True
        else:
            return False

    def endpoint_exists(endpoint_name):
        endpoint_exist = False

        client = boto3.client('sagemaker')
        response = client.list_endpoints()
        endpoints = response["Endpoints"]

        for endpoint in endpoints:
            if endpoint_name == endpoint["EndpointName"]:
                endpoint_exist = True
                break

        return endpoint_exist

    def get_deploy_step_name(self):
        return "deploy_"+self.config["name"]

    def get_finetune_step_name(self):
        return "finetune_"+self.config["name"]

    def get_deploy_finetuned_step_name(self):
        return "deploy_finetuned_"+self.config["name"]

    def get_register_step_name(self):
        return "register_"+self.config["name"]

    def get_cleanup_step_name(self):
        return "cleanup_"+self.config["name"]

    @staticmethod
    @abstractmethod
    def deploy_step(model, *args):
        pass

    @staticmethod
    @abstractmethod
    def finetune_step(model, *args):
        pass

    @staticmethod
    @abstractmethod
    def deploy_finetuned_step(model, finetune_step_ret, *args):
        pass

    @staticmethod
    @abstractmethod
    def register(model):
        pass

    @staticmethod
    @abstractmethod
    def cleanup_step(model, select_step_ret, *args):
        pass

    @staticmethod
    @abstractmethod
    def get_model_runner(model, content_template, *args):
        pass




