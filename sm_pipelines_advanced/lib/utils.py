import boto3
import yaml

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


def find_model_by_name(model_list, model_name):
    for model in model_list:
        if model.config["name"] == model_name:
            return model

    return None


class ConfigParser:
    """
    Provides a simple interface for reading and validating a evaluation configuration file
    @param config_file: The path to the evaluation configuration file
    """
    REQUIRED_KEYS = {
        "models": ["model_id", "model_version"],
        "datasets": [],
        "evaluation": []
    }

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = {}
        self.load_config()
        self.validate_config()

    def load_config(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)
            return self.config

    # TODO: Implement validation function using yaml schema definition
    def validate_config(self):
        # pass
        # check if the config file has all the required keys
        # iterate over all keys and subkeys in required_configs and validate that they exist in self.config
        # for key, subkeys in self.REQUIRED_KEYS.items():
        #     if key not in self.config:
        #         raise ValueError(f"Config file is missing required key: {key}")
        #     for subkey in subkeys:
        #         if subkey not in self.config[key] and subkey:
        #             raise ValueError(f"Config file is missing required key: {subkey}")
        pass

    def __get__(self, key):
        return self.config[key]

    # return the config file as a dictionary
    def get_config(self):
        return self.config

    # return a specific key from the config file
    def get_config_key(self, key):
        return self.config[key]