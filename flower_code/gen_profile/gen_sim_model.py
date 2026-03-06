import argparse
from ast import literal_eval

from utils.model.factory import ModelFactory
from utils.model.manipulation import ModelPersistence
from utils.simulation.config import ConfigRepository, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./pyproject.toml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-shape", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--root-model-dir", type=str, default=None)
    args = parser.parse_args()

    # Read config simulation file and validate it
    config_repo = ConfigRepository(args.config_file)
    cfg = config_repo.preprocess_app_config(config_repo.get_app_config())
    config_repo.validate_app_config(cfg)

    # Using seed
    set_seed(args.seed)

    model_name = args.model_name if args.model_name is not None else cfg["model-name"]
    input_shape = literal_eval(args.input_shape) if args.input_shape is not None else cfg["input-shape"]
    num_classes = int(args.num_classes) if args.num_classes is not None else cfg["num-classes"]

    # Creating a model
    model = ModelFactory.create(model_name=model_name, input_shape=input_shape, num_classes=num_classes)

    # Saving
    root_model_dir = args.root_model_dir if args.root_model_dir is not None else cfg["root-model-dir"]
    saving_path = root_model_dir + model_name + '.pth'
    ModelPersistence.save(model, saving_path)

if __name__ == "__main__":
    main()