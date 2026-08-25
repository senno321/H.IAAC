import argparse
import ast
from pathlib import Path

import torch

from utils.model.factory import ModelFactory
from utils.model.manipulation import ModelPersistence
from utils.simulation.config import ConfigRepository, set_seed


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./pyproject.toml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--agg", type=str, default="")
    parser.add_argument("--sel", type=str, default="")
    parser.add_argument("--name", "--model-name", dest="name", type=str, default=None)
    parser.add_argument("--input-shape", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--root-model-dir", type=str, default=None)
    parser.add_argument("--norm", type=str, default=None, help="bn (default) | gn")
    args = parser.parse_args()

    # Read config simulation file and validate it
    config_repo = ConfigRepository(args.config_file)
    cfg = config_repo.preprocess_app_config(config_repo.get_app_config())
    config_repo.validate_app_config(cfg)

    # Using seed
    set_seed(args.seed)

    model_name = args.name if args.name is not None else cfg["model-name"]
    input_shape = args.input_shape if args.input_shape is not None else cfg["input-shape"]
    if isinstance(input_shape, str):
        input_shape = ast.literal_eval(input_shape)
    num_classes = int(args.num_classes) if args.num_classes is not None else cfg["num-classes"]
    selector_name = args.sel if args.sel else cfg["selection-name"]
    aggregator_name = args.agg if args.agg else cfg["aggregation-name"]
    norm = args.norm if args.norm is not None else cfg.get("norm-layer", "bn")

    # Creating a model. `norm` precisa casar com o norm usado no run (workflow.py),
    # senão o load_state_dict do .pth falha por chaves incompatíveis (BN vs GN).
    model = ModelFactory.create(model_name=model_name, norm=norm, input_shape=input_shape, num_classes=num_classes)

    # Saving (manager-style): <model>_<selection>_<aggregation>_<seed>.pth
    root_model_dir = Path(args.root_model_dir) if args.root_model_dir is not None else Path(cfg["root-model-dir"])
    root_model_dir.mkdir(parents=True, exist_ok=True)
    saving_path = root_model_dir / f"{model_name}_{selector_name}_{aggregator_name}_{args.seed}.pth"
    ModelPersistence.save(model, saving_path)

if __name__ == "__main__":
    main()