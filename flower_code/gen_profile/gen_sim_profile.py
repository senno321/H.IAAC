import argparse
import json

from utils.profile.client_profiles import create_profiles
from utils.simulation.config import ConfigRepository, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./pyproject.toml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Read config simulation file and validate it
    config_repo = ConfigRepository(args.config_file)
    cfg = config_repo.get_app_config()
    cfg = config_repo.preprocess_app_config(cfg)
    config_repo.validate_app_config(cfg)

    # Using seed
    set_seed(args.seed)

    # creating profiles
    devices_profile_path = cfg["devices-profile-path"]
    num_clients = cfg["num-clients"]
    seed = args.seed
    prefer_time = cfg["prefer-time"]

    profiles = create_profiles(num_clients, seed, devices_profile_path, prefer_time)

    pro_files = cfg["root-profiles-dir"] + f"profiles.json"

    with open(pro_files, "w") as file:
        json.dump(profiles, file, indent=2)


if __name__ == '__main__':
    main()