import argparse
import os
import yaml
import shutil
from src.engine import Engine
from src.utils.utils import apply_logger_configs
import logging


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="./logs/temp",
        help="Directory to save config and model checkpoint",
    )
    parser.add_argument(
        "--config_path",
        default="./configs/default.yml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="indicates whether we are only testing",
    )
    parser.add_argument(
        "--sweep",
        default=False,
        action="store_true",
        help="indicates whether this is a sweep run",
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create the directory if it does not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Copy the provided config file into save_dir
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))

    logger = apply_logger_configs(save_dir=args.save_dir)

    # Create the engine taking care of building different components and starting training/inference
    engine = Engine(
        config=config,
        save_dir=args.save_dir,
        logger=logger,
        train=not args.test,
        sweep=args.sweep,
    )

    if args.test:
        engine.evaluate()
    else:
        engine.train_model()


if __name__ == "__main__":
    run()
