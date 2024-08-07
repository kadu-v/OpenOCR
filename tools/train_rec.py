import datetime
import os
import sys

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from tools.engine import Config, Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--eval",
        action="store_true",
        default=True,
        help="Whether to perform evaluation in train",
    )
    args = parser.parse_args()
    return args


def get_now_string():
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    return now.strftime("%Y-%m-%d-%H%M%S")


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop("opt")
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)

    ############################## MLFlow Configurations ######################
    mlflow_run_name = cfg.cfg["MLFlow"]["mlflow_run_name"] + "-" + get_now_string()
    model_type = cfg.cfg["MLFlow"]["model_type"]
    experiment_name = cfg.cfg["MLFlow"]["experiment_name"]
    mlflow.pytorch.autolog(log_datasets=False)
    mlflow.set_experiment(experiment_name)
    mlflow.set_tag(MLFLOW_RUN_NAME, mlflow_run_name)
    mlflow.set_tag("model-type", model_type)

    with open("config.txt", "w") as f:
        f.write(str(cfg.cfg))
    mlflow.log_artifact("config.txt")

    ############################## Training ###################################
    trainer = Trainer(cfg, mode="train_eval" if FLAGS["eval"] else "train")
    trainer.train()


if __name__ == "__main__":
    main()
