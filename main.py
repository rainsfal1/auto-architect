import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/nas'))
import logging
import warnings
from nni.nas.experiment import NasExperiment
from nni.nas.experiment.config import NasExperimentConfig
import nni.nas.strategy as Strategy
from src.models.ner_model import get_model_space
from src.config import NAS_CONFIG
from nni.nas.evaluator import FunctionalEvaluator
import nni
import json
import numpy as np
from  processing import create_data_loaders,get_data_loaders
from training import evaluate_model


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")


def force_print(message):
    print(message, flush=True)
    logger.debug(message)


# Define the label map


def main():
    force_print("Starting main function")

    try:
        num_labels=9

        create_data_loaders(batch_size=32)
        # print("yes")
        
        force_print("Data loaders created successfully")

        force_print("Loading search space...")
        with open('src/nas/search_space.json', 'r') as f:
            search_space = json.load(f)
        force_print(f"Loaded search space: {json.dumps(search_space, indent=2)}")
        force_print("Getting model space...")
        model_space = get_model_space(num_labels, search_space)
        force_print(f"Model space: {model_space}")

        force_print("Initializing strategy...")
        strategy = Strategy.PolicyBasedRL()
        force_print("Strategy initialized")

        
        evaluator = FunctionalEvaluator(evaluate_model)
        force_print("Evaluator set up")

        force_print("Configuring NAS experiment...")
        config = NasExperimentConfig(
            experiment_name=NAS_CONFIG['experiment_name'],
            trial_concurrency=NAS_CONFIG['trial_concurrency'],
            max_trial_number=NAS_CONFIG['max_trial_number'],
            max_experiment_duration=NAS_CONFIG['max_experiment_duration'],
            trial_code_directory=NAS_CONFIG['trial_code_directory'],
            execution_engine='sequential',
            model_format='raw'
        )
        force_print(f"NAS experiment config: {config}")

        force_print("Starting NAS experiment...")
        exp = NasExperiment(
            model_space=model_space,
            strategy=strategy,
            evaluator=evaluator,
            config=config
        )
        force_print("NAS experiment initialized")

        force_print("Running NAS experiment...")
        exp.run(wait_completion=True)
        force_print("NAS experiment run completed")
        
        force_print("Exporting top models...")
        force_print('Best architecture:')
        for model_dict in exp.export_top_models(formatter='dict'):
            print(model_dict)

    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        raise


if __name__ == '__main__':
    main()
