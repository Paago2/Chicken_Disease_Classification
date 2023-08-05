from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def prepare_model(self):
        # Get configuration for base model preparation
        prepare_base_model_config = self.config_manager.get_prepare_base_model_config()

        # Initialize PrepareBaseModel with the configuration
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        # Get and save the base model
        prepare_base_model.get_base_model()

        # Update and save the base model
        prepare_base_model.update_base_model()

    def run(self):
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        try:
            self.prepare_model()
            logger.info(
                f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == '__main__':
    pipeline = PrepareBaseModelTrainingPipeline()
    pipeline.run()
