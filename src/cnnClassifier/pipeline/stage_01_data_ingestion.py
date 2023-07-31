from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # The data ingestion process, including file downloading and extraction, is executed here
        train_df, test_df, valid_df = data_ingestion.execute()

        # Print a few lines of each DataFrame for verification
        print("Train DataFrame:")
        print(train_df.head())
        print("\nTest DataFrame:")
        print(test_df.head())
        print("\nValidation DataFrame:")
        print(valid_df.head())


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
