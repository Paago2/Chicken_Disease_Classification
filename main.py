from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

if __name__ == "__main__":
    try:
        # Run data ingestion pipeline
        logger.info(f">>>>>> stage Data Ingestion started <<<<<<")
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        train_df, test_df, valid_df, train_gen, valid_gen, test_gen, test_batch_size, test_steps = data_ingestion_pipeline.main()
        logger.info(
            f">>>>>> stage Data Ingestion completed <<<<<<\n\nx==========x")

        # Run prepare base model pipeline
        logger.info(f">>>>>> stage Prepare Base Model started <<<<<<")
        prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
        prepare_base_model_pipeline.run()
        logger.info(
            f">>>>>> stage Prepare Base Model completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
