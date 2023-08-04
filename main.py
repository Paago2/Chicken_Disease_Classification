from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

if __name__ == "__main__":
    try:
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
