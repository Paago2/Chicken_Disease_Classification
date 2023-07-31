from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

if __name__ == "__main__":
    # List of pipeline stages to be executed
    stages = [
        {
            "name": "Data Ingestion stage",
            "pipeline": DataIngestionTrainingPipeline,
        },
        # Add more stages here as needed
    ]

    for stage in stages:
        try:
            logger.info(f">>>>>> stage {stage['name']} started <<<<<<")
            pipeline_instance = stage['pipeline']()
            pipeline_instance.main()
            logger.info(
                f">>>>>> stage {stage['name']} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e
