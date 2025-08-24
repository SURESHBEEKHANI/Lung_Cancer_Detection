from models.model_builder import ModelBuilder

class ReportService:
    """
    Service to handle medical image report generation.
    """

    def __init__(self, api_key: str = None):
        self.builder = ModelBuilder(api_key=api_key)

    def generate_report_from_image(self, image_path: str) -> str:
        """
        Generates a report from a local image.

        Args:
            image_path (str): Path to the medical image.

        Returns:
            str: Generated report text.
        """
        return self.builder.generate_report(image_path)
