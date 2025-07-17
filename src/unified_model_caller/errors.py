class InvalidServiceError(Exception):
    """Error related to the invalid choice of the service"""
    def __init__(self, message: str):
        super().__init__(message)

class InvalidModelError(Exception):
    """Error related to the invalid choice of the model"""
    def __init__(self, message: str):
        super().__init__(message)

class ApiCallError(Exception):
    """Error related to the calls to models via API"""
    def __init__(self, message: str):
        super().__init__(message)

