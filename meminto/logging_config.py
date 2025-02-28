import logging
import time

LOG_FILE = "app.log"

def configure_logging():
    with open(LOG_FILE, "w"):
        pass

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] %(name)s : [%(levelname)s] %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def log_time(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)  # Use the module-level logger

        start_time = time.time()
        output = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Log the execution time
        logger.info(
            f"Finished {func.__name__!r} in {execution_time:.2f}s"
        )
        return output

    return wrapper