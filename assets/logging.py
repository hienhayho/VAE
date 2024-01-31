import logging
import colorlog
from .date import get_current_date

# Format example: 2018-07-11 20:12:06 - Admin logged in
def configure_logging(filename):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)

    # Create a stream handler (for console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('[%(levelname)s] - %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Set the formatter for both handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def initial_logging(model_name, args, subfolder = None):
    log_info("Model: {}".format(model_name))
    log_info("Dataset: {}".format(args.dataset))
    log_info("Optimizer: {}".format(args.optim))
    log_info("Learning rate: {}".format(args.lr))
    log_info("Batch size: {}".format(args.batch_size))
    log_info("Epochs: {}".format(args.epochs))
    log_info("Val interval: {}".format(args.val_interval))
    log_info("Save model: {}".format(args.save_model)) 
    if subfolder:
        save_path = args.save_path + "/" + subfolder + "/"
        log_info("Save path: {}".format(save_path))
    else:
        log_info("Save path: {}".format(args.save_path))

def log_info(message):
    if message != "":
        logging.info(message)

def log_error(message):
    logging.error(message)

def log_warning(message):
    logging.warning(message)
