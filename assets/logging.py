import logging
import platform
import torch

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
    get_environment_info()
    log_info("Configuration ...")
    log_info("Model:         {}".format(model_name))
    log_info("Dataset:       {}".format(args.dataset))
    log_info("Optimizer:     {}".format(args.optim))
    log_info("Learning rate: {}".format(args.lr))
    log_info("Batch size:    {}".format(args.batch_size))
    log_info("Seed:          {}".format(args.seed))
    log_info("Epochs:        {}".format(args.epochs))
    log_info("Val interval:  {}".format(args.val_interval))
    log_info("Save model:    {}".format(str(args.save_model))) 
    if subfolder:
        save_path = args.save_path + "/" + subfolder + "/"
        log_info("Save path:     {}".format(save_path))
    else:
        log_info("Save path:     {}".format(args.save_path))
    log_info("-"*50)

def finish_logging(time):
    log_info("Clear GPU cache ...")
    torch.cuda.empty_cache()
    log_info("Finish training")
    log_info("Time elapsed: {:.2f} seconds".format(time))

def get_environment_info():
    log_info("Environment information ...")
    log_info("Python version:   {}".format(platform.python_version()))
    log_info("PyTorch version:  {}".format(torch.__version__))
    log_info("CUDA available:   {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        log_info("CUDA version:     {}".format(torch.version.cuda))
        log_info("CUDA device:      {}".format(torch.cuda.get_device_name(0)))
    log_info("Platform:         {}".format(platform.platform()))
    log_info("System:           {}".format(platform.system()))
    log_info("Machine:          {}".format(platform.machine()))
    log_info("Version:          {}".format(platform.version()))
    log_info("User:             {}".format(platform.uname().node))
    log_info("-"*50)

def log_info(message):
    if message != "":
        logging.info(message)

def log_error(message):
    if message != "":
        logging.error(message)

def log_warning(message):
    if message != "":
        logging.warning(message)
