import logging

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
file_handler=logging.FileHandler('error.txt')
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.ERROR)

# Create formatter
formater = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Attach formatter to handler
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

# Add handler to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
