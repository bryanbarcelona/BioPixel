import logging

# List of loggers to suppress
LOGGERS_TO_SUPPRESS = [
    'czitools', 
    'czitools.utils.logger',
    'czitools.utils',
    'tdqm',
    'tqdm.cli',
]

def suppress_loggers():
    """
    Suppresses loggers specified in LOGGERS_TO_SUPPRESS by setting their level to CRITICAL and adding a NullHandler.
    """
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL)

    # Remove all existing handlers from the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add a NullHandler to the root logger
    root_logger.addHandler(logging.NullHandler())

    # Add NullHandlers to specific loggers to suppress their output
    for logger_name in LOGGERS_TO_SUPPRESS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.addHandler(logging.NullHandler())

# Automatically suppress loggers when this module is imported
suppress_loggers()


# import logging
# import sys
# import contextlib
# from tqdm import tqdm

# # Global flag to control tqdm progress display
# show_progress = False

# # List of loggers to suppress
# LOGGERS_TO_SUPPRESS = [
#     'czitools', 
#     'czitools.utils.logger',
#     'czitools.utils',
# ]

# # # List of tqdm instances to suppress
# # TQDM_INSTANCES_TO_SUPPRESS = []

# def suppress_loggers():
#     """
#     Suppresses loggers specified in LOGGERS_TO_SUPPRESS by setting their level to CRITICAL and adding a NullHandler.
#     """
#     # Configure the root logger
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.CRITICAL)

#     # Remove all existing handlers from the root logger
#     for handler in root_logger.handlers[:]:
#         root_logger.removeHandler(handler)

#     # Add a NullHandler to the root logger
#     root_logger.addHandler(logging.NullHandler())

#     # Add NullHandlers to specific loggers to suppress their output
#     for logger_name in LOGGERS_TO_SUPPRESS:
#         logger = logging.getLogger(logger_name)
#         logger.setLevel(logging.CRITICAL)
#         logger.addHandler(logging.NullHandler())

# def suppress_tqdm():
#     """
#     Suppresses tqdm instances specified in TQDM_INSTANCES_TO_SUPPRESS.
#     """
#     # Redefine tqdm.write() method to suppress output
#     original_write = tqdm.write

#     def tqdm_write(*args, **kwargs):
#         if not TQDM_INSTANCES_TO_SUPPRESS:
#             # If no tqdm instances to suppress, call original write
#             return original_write(*args, **kwargs)

#         caller_frame = sys._getframe(1)
#         caller_module = caller_frame.f_globals.get('__name__', '')
#         for instance in TQDM_INSTANCES_TO_SUPPRESS:
#             if isinstance(instance, tqdm):
#                 # Suppress output if the caller module matches the tqdm instance module
#                 if caller_module.startswith(instance.__module__):
#                     return

#         # Call original write if no suppression criteria match
#         return original_write(*args, **kwargs)

#     # Override tqdm.write() with the custom function
#     tqdm.write = tqdm_write

# # Automatically suppress loggers and tqdm instances when this module is imported
# suppress_loggers()
# suppress_tqdm()

# Example usage:
# with tqdm(total=100) as pbar:
#     for i in range(100):
#         pbar.update(1)
