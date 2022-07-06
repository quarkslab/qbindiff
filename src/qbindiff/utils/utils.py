import logging


def is_debug() -> bool:
    """Returns True if the current logging level is set to debug"""
    return logging.root.level <= logging.DEBUG
