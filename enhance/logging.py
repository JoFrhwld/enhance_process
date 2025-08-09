from fileinput import filename
from pathlib import Path
import logging



def make_loggers(name: str|None = None):
  logger = logging.getLogger(name) if name else logging.getLogger(__name__)
  logger.handlers.clear()
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  shandler = logging.StreamHandler()
  shandler.setFormatter(formatter)

  logger.addHandler(shandler)

  return logger

def make_file_handler(path: Path|str|None):
  path = Path(path) if path else Path("foo.log")
  formatter = logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s')
  filename = path.with_suffix(".log")

  fhandler = logging.FileHandler(filename=filename)
  fhandler.setFormatter(formatter)
  return fhandler
