from pathlib import Path
__version_info__ = (0, 1, 0)
__version__ = ".".join(str(vi) for vi in __version_info__)
data_path = Path(Path(__file__).parent, "bootstrap_data").resolve()
