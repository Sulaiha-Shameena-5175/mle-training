from pathlib import Path


def isPickleFileExist(filepath):
    path = Path(filepath)
    print(type(path))
    print(path.is_file())
    print(path.name.endswith(".pkl"))


isPickleFileExist("../artifacts/.gitkeep/lin_reg.pkl")
