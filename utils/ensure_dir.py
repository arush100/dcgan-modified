import os
def ensure_dir(x):
    try:
        os.mkdir(x)
    except FileExistsError:
        pass
    return x
