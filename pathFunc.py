from pathlib import Path


def dataDir()-> Path:
    cwd = Path.cwd()
    dc = cwd.parent
    return dc.joinpath('data')

def dbPath()-> Path:
    cwd = Path.cwd()
    dc = cwd.parent
    dbPath = dc.joinpath("data/database_final.db")
    return dbPath


def figPath()-> Path:
    cwd = Path.cwd()
    dc = cwd.parent
    return dc.joinpath("data/figures/")

