import zipfile
import patoolib


def extract_archive(file_path, extract_to):
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.endswith(".rar"):
        patoolib.extract_archive(file_path, outdir=extract_to)
