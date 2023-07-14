from flask import request
import os

def handle_multiupload(req: request, label: str, path: str) -> None:
    files_list = req.files.getlist(label)
    for file in files_list:
        name = file.filename
        save_path = os.path.join(path, name)
        file.save(save_path)
