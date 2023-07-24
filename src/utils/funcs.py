from flask import request
import pandas as pd
import numpy as np
import os

def allowed_file(filename: str) -> str:
    ALLOWED_EXTENSIONS = {'pkl', 'csv', 'ipynb', 'py'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_multiupload(req: request, label: str, path: str) -> None:
    files_list = req.files.getlist(label)
    for file in files_list:
        name = file.filename
        save_path = os.path.join(path, name)
        file.save(save_path)

def handle_ref_distributions(rootvar: str, targetvar: str, df : pd.DataFrame, dict_vars: dict) -> list:
    nroot = len(df[rootvar].unique())
    ntarget = len(df[targetvar].unique())
    final_list = []
    intermediate_list = []
    for i in range(nroot):
        for j in range(ntarget):
            intermediate_list.append(dict_vars[f'prob_{i}_{j}'])
        final_list.append(np.array(intermediate_list))
        intermediate_list = []
    return final_list


def write_reference_distributions_html(rootvar: str, targetvar: str, df: pd.DataFrame) -> str:
    nroot = len(df[rootvar].unique())
    ntarget = len(df[targetvar].unique())
    tot_refs = nroot * ntarget
    begin = '<div class="row" id="ref_dist">'
    end = '</div>'
    tot_html = ""
    nrows = tot_refs // 4
    if nrows == 0:
        nrows = 1
    ncols = tot_refs % 4
    c = 0
    d = 0
    for n in range(nrows):
        tot_html += f'<div class="row" id="ref_dist{n}">'
        for j in range(tot_refs):
            if j != 0 and j % 4 == 0:
                break
            tot_refs -= 1
            tot_html += '<div class="col-3">'
            tot_html += f'<input type="number" class="form-control" placeholder="rv{c}_tg{d}_ref" name="prob_{c}_{d}" id="prob_{c}_{d}" min="0" max="1" step=".01">'
            d += 1
            if d == ntarget:
                d = 0
                c += 1
            tot_html += '</div>'
        tot_html += '</div><br>'
    return tot_html
