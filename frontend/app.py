from flask import Flask, render_template, request, redirect, flash, url_for, session, Response, jsonify
import pickle
import pandas as pd
import numpy as np
import glob
import os
from subprocess import check_output
import sys
import subprocess
from statistics import mean, stdev

from src.utils.funcs import handle_multiupload, write_reference_distributions_html, handle_ref_distributions, allowed_file, order_violations
from src.utils.Preprocessing import Preprocessing

from sklearn.model_selection import train_test_split
from src.bias.threshold_calculator import threshold_calculator
from src.bias.BiasDetector import BiasDetector
from src.bias.FreqVsFreqBiasDetector import FreqVsFreqBiasDetector
from src.bias.FreqVsRefBiasDetector import FreqVsRefBiasDetector 

UPLOAD_FOLDER = os.path.abspath("uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dict_vars = {}
agg_funcs = {
	'min' : min,
	'max' : max,
    'mean': mean,
    'stdev': stdev
}

used_df = ""
comp_thr = ""
success_status = "text-warning"
ips = check_output(['hostname', '-I'])
localhost_ip = ips.decode().split(" ")[0]

@app.route('/', methods=['GET'])
def home():
    return render_template('homepage.html')

@app.route('/bias', methods=['GET', 'POST'])
def home_bias():
    global used_df
    global success_status
    global dict_vars
    if request.method == 'GET' and request.args.get('reset'):
        used_df = ""
        success_status = "text-warning"
        dict_vars = {}
        session.clear()
    if request.method == 'POST':
        keys = list(request.files.keys())
        uploads = [request.files[x].filename for x in keys if x != '']
        if all(x == '' for x in uploads):
            flash('No files uploaded, try again!', 'danger')
            return redirect('/bias')
        if 'dataset' in list(request.files.keys()):
            dataframe_file = request.files['dataset']
            dict_vars['dataset'] = dataframe_file.filename
            if allowed_file(dict_vars['dataset']):
                dataframe_file.save(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['dataset']))
                used_df = dict_vars['dataset']
                success_status = "text-success"
                flash('Dataframe uploaded successfully!', 'success')
            else:
                flash('Unsupported format for dataframe.', 'danger')
                return redirect('/bias')
        if ('dataset_custom' and 'notebook') in list(request.files.keys()):
            dict_vars['dataset_custom'] = request.files['dataset_custom'].filename
            dict_vars['notebook'] = request.files['notebook'].filename
            used_df = "Custom preprocessed " + dict_vars['dataset_custom']
            success_status = "text-success"
            if allowed_file(dict_vars['notebook']) and allowed_file(dict_vars['dataset_custom']):
                request.files['dataset_custom'].save(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['dataset_custom']))
                request.files['notebook'].save(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['notebook']))
            else:
                flash('Unsupported format for notebook or dataset.', 'danger')
                return redirect('/bias')
            if request.files['artifacts'].filename != '':
                handle_multiupload(request, 'artifacts', app.config['UPLOAD_FOLDER'])
            os.system("jupyter nbconvert --to python " + os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['notebook']))
            note_name = dict_vars['notebook'].split('.')[0]
            subprocess.run(["python3", os.path.join(app.config['UPLOAD_FOLDER'], note_name + ".py")])
            flash('Custom preprocessing pipeline uploaded and processed successfully!', 'success')
        return redirect('/bias')
    return render_template('bias/home.html', df_used=used_df, status=success_status)

@app.route('/bias/freqvsfreq', methods=['GET', 'POST'])
def freqvsfreq():
    global comp_thr
    if ('dataset_custom' and 'notebook') in list(dict_vars.keys()):
        list_of_files =  glob.glob(os.path.join(app.config['UPLOAD_FOLDER']) + "/*")
        latest_file = max(list_of_files, key=os.path.getctime)
        extension = latest_file.rsplit('.', 1)[1].lower()
        match extension:
            case 'pkl':
                df_pickle = open(latest_file, "rb")
                dict_vars['df'] = pickle.load(df_pickle)
            case 'csv':
                dict_vars['df'] = pd.read_csv(latest_file)
    if 'dataset' in list(dict_vars.keys()):
        df_uploaded = os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['dataset'])
        extension = df_uploaded.rsplit('.', 1)[1].lower()
        match extension:
            case 'pkl':
                df_pickle = open(df_uploaded, "rb")
                dict_vars['df'] = pickle.load(df_pickle)
            case 'csv':
                dict_vars['df'] = pd.read_csv(df_uploaded)
    
    list_var = dict_vars['df'].columns
    if request.method == 'POST':
        if list(request.form.keys()):
            if 'rv_selected' in list(request.form.keys()):
                html = open("templates/bias/stdev.html", "r")
                btn_opt = html.read()
                html.close()
                split_html = btn_opt.split('class="')
                rvar = request.form['rv_selected']
                hide_option = ""
                if len(dict_vars['df'][rvar].unique()) < 3:
                    # hide_option = "d-none "
                    return {'response': 'True'}
                return {'response': 'False'}
                # return {'response' : split_html[0] + 'class="' + hide_option + split_html[1] }
            dict_vars['root_var']= request.form['root_var']
            dict_vars['distance'] = request.form['distance']
            dict_vars['predictions'] = request.form['predictions']
            if 'agg_func' in list(request.form.keys()):
                dict_vars['agg_func'] = request.form['agg_func']
            if float(request.form['Slider']) > 0:
                dict_vars['thr'] = float(request.form['Slider'])
            else:
                dict_vars['thr'] = None
            dict_vars['cond_vars']= request.form.getlist('cond_var')
            dict_vars['a1_param'] = "high" 
            if 'auto_thr' in list(request.form.keys()):
                if request.form['auto_thr'] == 'active':
                    if 'a1_param' in list(request.form.keys()):
                        dict_vars['a1_param'] = request.form['a1_param']
                        dict_vars['thr'] = None
        return redirect('/bias/freqvsfreq')
    return render_template('bias/freqvsfreq.html', var_list=list_var, local_ip=localhost_ip) 

@app.route('/bias/freqvsfreq/results', methods=['GET', 'POST'])
def results_fvf():
    bd=FreqVsFreqBiasDetector(distance=dict_vars['distance']
                              )

    results1 = bd.compare_root_variable_groups(
		dataframe=dict_vars['df'],
		target_variable=dict_vars['predictions'],
		root_variable=dict_vars['root_var'],
        threshold=dict_vars['thr']
	)
    results2 = bd.compare_root_variable_conditioned_groups(
		dataframe=dict_vars['df'],
		target_variable=dict_vars['predictions'],
		root_variable=dict_vars['root_var'],
		conditioning_variables=dict_vars['cond_vars'],
        threshold=dict_vars['thr'],
		min_obs_per_group=30
	)
    violations = {k: v for k, v in results2.items() if not v[2]}

    if request.method == "POST":
        x = request.json.get('export-data', False)
        csv_data = "condition,num_observations,distance,distance_gt_threshold,threshold,standard_deviation\n"
        for key in list(results2.keys()):
            if len(results2[key]) == 3:
                csv_data += f"{key},{results2[key][0]},{results2[key][1]},{results2[key][2]}\n"
                continue
            csv_data += f"{key},{results2[key][0]},{results2[key][1]},{results2[key][2]},{results2[key][3]},{results2[key][4]}\n"
        # Create a Response with CSV data
        return jsonify({"csv_data": csv_data})

    return render_template('bias/results_freqvsfreq.html', results1=results1, results2=results2, violations=order_violations(violations), local_ip=localhost_ip)

@app.route('/bias/freqvsfreq/results/<violation>')
def details_fvf(violation):
    focus_df = dict_vars['df'].query(violation)
    bd_general=BiasDetector()
    
    results_viol1 = bd_general.get_frequencies_list(focus_df, dict_vars['predictions'],
                            dict_vars['df'][dict_vars['predictions']].unique(),
                            dict_vars['root_var'],  dict_vars['df'][dict_vars['root_var']].unique()
                            )
    results_viol2 = focus_df.groupby(dict_vars['root_var'])[dict_vars['predictions']].value_counts(normalize=True)
    return render_template('bias/violation_specific_fvf.html', viol = violation, res2 = results_viol2.to_frame().to_html(classes=['table table-hover mx-auto w-75']))

@app.route('/bias/freqvsref', methods=['GET', 'POST'])
def freqvsref():
    global comp_thr
    if ('dataset_custom' and 'notebook') in list(dict_vars.keys()):
        list_of_files =  glob.glob(os.path.join(app.config['UPLOAD_FOLDER']) + "/*")
        latest_file = max(list_of_files, key=os.path.getctime)
        extension = latest_file.rsplit('.', 1)[1].lower()
        match extension:
            case 'pkl':
                df_pickle = open(latest_file, "rb")
                dict_vars['df'] = pickle.load(df_pickle)
            case 'csv':
                dict_vars['df'] = pd.read_csv(latest_file)
    if 'dataset' in list(dict_vars.keys()):
        df_uploaded = os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['dataset'])
        extension = df_uploaded.rsplit('.', 1)[1].lower()
        match extension:
            case 'pkl':
                df_pickle = open(df_uploaded, "rb")
                dict_vars['df'] = pickle.load(df_pickle)
            case 'csv':
                dict_vars['df'] = pd.read_csv(df_uploaded)
    list_var = dict_vars['df'].columns
    if request.method == 'POST':
        if list(request.form.keys()):
            if ('pr_selected' and 'rv_selected') in list(request.form.keys()):
                html = open("templates/bias/stdev.html", "r")
                btn_opt = html.read()
                html.close()
                split_html = btn_opt.split('class="')
                rvar = request.form['rv_selected']
                print(rvar, flush=True)
                hide_option = ""
                # if len(dict_vars['df'][rvar].unique()) < 3:
                #     hide_option = "d-none "
                pvar = request.form['pr_selected']
                return {'response_refs' : write_reference_distributions_html(rvar, pvar, dict_vars['df'])}
            dict_vars['root_var']= request.form['root_var']
            dict_vars['predictions'] = request.form['predictions']
            dict_vars['distance'] = request.form['distance']
            if float(request.form['Slider']) > 0:
                dict_vars['thr'] = float(request.form['Slider'])
            else:
                dict_vars['thr'] = None
            if 'auto_thr' in list(request.form.keys()):
                if request.form['auto_thr'] == 'active':
                    if 'a1_param' in list(request.form.keys()):
                        dict_vars['a1_param'] = request.form['a1_param']
                        dict_vars['thr'] = None
            dict_vars['cond_vars']= request.form.getlist('cond_var')
            nroot = len(dict_vars['df'][dict_vars['root_var']].unique())
            ntarget = len(dict_vars['df'][dict_vars['predictions']].unique())
            for i in range(nroot):
                for j in range(ntarget):
                    cat = f'prob_{i}_{j}'
                    dict_vars[cat] = float(request.form[cat])
        return redirect('/bias/freqvsref')
    return render_template('bias/freqvsref.html', var_list=list_var, local_ip=localhost_ip) 

@app.route('/bias/freqvsref/results', methods = ['GET', 'POST'])
def results_fvr():
    
    bd=FreqVsRefBiasDetector(A1=dict_vars['a1_param'])

    ref_distribution = handle_ref_distributions(dict_vars['root_var'], dict_vars['predictions'], dict_vars['df'], dict_vars)

    results1 = bd.compare_root_variable_groups(
		dataframe=dict_vars['df'],
		target_variable=dict_vars['predictions'],
		root_variable=dict_vars['root_var'],
        reference_distribution=ref_distribution,
		threshold=dict_vars['thr']
	)
    results2 = bd.compare_root_variable_conditioned_groups(
		dataframe=dict_vars['df'],
		target_variable=dict_vars['predictions'],
		root_variable=dict_vars['root_var'],
		conditioning_variables=dict_vars['cond_vars'],
        reference_distribution=ref_distribution,
		threshold=dict_vars['thr'],
		min_obs_per_group=30
	)
    
    violations = {k: v for k, v in results2.items() if (not v[2][0] or not v[2][1])}
    print(results1, flush=True)
    print(results2, flush=True)
    print(violations, flush=True)
    if request.method == "POST":
        x = request.json.get('export-data', False)
        csv_data = "condition;num_observations;distance;distance_gt_threshold;threshold\n"
        for key in list(results2.keys()):
            if len(results2[key]) == 3:
                csv_data += f"{key};{results2[key][0]};{results2[key][1]};{results2[key][2]}\n"
                continue
            print(results2[key][1], flush=True)
            csv_data += f"{key};{results2[key][0]};{results2[key][1]};{results2[key][2]};{results2[key][3]}\n"
        # Create a Response with CSV data
        return jsonify({"csv_data": csv_data})
    return render_template('bias/results_freqvsref.html', results1=results1, results2=results2, violations=order_violations(violations), local_ip=localhost_ip)

@app.route('/bias/freqvsref/results/<violation>')
def details_fvr(violation):
    focus_df = dict_vars['df'].query(violation)
    bd_general=BiasDetector()
    
    results_viol1 = bd_general.get_frequencies_list(focus_df, 'predictions',
                            dict_vars['df'][dict_vars['predictions']].unique(),
                            dict_vars['root_var'],  dict_vars['df'][dict_vars['root_var']].unique()
                            )
    results_viol2 = focus_df.groupby(dict_vars['root_var'])[dict_vars['predictions']].value_counts(normalize=True)
    return render_template('bias/violation_specific_fvr.html', viol = violation, res2 = results_viol2.to_frame().to_html(classes=['table table-hover mx-auto w-75']))
