from flask import Flask, render_template, request, redirect, flash, url_for
import pickle
import pandas as pd
import numpy as np
import os
import sys
from statistics import mean

from src.data_processing.Preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from src.bias.BiasDetector import BiasDetector
from src.bias.TotalVariationDistance import TotalVariationDistance
from src.bias.KLDivergence import KLDivergence

UPLOAD_FOLDER = os.path.abspath("uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#filename='df_pickle.pkl'

dict_vars = {}
agg_funcs = {
	'min' : min,
	'max' : max,
    'mean': mean 
}

dict_vars['hide_option']="d-none"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        dataframe_file = request.files['dataframe']
        model_file = request.files['model']
        predictions_file = request.files['predictions']
        
        if dataframe_file.filename == '':
            flash('No dataframe uploaded, try again', 'danger')
            return redirect('/')
        
        if model_file.filename == '' and predictions_file.filename == '':
            flash('No model or predictions uploaded, try again', 'danger')
            return redirect('/')

        if dataframe_file:
            dict_vars['df_filename'] = dataframe_file.filename
            dataframe_file.save(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['df_filename']))
        if model_file:
            dict_vars['model_filename'] = model_file.filename
            model_file.save(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['model_filename']))
        if predictions_file.filename:
            dict_vars['pred_filename'] = predictions_file.filename
            predictions_file.save(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['pred_filename']))

        if dataframe_file and (model_file or predictions_file):
            flash('Files uploaded successfully!', 'success')
        if dataframe_file and model_file and predictions_file:
            flash('Model and predictions uploaded! Predictions will be used', 'warning')
        return redirect('/')
    
    return render_template('home.html')

@app.route('/freqvsfreq', methods=['GET', 'POST'])
def freqvsfreq():
    fitted_ohe = pickle.load(open('../data/mlflow_artifacts/_ohe.pkl', 'rb'))
    fitted_scaler = pickle.load(open('../data/mlflow_artifacts/_scaler.pkl', 'rb'))

    pp = Preprocessing(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['df_filename']), "default")
    X, Y = pp.read_dataframe()
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=420)
    X_test_ohe, _, _ = pp.preprocess_for_classification(df=X_test, fit_ohe=True, 
                                                fitted_ohe=fitted_ohe,
                                                perform_scaling=True,
                                                fitted_scaler=fitted_scaler)

    if 'model_filename' in list(dict_vars.keys()):
        model_pickle = open(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['model_filename']), "rb")
        model = pickle.load(model_pickle)
        pred = model.predict(X_test_ohe)

    if 'pred_filename' in list(dict_vars.keys()):
        pred_pickle = open(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['pred_filename']), "rb")
        pred = pickle.load(pred_pickle)

    list_var = X_test.columns

    dict_vars['df_with_pred'] = pd.concat(
    [X_test.reset_index(drop=True), pd.Series(pred)], axis=1).rename(columns={0:"predictions"})
	
    if request.method == 'POST':
        '''
        rootvar = request.data.decode('utf-8')
        if len(dict_vars['df_with_pred'][rootvar].unique()) > 2:
            print("inside if")
            dict_vars['hide_option']= ""
        else:
            dict_vars['hide_option']= "d-none"
        '''
        if list(request.form.keys()):
            dict_vars['distance'] = request.form['distance']
            dict_vars['agg_func'] = request.form['agg_func']
            dict_vars['thr'] = request.form['Slider']
            dict_vars['root_var']= request.form['root_var']
            dict_vars['cond_vars']= request.form.getlist('cond_var')
            if dict_vars['root_var'] in dict_vars['cond_vars']:
                dict_vars['cond_vars'].remove(dict_vars['root_var'])
        return redirect('/freqvsfreq')
    return render_template('freqvsfreq.html', list1=list_var, hide=dict_vars['hide_option'])

@app.route('/freqvsfreq/results')
def results_fvf():
    
    d=TotalVariationDistance(aggregating_function=agg_funcs[dict_vars['agg_func']])
    if dict_vars['distance'] == 'KLDivergence':
        d=KLDivergence(aggregating_function=agg_funcs[dict_vars['agg_func']])
    bd=BiasDetector(distance=d)

    results1 = bd.compare_root_variable_groups(
		dataframe=dict_vars['df_with_pred'],
		target_variable='predictions',
		root_variable=dict_vars['root_var'],
		threshold=float(dict_vars['thr'])
	)
    results2 = bd.compare_root_variable_conditioned_groups(
		dataframe=dict_vars['df_with_pred'],
		target_variable='predictions',
		root_variable=dict_vars['root_var'],
		conditioning_variables=dict_vars['cond_vars'],
		threshold=float(dict_vars['thr']),
		min_obs_per_group=30
	)
    
    violations = {k: v for k, v in results2.items() if not v[2]}
    return render_template('results_freqvsfreq.html', results1=results1, results2=results2, violations=violations)

@app.route('/freqvsfreq/results/<violation>')
def details_fvf(violation):
    focus_df = dict_vars['df_with_pred'].query(violation)
    d=TotalVariationDistance(aggregating_function=agg_funcs[dict_vars['agg_func']])
    if dict_vars['distance'] == 'KLDivergence':
        d=KLDivergence(aggregating_function=agg_funcs[dict_vars['agg_func']])
    bd=BiasDetector(distance=d)
    
    results_viol1 = bd.get_frequencies_list(focus_df, 'predictions',
                            dict_vars['df_with_pred'].predictions.unique(),
                            dict_vars['root_var'],  dict_vars['df_with_pred'][dict_vars['root_var']].unique()
                            )
    results_viol2 = focus_df.groupby(dict_vars['root_var']).predictions.value_counts(normalize=True)
    return render_template('violation_specific_fvf.html', viol = violation, res2 = results_viol2.to_frame().to_html(classes=['table table-hover mx-auto w-75']))

@app.route('/freqvsref', methods=['GET', 'POST'])
def freqvsref():
    #df_pickle = open(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['df_filename']), "rb")
    #df_raw = pickle.load(df_pickle)

    fitted_ohe = pickle.load(open('../data/mlflow_artifacts/_ohe.pkl', 'rb'))
    fitted_scaler = pickle.load(open('../data/mlflow_artifacts/_scaler.pkl', 'rb'))

    pp = Preprocessing(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['df_filename']), "default")
    X, Y = pp.read_dataframe()
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=420)
    X_test_ohe, _, _ = pp.preprocess_for_classification(df=X_test, fit_ohe=True, 
                                                fitted_ohe=fitted_ohe,
                                                perform_scaling=True,
                                                fitted_scaler=fitted_scaler)
    if 'model_filename' in list(dict_vars.keys()):
        model_pickle = open(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['model_filename']), "rb")
        model = pickle.load(model_pickle)
        pred = model.predict(X_test_ohe)

    if 'pred_filename' in list(dict_vars.keys()):
        pred_pickle = open(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['pred_filename']), "rb")
        pred = pickle.load(pred_pickle)
    list_var = X_test.columns

    dict_vars['df_with_pred'] = pd.concat(
    [X_test.reset_index(drop=True), pd.Series(pred)], axis=1).rename(columns={0:"predictions"})
	
    if request.method == 'POST':
        dict_vars['distance'] = request.form['distance']
        dict_vars['agg_func'] = request.form['agg_func']
        dict_vars['thr'] = request.form['Slider']
        dict_vars['root_var']= request.form['root_var']
        dict_vars['cond_vars']= request.form.getlist('cond_var')
        if dict_vars['root_var'] in dict_vars['cond_vars']:
            dict_vars['cond_vars'].remove(dict_vars['root_var'])
        dict_vars['prob_a_0'] = float(request.form['prob_a_0'])
        dict_vars['prob_a_1'] = float(request.form['prob_a_1'])
        dict_vars['prob_b_0'] = float(request.form['prob_b_0'])
        dict_vars['prob_b_1'] = float(request.form['prob_b_1'])
        return redirect('/freqvsref')
    return render_template('freqvsref.html', list1=list_var)

@app.route('/freqvsref/results')
def results_fvr():
    
    d=TotalVariationDistance(aggregating_function=agg_funcs[dict_vars['agg_func']])
    if dict_vars['distance'] == 'KLDivergence':
        d=KLDivergence(aggregating_function=agg_funcs[dict_vars['agg_func']])
    bd=BiasDetector(distance=d)

    ref_distribution = [np.array([dict_vars['prob_a_0'], dict_vars['prob_a_1']]), np.array([dict_vars['prob_b_0'], dict_vars['prob_b_1']])]

    results1 = bd.compare_root_variable_groups(
		dataframe=dict_vars['df_with_pred'],
		target_variable='predictions',
		root_variable=dict_vars['root_var'],
		threshold=float(dict_vars['thr']),
        reference_distribution=ref_distribution
	)
    results2 = bd.compare_root_variable_conditioned_groups(
		dataframe=dict_vars['df_with_pred'],
		target_variable='predictions',
		root_variable=dict_vars['root_var'],
		conditioning_variables=dict_vars['cond_vars'],
		threshold=float(dict_vars['thr']),
		min_obs_per_group=30,
        reference_distribution=ref_distribution
	)
    
    violations = {k: v for k, v in results2.items() if (not v[2][0] or not v[2][1])}
    return render_template('results_freqvsref.html', results1=results1, results2=results2, violations=violations)

@app.route('/freqvsref/results/<violation>')
def details_fvr(violation):
    focus_df = dict_vars['df_with_pred'].query(violation)
    d=TotalVariationDistance(aggregating_function=agg_funcs[dict_vars['agg_func']])
    if dict_vars['distance'] == 'KLDivergence':
        d=KLDivergence(aggregating_function=agg_funcs[dict_vars['agg_func']])
    bd=BiasDetector(distance=d)
    
    results_viol1 = bd.get_frequencies_list(focus_df, 'predictions',
                            dict_vars['df_with_pred'].predictions.unique(),
                            dict_vars['root_var'],  dict_vars['df_with_pred'][dict_vars['root_var']].unique()
                            )
    results_viol2 = focus_df.groupby(dict_vars['root_var']).predictions.value_counts(normalize=True)
    return render_template('violation_specific_fvr.html', viol = violation, res2 = results_viol2.to_frame().to_html(classes=['table table-hover mx-auto w-75']))

