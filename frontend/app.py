from flask import Flask, render_template, request, redirect, flash
import pickle
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.abspath("uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'pkl'}

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#filename='df_pickle.pkl'

dict_vars = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        dataframe_file = request.files['dataframe']
        model_file = request.files['model']
        predictions_file = request.files['predictions']
        
        #TODO fix error messages
        if dataframe_file.filename == '':
            flash('No dataframe uploaded, try again')
            return redirect('/')
        '''
        if  model_file.filename == '':
            flash('No model uploaded, try again')
            return redirect('/')

        if predictions_file.filename == '':
            flash('No predictions uploaded, try again')
            return redirect('/')
        '''
        #TODO: scrivi il codice per manipolare i file
        if dataframe_file:
            dict_vars['df_filename'] = dataframe_file.filename
            #filename = secure_filename(dataframe_file.filename)
            dataframe_file.save(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['df_filename']))
        return redirect('/')
    
    return render_template('home.html')

@app.route('/freqvsfreq', methods=['GET', 'POST'])
def freqvsfreq():
    df_pickle = open(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['df_filename']), "rb")
    df = pickle.load(df_pickle)
    list_var = df.columns[:5]
    if request.method == 'POST':
        thr = request.form['Slider']
        root_var = request.form['root_var']
        cond_vars_raw = request.form['mytext']
        cond_vars = cond_vars_raw.split()
        cond_vars_final = []
        for item in cond_vars:
            if item not in cond_vars_final:
                cond_vars_final.append(item)
        with open ('vars_freq.txt', 'w') as f:
            f.write(thr)
            f.write('\n')
            f.write(root_var)
            f.write('\n')
            f.write(','.join(str(x) for x in cond_vars_final))
        return redirect('/freqvsfreq')
    return render_template('freqvsfreq.html', list1=list_var)

@app.route('/freqvsfreq/results')
def results_fvf():
    threshold = 0
    root_var = ''
    cond_vars = []
    with open ('vars_freq.txt') as f:
        threshold = f.readline()
        root_var = f.readline()
        cond_vars = f.readline().split()
    return render_template('results_freqvsfreq.html', threshold=threshold, root_var=root_var, cond_vars=cond_vars) 

@app.route('/freqvsref', methods=['GET', 'POST'])
def freqvsref():
    df_pickle = open(os.path.join(app.config['UPLOAD_FOLDER'], dict_vars['df_filename']), "rb")
    df = pickle.load(df_pickle)
    list_var = df.columns[:5]
    if request.method == 'POST':
        thr = request.form['Slider']
        root_var = request.form['root_var']
        cond_vars_raw = request.form['mytext']
        cond_vars = cond_vars_raw.split()
        cond_vars_final = []
        for item in cond_vars:
            if item not in cond_vars_final:
                cond_vars_final.append(item)
        with open ('vars_ref.txt', 'w') as f:
            f.write(thr)
            f.write('\n')
            f.write(root_var)
            f.write('\n')
            f.write(','.join(str(x) for x in cond_vars_final))
        return redirect('/freqvsref')
    
    return render_template('freqvsref.html', list1=list_var)

@app.route('/freqvsref/results')
def results_fvr():
    return "<p>Results here!</p>"



