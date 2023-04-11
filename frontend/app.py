from flask import Flask, render_template, request, redirect, flash

ALLOWED_EXTENSIONS = {'pkl'}

app = Flask(__name__)
app.secret_key = 'qw89mvcty342cvn4rcv89m4q3ynccq89xr7n4'

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

        if  model_file.filename == '':
            flash('No model uploaded, try again')
            return redirect('/')

        if predictions_file.filename == '':
            flash('No predictions uploaded, try again')
            return redirect('/')
        #TODO: scrivi il codice per manipolare i file
        
        return redirect('/')
    
    return render_template('home.html')

@app.route('/freqvsfreq', methods=['GET', 'POST'])
def freqvsfreq():
    return render_template('freqvsfreq.html')
    if request.method == 'POST':
        return redirect('/')

@app.route('/freqvsrel')
def freqvsrel():
    return "<p>Hello, World!</p>"


