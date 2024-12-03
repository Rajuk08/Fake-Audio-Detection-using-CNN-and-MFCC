from flask import Flask, request, render_template_string, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
from detect_fake1 import detect_fake_audio
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'wav'}

HISTORY_FILE = 'static/uploads/history.txt'

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        f.write('Filename,Result,Real Confidence (%),Fake Confidence (%)\n')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_confidence_image(real_prob, fake_prob):
    labels = ['Real', 'Fake']
    scores = [real_prob, fake_prob]

    plt.bar(labels, scores, color=['green', 'red'])
    plt.ylabel('Confidence (%)')
    plt.title('Confidence Scores')
    plt.savefig('static/confidence_scores.png')
    plt.close()

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake Audio Detection</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f7f9fc; margin: 0; padding: 20px; }
        .container { max-width: 780px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
        h1, h2 { text-align: center; }
        button, input[type="file"] { padding: 10px; border: none; border-radius: 5px; margin: 5px; cursor: pointer; }
        button { background-color: #007bff; color: white; }
        button:hover { background-color: #0056b3; }
        table { width: 100%; margin: 20px 0; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        audio { margin: 20px auto; display: block; }
        .pagination { text-align: center; }
        .pagination button { margin: 5px; background-color: #007bff; color: white; border: none; }
        .pagination button:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fake Audio Detection</h1>
        <form id="upload-form" action="/upload" method="post" enctype="multipart/form-data">
            <label for="file">Choose an audio file (.wav):</label>
            <input type="file" id="file" name="file" accept=".wav" required>
            <button type="submit">Upload & Analyze</button>
        </form>
        <form id="record-form" action="/record" method="post">
            <button type="submit">Record & Analyze</button>
        </form>
        <h2>Upload History</h2>
        <div id="history-container"></div>
    </div>
    <script>
        async function fetchHistory(page = 1) {
            const response = await fetch(`/history?page=${page}`);
            const data = await response.json();
            const historyContainer = document.getElementById('history-container');
            historyContainer.innerHTML = `
                <table>
                    <tr>
                        <th>Filename</th>
                        <th>Result</th>
                        <th>Real Confidence (%)</th>
                        <th>Fake Confidence (%)</th>
                        <th>Delete</th>
                    </tr>
                    ${data.items.map(item => `
                        <tr>
                            <td>${item.filename}</td>
                            <td>${item.result}</td>
                            <td>${item.real_confidence}</td>
                            <td>${item.fake_confidence}</td>
                            <td>
                                <form method="POST" action="/delete_file">
                                    <input type="hidden" name="filename" value="${item.filename}">
                                    <button type="submit">Delete</button>
                                </form>
                            </td>
                        </tr>`).join('')}
                </table>
                <div class="pagination">
                    ${Array.from({ length: data.total_pages }, (_, i) => `
                        <button onclick="fetchHistory(${i + 1})">${i + 1}</button>`).join('')}
                </div>
            `;
        }
        fetchHistory();
    </script>
</body>
</html>
'''

RESULT_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Result</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f7f9fc; margin: 0; padding: 20px; }
        .container { max-width: 650px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
        h1, h2 { text-align: center; }
        img { display: block; margin: 20px auto; }
        audio { margin: 20px auto; display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Analysis Result</h1>
        <h2>The audio is classified as: {{ result }}</h2>
        <img src="{{ url_for('static', filename='confidence_scores.png') }}" alt="Confidence Scores">
        <audio controls>
            <source src="{{ url_for('static', filename='uploads/' + filename) }}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        <p>
            <a href="{{ url_for('static', filename='uploads/' + filename) }}" download>Download Uploaded Audio</a>
        </p>
        <a href="/">Back to Home</a>
    </div>
</body>
</html>
'''

def analyze_audio(file_path, filename):
    try:
        result, real_prob, fake_prob = detect_fake_audio(file_path, visualize=True)
        
        save_confidence_image(real_prob, fake_prob)
        with open(HISTORY_FILE, 'a') as f:
            f.write(f"{filename},{result},{real_prob:.2f},{fake_prob:.2f}\n")
        
        return render_template_string(RESULT_TEMPLATE, result=result, filename=filename)
    except Exception as e:
        flash(f"Error analyzing audio: {e}")
        return redirect(url_for('index'))

# Home route
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/history')
def history():
    page = int(request.args.get('page', 1))
    items_per_page = 5
    history = []
    with open(HISTORY_FILE, 'r') as f:
        next(f) 
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                history.append({
                    'filename': parts[0],
                    'result': parts[1],
                    'real_confidence': parts[2],
                    'fake_confidence': parts[3]
                })
    total_items = len(history)
    start = (page - 1) * items_per_page
    end = start + items_per_page
    paginated = history[start:end]
    return jsonify({
        'items': paginated,
        'total_pages': (total_items + items_per_page - 1) // items_per_page
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return analyze_audio(file_path, filename)
    else:
        flash('Invalid file type. Please upload a .wav file.')
        return redirect(url_for('index'))

@app.route('/record', methods=['POST'])
def record_audio():
    fs = 44100
    seconds = 5 
    filename = 'recorded_audio.wav'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait() 
    sf.write(file_path, recording, fs)

    return analyze_audio(file_path, filename)

@app.route('/delete_file', methods=['POST'])
def delete_file():
    filename = request.form['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f'File {filename} deleted successfully.')
    else:
        flash(f'File {filename} not found.')

    with open(HISTORY_FILE, 'r') as f:
        lines = f.readlines()

    with open(HISTORY_FILE, 'w') as f:
        for line in lines:
            if line.split(',')[0] != filename: 
                f.write(line)

    return redirect(url_for('index'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
