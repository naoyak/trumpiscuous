from flask import Flask, request, render_template, jsonify
from flask_bootstrap import Bootstrap
from keras.models import load_model
from forms import TrumpForm
from train import prep_data, generate_speech


filenames = ['trump_speeches.txt', 'mgmt_of_savagery.txt', 'cartman.txt']

app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = 'devkey'

app.config.from_object(__name__)

prep_data = prep_data(filenames, 0.5, 15, 10)
model = load_model('model_save/all.h5')
print('Model loaded!')

@app.route('/gen/', methods=["POST"])
def generate_text():
    """
    Accepts a POST request with parameters and generates a speech from the stored model.
    """
    data = request.json
    diversity = float(data['diversity'])
    length = float(data['length'])
    text = generate_speech(model, diversity, prep_data)
    message = 'hello'

    return text

@app.route('/', methods=['GET'])
def main_page():
    data = request.json
    form = TrumpForm()
    return render_template("index.html", form=form)

def main(host="0.0.0.0", port=9000, debug=True):
    print('Starting app...')
    app.run(host=host, port=port,debug=debug)

if __name__ == '__main__':
    main()
