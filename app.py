from flask import Flask, request, render_template, jsonify
from flask_bootstrap import Bootstrap
from forms import TrumpForm

app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = 'devkey'

app.config.from_object(__name__)

@app.route('/gen/', methods=["POST"])
def generate_text():
    """
    Accepts a POST request with parameters and generates a speech from the stored model.
    """
    data = request.json
    message = ''

    return message

@app.route('/', methods=['GET'])
def main_page():
    data = request.json
    form = TrumpForm()
    return render_template("index.html", form=form)

def main(host="0.0.0.0", port=9000, debug=True):
    app.run(host=host, port=port,debug=debug)

if __name__ == '__main__':
    main()
