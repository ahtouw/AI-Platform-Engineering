from flask import Flask, render_template, request
from model_init import Model
import os
from scipy.io import wavfile

# Model declared and loaded
cnn_model = Model('gru', 'pb')

app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'temp')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def inference():
    if not request.files["audio"]:
        return render_template("badinp.html")

    # Load audio file for preprocessing
    file = request.files["audio"]
    write_path = os.path.join(uploads_dir, file.filename)
    file.save(write_path)
    rate, wav  = wavfile.read(write_path)
    os.remove(write_path)

    # Processing and passing data through model, returning inference data and highest result
    infer, highest_conf = cnn_model.process_audio(wav, rate)

    return render_template("infer.html", result=infer, output=highest_conf)


if __name__ == "__main__":
    app.run(debug=True)