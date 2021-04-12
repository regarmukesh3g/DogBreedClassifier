
from PIL import Image

from flask import render_template, request, send_from_directory
from webapp.forms import ProductForm
from predict import predict_breed

from flask import Flask

SECRET_KEY = 'cdsandj13e3'
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = '/upload_images'
app.config['TRAIN_PATH'] = '/train_images'


@app.route('/upload_images/<path:filepath>')
def send_file(filepath):
    return send_from_directory('upload_images', filepath)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Homepage of the webapp.
    """
    form = ProductForm()
    return render_template('master.html', form=form)


# web page that handles user query and displays model results
@app.route('/go', methods=['GET', 'POST'])
def predict():
    """
    Page to show results.
    Returns:
        webpage to results.
    """
    # save user input in query

    form = ProductForm()
    img_file = request.files['image_file']
    print(img_file)
    image1 = Image.open(img_file)
    img_path = 'upload_images/temp_file.jpeg'
    image1.save(img_path)
    pred_breed = predict_breed(img_path).split('.')[-1]
    return render_template(
        'go.html',
        title=pred_breed,
        image_path=img_path,
        form=form
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True, threaded=False)


if __name__ == '__main__':
    main()
