

from flask_wtf import FlaskForm
from wtforms import StringField, FileField, SubmitField

class ProductForm(FlaskForm):
    image_file = FileField('upload image')
    submit = SubmitField('Submit')