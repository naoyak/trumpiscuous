from flask_wtf import Form
from wtforms import DecimalField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired

class TrumpForm(Form):
    length = IntegerField('Speech Length', validators=[DataRequired()])
    diversity = DecimalField('Diversity', validators=[DataRequired()])
    corpus = SelectField('Choose Corpus', choices=[('trump', 'Trump Speeches'), ('all', 'Trump-Jihadi-Cartman')])
    submit = SubmitField('Generate text! 🔥')
