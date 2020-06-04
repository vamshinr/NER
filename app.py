import flask
from flask import Flask,request,jsonify
from flair.models import SequenceTagger
from flair.data import Sentence
app=Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('home.html')

@app.route('/result',methods=['POST'])
def ner():
    text = request.form
    key,value=list(text.items())[0]
    tagger = SequenceTagger.load('ner')
    sentence = Sentence(str(value))
    tagger.predict(sentence)
    sent=sentence.to_tagged_string()
    result=sentence.to_dict(tag_type="ner")
    length=len(result['entities'])
    res=[]
    for i in range(0,length):
        txt=result['entities'][i]['text']
        label=str(result['entities'][i]['labels'][0])[0:3]
        res.append([txt,label])
    return flask.render_template('result.html',length=length,sent=sent,res=res,value=value)

if __name__=="__main__":
    app.run()