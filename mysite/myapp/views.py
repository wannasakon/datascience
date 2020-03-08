from django.shortcuts import render
from joblib import load
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
categories = ['talk.religion.misc', 'sci.electronics','rec.motorcycles', 'comp.graphics']
# categories = ['talk.religion.misc', 'soc.religion.christian','sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
# fetch_20newsgroups(subset='train', categories=categories)
# Create your views here.
def index(req):
    model = load('./myapp/static/chatgroup.model')
    result = ""
    group = ""
    # submit = 'แสดงกราฟ'
    if req.method == 'POST':
        print('เขา POST มา')
        group = str(req.POST['group'])
        print(group)
        pred = model.predict([group])
        result = train.target_names[pred[0]]
    
    # else:
    #     print('เขากด enter มา')
    
       
    return render(req, 'myapp/index.html',{ 
        'result': result,
        # 'submit': submit, 
    })

