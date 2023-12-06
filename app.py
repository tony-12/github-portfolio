#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,redirect,render_template,request
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from Feature_Extraction import featureExtraction

with open('DT_Model1.pickle','rb') as f:
    model = pickle.load(f)
model1 = pickle.load(open('DT_Model1.pickle','rb'))
#with open('Scaler.pickle','rb') as f:
    #Scaler = pickle.load(f)


app =  Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/Predict', methods = ['GET','POST'])
def Predict():
    if request.method == 'POST':
        geturl = request.form['url']
        check = featureExtraction(geturl)
        x = np.array([check])
        #scal = [np.array(check)]
        #scaler = Scaler.fit(check)
        prediction = model1.predict(x)
        
        output = prediction
        if(output==1):
            pred = 'Your are safe !!'
        else:
            pred = 'Phishing site, Be Cautious!'
            
        
        return render_template('index.html',url_path=geturl,url=pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0')





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




