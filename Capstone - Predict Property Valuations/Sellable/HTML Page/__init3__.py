import flask
import dill 

import numpy as np
import pandas as pd 

user_input = {
          'Suburb' : 'Harris Park'
         , 'Beds'  : 1
         , 'Baths' : 2 
         , 'Car'   : 1
         , 'Lot'   : 560 
}

def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(18)
    
    # set the numerical input as they are
    enc_input[0] = data['Beds']
    enc_input[1] = data['Baths']
    enc_input[2] = data['Car']
    enc_input[3] = data['Lot']
    
 app = flask.Flask(__name__)

with open('RandomForest_Sellable.pkl', 'rb') as f:
    PREDICTOR = dill.load(f)
    
##################################
@app.route("/")
def hello():
    return '''
    <body>
    <h2> This is my Capstone Project for <h2><b>S E L L A B L E </b>
    </body>
    '''

##################################
@app.route('/greet/<name>')
def greet(name):
    '''Say hello to your first parameter'''
    return "Hello, %s!" %name

@app.route('/predict', methods=["GET"])
def predict():
    print ('PPPPRRREEEDDDIIICCCTTT')
    user_input['Suburb'] = flask.request.args['Suburb']
    user_input['Beds']   = flask.request.args['Beds']
    user_input['Baths']  = flask.request.args['Baths']
    user_input['Cars']   = flask.request.args['Cars']
    user_input['Lot']    = flask.request.args['Lot']
    
    item = pd.DataFrame([user_input])
    
    print (item)

    score = PREDICTOR.predict_proba(item)
    results = {'survival chances': score[0,1], 'death chances': score[0,0]}
    return flask.jsonify(results)

##################################
#@app.route('/page')
#def show_page():
#    return flask.render_template('dataentrypage.html')

##################################
@app.route('/page', methods=['POST', 'GET'])
def page():
    print ('PPPPPPAAAAAGGGGEEEE')
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form
    
       print ('AAAAAAAAAAA')

       user_input['Suburb'] = inputs['Suburb']
        
       user_input['Beds']   = inputs['Beds']
       user_input['Baths']  = inputs['Baths']
       user_input['Cars']   = inputs['Cars']
       user_input['Lot']    = inputs['Lot']
       
#       item = pd.DataFrame([[Suburb, Beds, Baths, Cars, Lot]], columns=['Suburb', 'Beds', 'Baths', 'Cars', 'Lot'])
       item = pd.DataFrame([user_input])
    
       print (item)
     
#       print (item)
        
       score = PREDICTOR.predict_proba(item)
       results = {'survival chances': score[0,1], 'death chances': score[0,0]}
       survive = int(score[0,1] * 100)
       dead = int(score[0,0] * 100)
    else:
        survive = 0
        dead = 0
    return flask.render_template('dataentrypage.html', survive=survive, dead=dead)

##################################
if __name__ == '__main__':
    print ('This is THE START')
    app.run(debug=True)