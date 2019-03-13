import flask
import dill 
import numpy as np
import pandas as pd 
import locale

app = flask.Flask(__name__)

user_input = {
           'Beds'  : 0
         , 'Baths' : 0 
         , 'Cars'  : 0
         , 'Lot'   : 0 
         , 'suburb_Blacktown'        : 0
         , 'suburb_Harris Park'      : 0
         , 'suburb_Kings Langley'    : 0
         , 'suburb_Lalor Park'       : 0
         , 'suburb_Marayong'         : 0
         , 'suburb_North Parramatta' : 0
         , 'suburb_North Rocks'      : 0
         , 'suburb_Northmead'        : 0
         , 'suburb_Old Toongabbie'   : 0
         , 'suburb_Parramatta'       : 0
         , 'suburb_Prospect'         : 0
         , 'suburb_Seven Hills'      : 0
         , 'suburb_Toongabbie'       : 0    
}


with open('RandomForest_Sellable.pkl', 'rb') as f:
    PREDICTOR = dill.load(f)
    
@app.route("/")
def hello():
    return '''
    <body>
    <h2> This is my Capstone Project for <h2><b>S E L L A B L E</b>
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
    
    deploy_model = pd.DataFrame([user_input])
    a = input_to_one_hot(deploy_model.loc[0])    
    
    print (a)
    
    item = pd.DataFrame([user_input])
    print ('*'*30, item)
    
    print (item)

    score = PREDICTOR.predict(item)
    results = {'survival chances': score[0,1], 'death chances': score[0,0]}
    return flask.jsonify(results)

##################################
#@app.route('/page')
#def show_page():
#    return flask.render_template('dataentrypage.html')

##################################
@app.route('/page', methods=['POST', 'GET'])
def page():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':        
       enc_input = np.zeros(17)
       user_input = init_user_input
       inputs = flask.request.form
        
       hold_field = ('suburb_' + inputs['Suburb'])
        
       user_input[hold_field] = 1
        
       user_input['Beds']  = inputs['Beds'][0]
       user_input['Baths'] = inputs['Baths'][0]
       user_input['Cars']  = inputs['Cars'][0]
       user_input['Lot']   = inputs['Lot']

       dsuburb   = inputs['Suburb']
       dbeds   = inputs['Beds'][0]
       dbaths = inputs['Baths'][0]
       dcars   = inputs['Cars'][0]
       dlot     = inputs['Lot']
       count  = 0 
       for key,value in user_input.items():
           
           try:
               enc_input [count] = value
           except:
               Not_Found = ('Suburb' + key + 'Not Found')
               print (Not_Found)
           print (count, '\t', key, '\t', value)
           count += 1
       print ('X'*30, count, "\t", enc_input) 
       item = enc_input.reshape(1,-1)

       score = PREDICTOR.predict(item)
#       locale.currency(score, grouping=True )
       valuation = int(score)
#       valuation = locale.currency(int(score), grouping = True)
       
    else:
       print ('It has no match' , 'm'*30) 
       valuation = 0
        
       dsuburb = ' ' 
       dbeds   = 0
       dbaths  = 0
       dcars   = 0
       dlot    = 0
       
#    return flask.render_template('dataentrypage.html', valuation=valuation, dbeds=dbeds, dbaths=dbaths, dcars=dcars, dlot=dlot)
    return flask.render_template('dataentrypage.html', valuation=valuation, dsuburb=dsuburb, dbeds=dbeds, dbaths=dbaths, dcars=dcars, dlot=dlot)

##################################
if __name__ == '__main__':
    init_user_input = user_input
    print ('S T A R T   PROCESS')
    app.run(debug=True)