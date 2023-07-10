from http.client import OK
from flask import Flask
from python import get_token
import requests
app = Flask(__name__)   # Flask constructor
  
# A decorator used to tell the application
# which URL is associated function
@app.route('/')      
def home():
    print('hi')
    get_token().token_results()
    
    print("hello")
    return "OK"
  
if __name__=='__main__':
   app.run()