from flask import Flask, render_template, request
import model.model as m
import numpy as np

app = Flask(__name__)
label = ['condition']
features = ['MEAN_RR','RMSSD','pNN25','pNN50','LF','HF','LF_HF']

@app.route("/", methods = ["GET","POST"])
def hello():
     st = ""  # Prediction string, reset on each request
     form_data = {} # Dictionary to hold the form values

     if request.method == "POST":
        # We will pass the form data back to the template
         form_data = request.form
        
         l=[]
         for i in features:
             try:
                # This is the fix: use 'or 0' to handle empty strings
                 l.append(float(request.form[i] or 0))
             except ValueError:
                # Default to 0 if input is invalid
                 l.append(0.0)
                
         p = m.predict_pipe(l)
         p = np.argmax(p[0])

         if p == 0:
             st = "No Stress"
         elif p == 1:
             st = "Low Stress"
         else:
             st = "High Stress"
    
    # Pass both the prediction (cond) and the form data (form_data)
     return render_template('index.html', cond=st, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)