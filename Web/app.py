from flask import Flask, render_template, request
from statistics import mode
from warnings import filters
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from werkzeug.utils import secure_filename
import os
import cv2
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from PIL import Image
#for tensor based operation
import keras
# from keras.utils import normalize
from tensorflow.keras.utils import normalize
# from tensorflow.keras.utils import  normalize
import tensorflow
# load a model => charger le model model.h5
model1 = load_model("model.h5")
model2 = load_model("Functionalmodel.h5")

from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "cairocoders-ednalan"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')


# @app.route('/prediction', methods=["POST"])
def makePrediction(filenames):
    images= []
    for filename in filenames:
        test_image_file_path = "static/uploads/"+ filename
        #loading the image 
        img = plt.imread(test_image_file_path)
        #initializing the detector 
        detector = MTCNN()
        #Detecting the face in the image 
        faces = detector.detect_faces(img)
        #next performing face detector and image pre-processing together
        # reading in the image as a grayscale image 
        img_array = cv2.imread(test_image_file_path, cv2.IMREAD_GRAYSCALE)
        #initializing the detector .3
        detector = MTCNN()
        #detecting the faces in in the images
        faces = detector.detect_faces(img)
        #getting the values for bounding box 
        x1,x2,width,height= faces[0]["box"]
        #selection the portion covred by the bounding box
        crop_image = img_array[x2 : x2 + height, x1 : x1 + width]
        #resizing the image 
        img_size = 50
        new_img_array = cv2.resize(crop_image,(img_size,img_size))
        # some more pre-processing
        #reshaping image
        x = new_img_array.reshape(-1,50,50,1)
        #normalizing to the range between 0 and 1
        x = normalize(x, axis=1)
        image=images.append(x)
        print('images1')
        print(image)
    image = np.array(images)
    image = np.vstack((images))
    print('images2')
    print(image)
    classes = model1.predict(image) 
    print('class')
    print(classes) 
    preds=[]
    prc1=[]
    prc2 =[]    
    for prediction in classes :
            pred0 = np.argmax(prediction)
            prc01,prc02=prediction
            preds.append(pred0)
            prc1.append(prc01)
            prc2.append(prc02)
            print('images3')
            print(prc1)
            print(prc2)
            print(preds)
           

    print('images4')
    print(prc1)
    print(prc2)
    print(preds)
    return preds,prc1,prc2


    # classes = model1.predict(images)
    # prc1=prediction[0][0]* 100
    # prc2=prediction[0][1]* 100
    # print("la valeur du preduction Sequentiel")
    # print(prc1,prc2)
    # print(prediction)
    #     #interpreting these predictions 
    #     #we can use the np.argmax() methodto find the index with the highest probability values 
    #     # returns the index of maximum value
    # pred = np.argmax(prediction)
    # print(pred)
    # return pred,prc1,prc2    #making a prediction

        

def PredictionFunctionalmodel(filenames):
    images= []
    for filename in filenames:
        test_image_file_path = "static/uploads/"+ filename
        #loading the image 
        img = plt.imread(test_image_file_path)
        #initializing the detector 
        detector = MTCNN()
        #Detecting the face in the image 
        faces = detector.detect_faces(img)
        #next performing face detector and image pre-processing together
        # reading in the image as a grayscale image 
        img_array = cv2.imread(test_image_file_path, cv2.IMREAD_GRAYSCALE)
        #initializing the detector .3
        detector = MTCNN()
        #detecting the faces in in the images
        faces = detector.detect_faces(img)
        #getting the values for bounding box 
        x1,x2,width,height= faces[0]["box"]
        #selection the portion covred by the bounding box
        crop_image = img_array[x2 : x2 + height, x1 : x1 + width]
        #resizing the image 
        img_size = 50
        new_img_array = cv2.resize(crop_image,(img_size,img_size))
        # some more pre-processing
        #reshaping image
        x = new_img_array.reshape(-1,50,50,1)
        #normalizing to the range between 0 and 1
        x = normalize(x, axis=1)
        image=images.append(x)
        print('images1')
        print(image)
    image = np.array(images)
    image = np.vstack((images))
    print('images2')
    print(image)
    classes = model2.predict(image) 
    print('class')
    print(classes) 
    preds=[]
    prc1=[]
    prc2 =[]    
    for prediction in classes :
            pred0 = np.argmax(prediction)
            prc01,prc02=prediction
            preds.append(pred0)
            prc1.append(prc01)
            prc2.append(prc02)
            print('images3')
            print(prc1)
            print(prc2)
            print(preds)
           

    print('images4')
    print(prc1)
    print(prc2)
    print(preds)
    return preds,prc1,prc2


 
@app.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    # if files == '':
    #     flash('No image selected for uploading')
    #     return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names=[] 
    for file in files:   
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
            
            img = Image.open("static/uploads/"+ filename)
            
            if img.width > 190 or img.height > 150:
                output_size = (190, 150)
                img.thumbnail(output_size)
                img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
          
        else:
            flash('Mettre un image de types  - png, jpg, jpeg, gif')
            return redirect(request.url)

    flash('Image est charger par succes')
    preds,prc1,prc2 = makePrediction(file_names)
    Zipp=zip(file_names,preds,prc1,prc2)
    return render_template('index.html', RESULTAT=Zipp )
    # return render_template('index.html', filenames=file_names, datas=preds,pctgs1=prc1,pctgs2=prc2)
 
@app.route('/display/<filename>')
def display_image(filename):
    resultat = PredictionFunctionalmodel(filename)
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)



@app.route('/fonctional', methods=['POST'])
def functional():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    # if files == '':
    #     flash('No image selected for uploading')
    #     return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names=[] 
    for file in files:   
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
            
            img = Image.open("static/uploads/"+ filename)
            
            if img.width > 190 or img.height > 150:
                output_size = (190, 150)
                img.thumbnail(output_size)
                img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
          
        else:
            flash('Mettre un image de types  - png, jpg, jpeg, gif')
            return redirect(request.url)

    flash('Image est charger par succes')
    preds,prc1,prc2 = PredictionFunctionalmodel(file_names)
    Zipp=zip(file_names,preds,prc1,prc2)
    return render_template('fonctional.html', RESULTAT=Zipp )

# #Machine learning
# @app.route('/save', methods=["POST"])
# def save():
 
#     f = request.files['image']
#     f_name = secure_filename(f.filename)
#     f.save(os.path.join("images",f_name))

    
#     return "Saved Image"



# @app.route('/saveandpredict', methods=["POST"])
# def saveAndPredict():
#     # stocker l'image dans un variable file
#     file = request.files['image']
#     #Chargement de l’image par PIL et affichage des propriétés de l’image
#     # ajoutre try except pour gerer les exceptions
#     try: 
#         # charger l'image et le stocker dans la variable photo
#         photo = Image.open(file)
#     except IOError :
#          print (" Erreur lors de l’ouverture du fichier !")

#     #np.array( ) permet d’obtenir une matrice du type array de numpy :
#     tableauPixels =np.array(photo)
#     # on fait une copie du tableau initial pour garder la matrice originale :
#     M=np.copy(tableauPixels)
#     print("table pix images")
#     print(M)
#     #exportation en png : très rapide et ne plante pas la console
#     # pour exporter le tableau de Pixels en image
#     nouvellePhoto =Image.fromarray(M)
#     nouvellePhoto.save("images/image.jpeg", "JPEG")
#     pred = prediction()
    
#     return render_template('predictionResultat.html', data=pred )
 


if __name__ == "__main__":
   app.run(debug=True)