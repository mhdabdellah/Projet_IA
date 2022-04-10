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
#for tensor based operations
from tensorflow.keras.utils import normalize

# load a model => charger le model model.h5
model = load_model("model.h5")


app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/save', methods=["POST"])
def save():
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    f = request.files['image']
    f_name = secure_filename(f.filename)
    f.save(os.path.join("images",f_name))
    # file = request.form['image']
    # pixels =list(file.getdata())
    # tableauPixels =np.array(file)
    # nouvellePhoto =Image.fromarray(tableauPixels)
    # nouvellePhoto.save("sample_test_images/image.png", "PNG")
    return "Saved Image"


# @app.route('/prediction', methods=["POST"])
def prediction():
    #in this part we will be trying to detect if a person in a image is wearing a face mask or not ,let's start by reading in a sample image that image is out of the training samlple images
    #Image file path for sample image images  image
    # test_image_file_path = request.form['image']
    #lire l'image
    test_image_file_path = "images/image.jpeg"
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
    #making a prediction
    prediction = model.predict(x)
    #interpreting these predictions 
    #we can use the np.argmax() methodto find the index with the highest probability values 
    # returns the index of maximum value
    pred = np.argmax(prediction)
    return pred

@app.route('/saveandpredict', methods=["POST"])
def saveAndPredict():
    # stocker l'image dans un variable file
    file = request.files['image']
    #Chargement de l’image par PIL et affichage des propriétés de l’image
    # ajoutre try except pour gerer les exceptions
    try: 
        # charger l'image et le stocker dans la variable photo
        photo = Image.open(file)
    except IOError :
         print (" Erreur lors de l’ouverture du fichier !")

    #np.array( ) permet d’obtenir une matrice du type array de numpy :
    tableauPixels =np.array(photo)
    # on fait une copie du tableau initial pour garder la matrice originale :
    M=np.copy(tableauPixels)
    #exportation en png : très rapide et ne plante pas la console
    # pour exporter le tableau de Pixels en image
    nouvellePhoto =Image.fromarray(M)
    nouvellePhoto.save("images/image.jpeg", "JPEG")
    pred = prediction()
    return render_template('predictionResultat.html', data=pred)


@app.route('/predict', methods=['POST'])
def home():
    return render_template('prediction.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















