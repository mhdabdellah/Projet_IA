import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5 import uic,QtWidgets
from PyQt5.QtCore import QDir
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import  QApplication, QFileDialog
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
#for tensor based operations
from tensorflow.keras.utils import normalize
# visualisation de la model pour bien comprendre ce qui va se passé
from tensorflow.keras.utils import plot_model




qtCreatorFile = "Accueil.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
class MyAppp(QtWidgets.QMainWindow,Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.Bt1.clicked.connect(self.open)
        self.Bt.clicked.connect(self.SequentialModel)
        self.Bt2.clicked.connect(self.Fermer)
        self.Bt3.clicked.connect(self.FonctionalModel)
        self.path_image = 'images'

    def Fermer(self):
        self.close()

    def open(self):
        path= QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(), 'Images (*.png *.jpg)')
        img = Image.open(path[0])
        # tableauPixels = np.array(img)
        # nouvellePhoto = Image.fromarray(tableauPixels)
        # nouvellePhoto.save("mohamed/image.png", "PNG")
        if path != ('', ''):
            print(path[0])
            tableauPixels = np.array(img)
            print(tableauPixels)
            nouvellePhoto = Image.fromarray(tableauPixels)
            print(nouvellePhoto)
            nouvellePhoto.save("images/image.jpeg", "JPEG")
            self.logo.setPixmap(QPixmap("images/image.jpeg"))
            img.show()



    def SequentialModel(self):

        # load a model => charger le model model.h5
        model = load_model("model.h5")

        # in this part we will be trying to detect if a person in a image is wearing a face mask or not ,let's start by reading in a sample image that image is out of the training samlple images
        # Image file path for sample image images  image
        # test_image_file_path = request.form['image']
        # lire l'image
        test_image_file_path = "images/image.jpeg"
        # loading the image
        img = plt.imread(test_image_file_path)
        # initializing the detector
        detector = MTCNN()
        # Detecting the face in the image
        faces = detector.detect_faces(img)
        # next performing face detector and image pre-processing together
        # reading in the image as a grayscale image
        img_array = cv2.imread(test_image_file_path, cv2.IMREAD_GRAYSCALE)
        # initializing the detector .3
        detector = MTCNN()
        # detecting the faces in in the images
        faces = detector.detect_faces(img)
        # getting the values for bounding box
        x1, x2, width, height = faces[0]["box"]
        # selection the portion covred by the bounding box
        crop_image = img_array[x2: x2 + height, x1: x1 + width]
        # resizing the image
        img_size = 50
        new_img_array = cv2.resize(crop_image, (img_size, img_size))
        # some more pre-processing
        # reshaping image
        x = new_img_array.reshape(-1, 50, 50, 1)
        # normalizing to the range between 0 and 1
        x = normalize(x, axis=1)
        # making a prediction
        prediction = model.predict(x)
        # interpreting these predictions
        # we can use the np.argmax() methodto find the index with the highest probability values
        # returns the index of maximum value
        pred = np.argmax(prediction)
        result = None
        if pred == 1:
            result = "c'est bien il porte une masque"
        else:
            result = "c'est grave il ne porte pas une masque"
        self.L1.setText(result)
        plot_model(model, to_file='model.png')
        self.sequential.setPixmap(QPixmap("model.png"))
        pourcentage_prediction = None
        if pred == 1:
            pourcentage_prediction = 'la pourcentage de prediction ' + '{:.2f} %'.foramt(prediction[0][1] * 100)
        else:
            pourcentage_prediction = 'la pourcentage de prediction ' + str( int(prediction[0][0] * 100) ) + '%' 
        self.label_sequential.setText(pourcentage_prediction)
        print('Sequential Model')

    def FonctionalModel(self):

        # load a model => charger le model Functionalmodel.h5
        model = load_model("Functionalmodel.h5")

        # in this part we will be trying to detect if a person in a image is wearing a face mask or not ,let's start by reading in a sample image that image is out of the training samlple images
        # Image file path for sample image images  image
        # test_image_file_path = request.form['image']
        # lire l'image
        test_image_file_path = "images/image.jpeg"
        # loading the image
        img = plt.imread(test_image_file_path)
        # initializing the detector
        detector = MTCNN()
        # Detecting the face in the image
        faces = detector.detect_faces(img)
        # next performing face detector and image pre-processing together
        # reading in the image as a grayscale image
        img_array = cv2.imread(test_image_file_path, cv2.IMREAD_GRAYSCALE)
        # initializing the detector .3
        detector = MTCNN()
        # detecting the faces in in the images
        faces = detector.detect_faces(img)
        # getting the values for bounding box
        x1, x2, width, height = faces[0]["box"]
        # selection the portion covred by the bounding box
        crop_image = img_array[x2: x2 + height, x1: x1 + width]
        # resizing the image
        img_size = 50
        new_img_array = cv2.resize(crop_image, (img_size, img_size))
        # some more pre-processing
        # reshaping image
        x = new_img_array.reshape(-1, 50, 50, 1)
        # normalizing to the range between 0 and 1
        x = normalize(x, axis=1)
        # making a prediction
        prediction = model.predict(x)
        evaluation = model.evaluate(x)
        # score = model.score(x)
        print(evaluation)
        # predictionp = model.predict_proba(x)
        print(prediction)
        print(str( prediction[0][0] * 100 ) + '%')
        # print(predictionp)
        # print(score)
        # interpreting these predictions
        # we can use the np.argmax() methodto find the index with the highest probability values
        # returns the index of maximum value
        pred = np.argmax(prediction)
        result = None
        if pred == 1:
            result = "c'est bien il porte une masque"
        else:
            result = "c'est grave il ne porte pas une masque"
        self.L1.setText(result)
        plot_model(model, to_file='model.png')
        self.functional.setPixmap(QPixmap("model.png"))
        pourcentage_prediction = None
        if pred == 1:
            pourcentage_prediction = 'la pourcentage de prediction ' + str( int(prediction[0][1] * 100) ) + '%'
        else:
            pourcentage_prediction = 'la pourcentage de prediction ' + str( int(prediction[0][0] * 100) ) + '%' 
        self.label_functional.setText(pourcentage_prediction)
        print('FonctionalModel')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MyAppp()
    gui.show()
    sys.exit(app.exec_())
