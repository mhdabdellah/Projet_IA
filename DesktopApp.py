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
from keras.preprocessing import image
import os
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
        self.Bt1.clicked.connect(self.choisirImage)
        self.Bt.clicked.connect(self.SequentialModel)
        self.Bt2.clicked.connect(self.predictionMultiImageparFunctionalModel)
        # self.Bt2.clicked.connect(self.Fermer)
        self.Bt3.clicked.connect(self.FonctionalModel)
        self.Bt4.clicked.connect(self.predictionMultiImageparsequentialModel)
        self.path_image = 'images'

    def Fermer(self):
        self.close()

    # def open_directory_callback(self):

    #     # path 
    #     self._base_dir = os.getcwd()
    #     self._images_dir = os.path.join(self._base_dir, 'test_images')

    #     # open a file dialog and select the folder path 
    #     dialog = QFileDialog()
    #     self._folder_path = dialog.getExistingDirectory(None, 'select Folder')

    #     # get the list of images in the folder and read using matplotlib and print its shape
    #     self.list_of_images = os.listdir(self._folder_path)
    #     self.list_of_images = sorted(self.list_of_images)

    #     #length of images 
    #     print('Number of Imagesin the selected folder :{}'.format(len(self.list_of_images)))
    #     input_img_raw_string = '{}\\{}'.format(self._images_dir,self.list_of_images[0])

    #     # show the first image in the same windows . (self.label comes from the Ui_main_window class)
    #     self.logo.setPixmap(QPixmap(input_img_raw_string))
    #     self.logo.show()

    #     self.i = 0


    # def next_buttton_callback():
    #     # total images in list 
    #     total_images = len(self.list_of_images)
    #     if self.list_of_images:
    #         try:
    #             self.i = (self.i + 1) % total_images
    #             img = self.list_of_images[self.i]
    #             self.label.selPixmap(QPixmap('{}\\{}'.format(self._images_dir, img)))
    #             self.label.show()

    #         except ValueError as e:
    #             print('The selected folder does not contain any images')

    def predictionMultiImageparsequentialModel(self):
        # load a model => charger le model model.h5
        model = load_model("model.h5")

        # path 
        self._base_dir = os.getcwd()
        self._images_dir = os.path.join(self._base_dir, 'test_images')

        # open a file dialog and select the folder path 
        dialog = QFileDialog()
        self._folder_path = dialog.getExistingDirectory(None, 'select Folder')

        # get the list of images in the folder and read using matplotlib and print its shape
        self.list_of_images = os.listdir(self._folder_path)
        self.list_of_images = sorted(self.list_of_images)

        #length of images 
        print('Number of Imagesin the selected folder :{}'.format(len(self.list_of_images)))

        # image folder 
        # folder_path = 'sample_test_images'
        # dimentions of images
        img_width, img_height = 50, 50

        # load all images into a list
        images= []
        # prede = []
        # for img in os.listdir(folder_path):
        for img in self.list_of_images:
            # print(os.listdir(folder_path))
            print(img)
            path = os.path.join(self._folder_path, img)
             # loading the image
            img = plt.imread(path)
            # initializing the detector
            detector = MTCNN()
            # Detecting the face in the image
            faces = detector.detect_faces(img)
            # next performing face detector and image pre-processing together
            # reading in the image as a grayscale image
            img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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
           
            # img = image.load_img(img, target_size=(img_width, img_height))
            # img = image.img_to_array(img)
            # img = img.reshape(-1, 50, 50, 1)
            # img = normalize(img, axis=1)
            # img = np.expand_dims(img, axis=0)
            # pred = model.predict(img)
            # print(pred)
            # prede.append(pred)
            images.append(x)

        # stack up images list to pass for prediction 
        print(len(images))
        images = np.array(images)
        images = np.vstack((images))
        print(images.shape)
        print('yes')
        print(len(images))
        classes = model.predict(images)
        print(classes)
        result_prediction = []
        centage_prediction = []
        result1 = ""
        result2 = ""
        pourcentage_prediction1 = ""
        pourcentage_prediction2 = ""
        i = 0
        for prediction in classes :
            pred = np.argmax(prediction)
            print(prediction)
            print(pred)
            pred0 , pred1 = prediction
            print(pred0)
            print(pred1)
            # print(prediction[1][0])
            # print(prediction[0][0])
            
            if pred == 1:
                result1 = "c'est bien il porte une masque et "
                pourcentage_prediction1 = 'la pourcentage de prediction ' + str( int(pred1 * 100) ) + '%'
                # print(prediction[0][1])
                result_prediction.append(result1 + pourcentage_prediction1)
                i+= 1
            else:
                result2 = "c'est grave il ne porte pas une masque et "
                pourcentage_prediction2 = 'la pourcentage de prediction ' + str( int(pred0 * 100) ) + '%'
                # print(prediction[0][0])
                result_prediction.append(result2 + pourcentage_prediction2)
                i+= 1

        print(result1)
        print(result2)
        result = result1 + result2
        print(result_prediction)
        self.L1.setText(result)

    

    def predictionMultiImageparFunctionalModel(self):
        # load a model => charger le model model.h5
        model = load_model("Functionalmodel.h5")

        # path 
        self._base_dir = os.getcwd()
        self._images_dir = os.path.join(self._base_dir, 'test_images')

        # open a file dialog and select the folder path 
        dialog = QFileDialog()
        self._folder_path = dialog.getExistingDirectory(None, 'select Folder')

        # get the list of images in the folder and read using matplotlib and print its shape
        self.list_of_images = os.listdir(self._folder_path)
        self.list_of_images = sorted(self.list_of_images)

        #length of images 
        print('Number of Imagesin the selected folder :{}'.format(len(self.list_of_images)))

        # image folder 
        # folder_path = 'sample_test_images'
        # dimentions of images
        img_size = 50

        # load all images into a list
        images= []
        # prede = []
        # for img in os.listdir(folder_path):
        for img in self.list_of_images:
            # print(os.listdir(folder_path))
            print(img)
            path = os.path.join(self._folder_path, img)
             # loading the image
            img = plt.imread(path)
            # initializing the detector
            detector = MTCNN()
            # Detecting the face in the image
            faces = detector.detect_faces(img)
            # next performing face detector and image pre-processing together
            # reading in the image as a grayscale image
            img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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
            # img = image.load_img(img, target_size=(img_width, img_height))
            # img = image.img_to_array(img)
            # img = img.reshape(-1, 50, 50, 1)
            # img = normalize(img, axis=1)
            # img = np.expand_dims(img, axis=0)
            # pred = model.predict(img)
            # print(pred)
            # prede.append(pred)
            images.append(x)

        # stack up images list to pass for prediction 
        print(len(images))
        images = np.array(images)
        images = np.vstack((images))
        print(images.shape)
        print('yes')
        print(len(images))
        classes = model.predict(images)
        print(classes)
        result_prediction = []
        result1 = ""
        result2 = ""
        pourcentage_prediction1 = ""
        pourcentage_prediction2 = ""
        i = 1
        for prediction in classes :
            pred = np.argmax(prediction)
            print(prediction)
            print(pred)
            pred0 , pred1 = prediction
            print(pred0)
            print(pred1)
            # print(prediction[1][0])
            # print(prediction[0][0])
            
            if pred == 1:
                result1 = "c'est bien il porte une masque et "
                pourcentage_prediction1 = 'la pourcentage de prediction ' + str( int(pred1 * 100) ) + '%'
                # print(prediction[0][1])
                result_prediction.append(result1 + pourcentage_prediction1)
                i+= 1
            else:
                result2 = "c'est grave il ne porte pas une masque et "
                pourcentage_prediction2 = 'la pourcentage de prediction ' + str( int(pred0 * 100) ) + '%'
                # print(prediction[0][0])
                result_prediction.append(result2 + pourcentage_prediction2)
                i+= 1

        print(result1)
        print(result2)
        result = result1 + result2
        print(result_prediction)
        self.L1.setText(result)



    def choisirImage(self):
        path= QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(), 'Images (*.*)')
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
            # img.show()



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
        print('{:.2f} %'.format(prediction[0] * 100))
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

        # if pred == 1:
        #     pourcentage_prediction = 'la pourcentage de prediction {:.2f} %'.foramt(prediction[0][1] * 100)
        # else:
        #     pourcentage_prediction = 'la pourcentage de prediction {:.2f} %'.foramt(prediction[0][0] * 100) 
           
        self.label_functional.setText(pourcentage_prediction)
        print('FonctionalModel')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MyAppp()
    gui.show()
    sys.exit(app.exec_())
