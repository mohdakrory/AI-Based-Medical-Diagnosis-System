from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QRadioButton,
    QPushButton,
    QLabel,
    QFileDialog,
    QStackedWidget,
    QAction,
)
from PyQt5 import QtCore
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QFont
import sys
from keras.models import load_model, Model
import cv2
import numpy as np
from keras import backend as K
from keras.utils import img_to_array, array_to_img
from tensorflow import (
    GradientTape,
    argmax,
    reduce_mean,
    newaxis,
    squeeze,
    maximum,
    math,
)
import matplotlib.cm as cm
from torch import hub
from wfdb import rdrecord
from scipy.signal import resample
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
from os import environ
from socket import gethostbyname, gethostname
import paho.mqtt.client as mqtt
from AFR import AFR
from AFRLayer1D import AFRLayer1D

environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # Load the ui file
        uic.loadUi("home_page.ui", self)

        self.stackedwidget = self.findChild(QStackedWidget, "stackedwidget")

        # home page elements
        self.home_btn = self.findChild(QAction, "actionhome")
        self.brain_btn = self.findChild(QPushButton, "brain_btn")
        self.covid_btn = self.findChild(QPushButton, "covid_btn")
        self.ecg_btn = self.findChild(QPushButton, "ecg_btn")
        self.chest_btn = self.findChild(QPushButton, "chest_btn")
        self.skin_btn = self.findChild(QPushButton, "skin_btn")
        self.guide = self.findChild(QLabel, "guide")

        # navigation functions
        self.home_btn.triggered.connect(self.to_home)
        self.brain_btn.clicked.connect(self.to_brain)
        self.covid_btn.clicked.connect(self.to_covid)
        self.ecg_btn.clicked.connect(self.to_ecg)
        self.chest_btn.clicked.connect(self.to_chest)
        self.skin_btn.clicked.connect(self.to_skin)

        #################################################
        # elements of each service page
        #################################################

        # brain tumor page elements
        self.brain_choose_btn = self.findChild(QPushButton, "brain_choose_btn")
        self.brain_choose_image = self.findChild(QLabel, "brain_choose_image")
        self.brain_diagnose_btn = self.findChild(QPushButton, "brain_diagnose_btn")
        self.brain_result = self.findChild(QLabel, "brain_result")
        self.brain_choose_btn.clicked.connect(self.choose_brain)
        self.brain_diagnose_btn.clicked.connect(self.diagnose_brain)
        self.brain_seg_result = self.findChild(QLabel, "brain_seg_result")

        # covid page elements
        self.covid_choose_btn = self.findChild(QPushButton, "covid_choose_btn")
        self.covid_choose_image = self.findChild(QLabel, "covid_choose_image")
        self.covid_gradcam = self.findChild(QLabel, "covid_gradcam")
        self.covid_diagnose_btn = self.findChild(QPushButton, "covid_diagnose_btn")
        self.covid_result = self.findChild(QLabel, "covid_result")
        self.covid_choose_btn.clicked.connect(self.choose_covid)
        self.covid_diagnose_btn.clicked.connect(self.diagnose_covid)

        # ecg page elements
        self.ecg_choose_btn = self.findChild(QPushButton, "ecg_choose_btn")
        self.ecg_choose_image = self.findChild(QLabel, "ecg_choose_image")
        self.ecg_diagnose_btn = self.findChild(QPushButton, "ecg_diagnose_btn")
        self.ecg_result = self.findChild(QLabel, "ecg_result")
        self.ecg_choose_btn.clicked.connect(self.choose_ecg)
        self.ecg_diagnose_btn.clicked.connect(self.diagnose_ecg)
        self.ecg_beat = self.findChild(QLabel, "ecg_beat")
        self.ecg_hw_btn = self.findChild(QRadioButton, "ecg_hw_btn")
        self.ecg_hw_btn.clicked.connect(self.diagnose_hw_ecg)

        # chest page elements
        self.chest_choose_btn = self.findChild(QPushButton, "chest_choose_btn")
        self.chest_choose_image = self.findChild(QLabel, "chest_choose_image")
        self.chest_diagnose_btn = self.findChild(QPushButton, "chest_diagnose_btn")
        self.chest_result_image = self.findChild(QLabel, "chest_result_image")
        self.chest_choose_btn.clicked.connect(self.choose_chest)
        self.chest_diagnose_btn.clicked.connect(self.diagnose_chest)

        # skin cancer page
        self.skin_choose_btn = self.findChild(QPushButton, "skin_choose_btn")
        self.skin_choose_image = self.findChild(QLabel, "skin_choose_image")
        self.skin_diagnose_btn = self.findChild(QPushButton, "skin_diagnose_btn")
        self.skin_result = self.findChild(QLabel, "skin_result")
        self.skin_result_image = self.findChild(QLabel, "skin_result_image")
        self.skin_choose_btn.clicked.connect(self.choose_skin)
        self.skin_diagnose_btn.clicked.connect(self.diagnose_skin)

        #######################################

        # initialization
        # home page
        self.stackedwidget.setCurrentIndex(0)
        # set input samples for each page
        self.brain_seg_result.setPixmap(QPixmap("images/icons/brain-samples.jpg"))
        self.covid_gradcam.setPixmap(QPixmap("images/icons/covid-samples.jpg"))
        self.chest_result_image.setPixmap(QPixmap("images/icons/chest-samples.jpg"))
        self.skin_result_image.setPixmap(QPixmap("images/icons/skin-samples.jpg"))
        # guide
        self.guide.setText(
            "Welcom to medical diagnosis system, you can choose from our services form the left side"
        )
        self.guide.setAlignment(QtCore.Qt.AlignTop)
        self.show()

    ####################################
    # pages navigation functions
    ####################################

    def reset(self):
        #### reset all elements to initial state

        # covid page
        self.covid_result.setText("Result")
        self.covid_result.setAlignment(QtCore.Qt.AlignCenter)

        self.covid_choose_image.setText("No file choosen, yet")
        self.covid_choose_image.setAlignment(QtCore.Qt.AlignCenter)
        self.covid_choose_image.setFont(QFont("Arial", 26))
        self.covid_gradcam.setPixmap(QPixmap("images/icons/covid-samples.jpg"))

        # brain tumor page
        self.brain_result.setText("Result")
        self.brain_result.setAlignment(QtCore.Qt.AlignCenter)

        self.brain_choose_image.setText("No file choosen, yet")
        self.brain_choose_image.setAlignment(QtCore.Qt.AlignCenter)
        self.brain_choose_image.setFont(QFont("Arial", 26))
        self.brain_seg_result.setPixmap(QPixmap("images/icons/brain-samples.jpg"))

        # ecg page
        self.ecg_result.setText("Result")
        self.ecg_result.setAlignment(QtCore.Qt.AlignCenter)

        self.ecg_choose_image.setText("No file choosen, yet")
        self.ecg_choose_image.setAlignment(QtCore.Qt.AlignCenter)
        self.ecg_choose_image.setFont(QFont("Arial", 26))
        self.ecg_beat.clear()

        # chest page
        self.chest_choose_image.setText("No file choosen, yet")
        self.chest_choose_image.setAlignment(QtCore.Qt.AlignCenter)
        self.chest_choose_image.setFont(QFont("Arial", 26))
        self.chest_result_image.setPixmap(QPixmap("images/icons/chest-samples.jpg"))

        # skin page
        self.skin_result.setText("Result")
        self.skin_result.setAlignment(QtCore.Qt.AlignCenter)

        self.skin_choose_image.setText("No file choosen, yet")
        self.skin_choose_image.setAlignment(QtCore.Qt.AlignCenter)
        self.skin_choose_image.setFont(QFont("Arial", 26))
        self.skin_result_image.setPixmap(QPixmap("images/icons/skin-samples.jpg"))

    def to_home(self):
        self.stackedwidget.setCurrentIndex(0)
        # reset pages
        self.reset()
        # guide
        self.guide.setText(
            "Welcom to medical diagnosis system, you can choose from our services form the left side"
        )
        self.guide.setAlignment(QtCore.Qt.AlignTop)

    def to_brain(self):
        self.stackedwidget.setCurrentIndex(1)
        self.guide.setText(
            "Differentiate between Meningioma, Glioma, pituitary and the normal case from MRI images\nYou are expected to upload an RGB image to be able to utilize the segmentation service, you can see samples of the input images on the right side\nClick on 'Choose a file' to upload your image then click 'Diagnose'\nThe diagnosis result will be shown in the right bottom blue section along with the confidance of that diagnosis\nThe segmentation result will be shown in samples section "
        )
        self.guide.setAlignment(QtCore.Qt.AlignTop)
        # reset pages
        self.reset()

    def to_covid(self):
        self.stackedwidget.setCurrentIndex(2)
        self.guide.setText(
            "Differentiate between Covid-19, Pneumonia and the normal case from chest x-ray images\nYou are expected to upload an RGB image to be able to use this service, you can see samples of the input images on the right side\nClick on 'Choose a file' to upload your image then click 'Diagnose'\nThe diagnosis result will be shown in the right bottom blue section along with the confidance of that diagnosis\nAn image with a highlighted area on which the diagnosis was based will appear in the samples section"
        )
        self.guide.setAlignment(QtCore.Qt.AlignTop)
        # reset pages
        self.reset()

    def to_ecg(self):
        self.stackedwidget.setCurrentIndex(3)
        self.guide.setText(
            "Differentiate between Supraventricular premature beat, Premature ventricular contraction, Fusion of ventricular and Normal Beat from ECG signal\nYou are expected to upload a biomedical signal file (.dat or .mat) and make sure to include the header file (.hea) in the same direction as the first file\nClick on 'Choose a file' to upload your file then click 'Diagnose'\nThe diagnosis result will be shown in the right bottom blue section along with the confidance of that diagnosis\nThe extracted heartbeat on which the diagnosis was based will be highlighted in red "
        )
        self.guide.setAlignment(QtCore.Qt.AlignTop)
        # reset pages
        self.reset()

    def to_chest(self):
        self.stackedwidget.setCurrentIndex(4)
        self.guide.setText(
            "Localize and detect chest abnormalities from chest x-ray images \nYou are expected to upload an RGB image to be able to use this service, you can see samples of the input images on the right side\nClick on 'Choose a file' to upload your image then click 'Diagnose'\nAn image with a bounding box, diagnosis and confidance of each finding will appear in samples section"
        )
        self.guide.setAlignment(QtCore.Qt.AlignTop)
        # reset pages
        self.reset()

    def to_skin(self):
        self.stackedwidget.setCurrentIndex(5)
        self.guide.setText(
            "Differentiate between Benign and Malignant skin canser from skin images\nYou are expected to upload an RGB image to be able to use this service, you can see samples of the input images on the right side\nClick on 'Choose a file' to upload your image then click 'Diagnose'\nThe diagnosis result will be shown in the right bottom blue section along with the confidance of that diagnosis"
        )
        self.guide.setAlignment(QtCore.Qt.AlignTop)
        # reset pages
        self.reset()

    #################################################################
    # choosing a file in each page
    #################################################################

    # choose an image in the brain tumor page
    def choose_brain(self):
        self.brain_path = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "c:\\gui\\images",
            "All Files (*);;jpg files (*.jpg);; PNG Files (*.png)",
        )
        if self.brain_path[0]:
            self.brain_pixmap = QPixmap(self.brain_path[0])
            self.brain_choose_image.setPixmap(self.brain_pixmap)
            self.brain_result.setText("Result")
            self.brain_result.setAlignment(QtCore.Qt.AlignCenter)
            self.brain_seg_result.setPixmap(QPixmap("images/icons/brain-samples.jpg"))

        else:
            self.brain_result.setText("Result")
            self.brain_result.setAlignment(QtCore.Qt.AlignCenter)
            self.brain_choose_image.setText("No file choosen, yet")
            self.brain_choose_image.setAlignment(QtCore.Qt.AlignCenter)
            self.brain_choose_image.setFont(QFont("Arial", 26))
            self.brain_seg_result.setPixmap(QPixmap("images/icons/brain-samples.jpg"))

    # choose an image in the covid page
    def choose_covid(self):
        self.covid_path = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "c:\\gui\\images",
            "All Files (*);;jpg files (*.jpg);; PNG Files (*.png)",
        )
        if self.covid_path[0]:
            self.covid_pixmap = QPixmap(self.covid_path[0])
            self.covid_choose_image.setPixmap(self.covid_pixmap)
            self.covid_result.setText("Result")
            self.covid_result.setAlignment(QtCore.Qt.AlignCenter)
            self.covid_gradcam.setPixmap(QPixmap("images/icons/covid-samples.jpg"))
        else:
            self.covid_result.setText("Result")
            self.covid_result.setAlignment(QtCore.Qt.AlignCenter)
            self.covid_choose_image.setText("No file choosen, yet")
            self.covid_choose_image.setAlignment(QtCore.Qt.AlignCenter)
            self.covid_choose_image.setFont(QFont("Arial", 26))
            self.covid_gradcam.setPixmap(QPixmap("images/icons/covid-samples.jpg"))

    # choose an image in the heartbeat page
    def choose_ecg(self):
        self.ecg_choose_image.setText("Reading signal")
        self.ecg_choose_image.setAlignment(QtCore.Qt.AlignCenter)
        self.ecg_choose_image.setFont(QFont("Arial", 26))
        self.ecg_path = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "c:\\gui\\images",
            "dat files (*.dat);; mat Files (*.mat)",
        )
        if self.ecg_path[0]:
            self.ecg_path = self.ecg_path[0].split(".")[0]
            record = rdrecord(
                record_name=self.ecg_path,
                sampfrom=0,
                channels=[0],
                physical=False,
                m2s=True,
                smooth_frames=False,
                ignore_skew=False,
                return_res=16,
                force_channels=True,
                channel_names=None,
                warn_empty=False,
            )

            ecg_signal = record.e_d_signal[0]
            self.fs = record.fs
            new_fs = 125
            ecg_signal = resample(ecg_signal, int(len(ecg_signal) * new_fs / self.fs))
            ecg_signal = ecg_signal[0 : 4 * new_fs]
            self.ecg_signal = (ecg_signal - np.min(ecg_signal)) / (
                np.max(ecg_signal) - np.min(ecg_signal)
            )

            plt.ioff()
            fig = plt.figure(figsize=(28, 12))
            plt.plot(self.ecg_signal, linewidth=2)
            plt.axis("off")
            plt.savefig("images/ecg.jpg", bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            ecg_choose = QPixmap("images/ecg.jpg")
            self.ecg_choose_image.setPixmap(ecg_choose)
            self.ecg_result.setText("Result")
            self.ecg_result.setAlignment(QtCore.Qt.AlignCenter)
            self.ecg_beat.clear()
        else:
            self.ecg_result.setText("Result")
            self.ecg_result.setAlignment(QtCore.Qt.AlignCenter)
            self.ecg_choose_image.setText("No file choosen, yet")
            self.ecg_choose_image.setAlignment(QtCore.Qt.AlignCenter)
            self.ecg_choose_image.setFont(QFont("Arial", 26))
            self.ecg_beat.clear()

    def choose_chest(self):
        self.chest_path = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "c:\\gui\\images",
            "All Files (*);;jpg files (*.jpg);; PNG Files (*.png)",
        )
        if self.chest_path[0]:
            self.chest_pixmap = QPixmap(self.chest_path[0])
            self.chest_choose_image.setPixmap(self.chest_pixmap)
            self.chest_result_image.setPixmap(QPixmap("images/icons/chest-samples.jpg"))
        else:
            self.chest_choose_image.setText("No file choosen, yet")
            self.chest_choose_image.setAlignment(QtCore.Qt.AlignCenter)
            self.chest_choose_image.setFont(QFont("Arial", 26))
            self.chest_result_image.setPixmap(QPixmap("images/icons/chest-samples.jpg"))

    def choose_skin(self):
        self.skin_path = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "c:\\gui\\images",
            "All Files (*);;jpg files (*.jpg);; PNG Files (*.png)",
        )
        if self.skin_path[0]:
            self.skin_pixmap = QPixmap(self.skin_path[0])
            self.skin_choose_image.setPixmap(self.skin_pixmap)
            self.skin_result.setText("Result")
            self.skin_result.setAlignment(QtCore.Qt.AlignCenter)
            self.skin_result_image.setPixmap(QPixmap("images/icons/skin-samples.jpg"))

        else:
            self.skin_result.setText("Result")
            self.skin_result.setAlignment(QtCore.Qt.AlignCenter)
            self.skin_choose_image.setText("No file choosen, yet")
            self.skin_choose_image.setAlignment(QtCore.Qt.AlignCenter)
            self.skin_choose_image.setFont(QFont("Arial", 26))
            self.skin_result_image.setPixmap(QPixmap("images/icons/skin-samples.jpg"))

    #######################################################
    # diagnosis functions
    #######################################################

    def diagnose_brain(self):
        img = cv2.imread(self.brain_path[0])
        model = load_model("brain_best_model.h5")
        img = cv2.resize(img, (64, 64))
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img / 255
        pred1 = model.predict(img.reshape(1, 64, 64, 1), batch_size=1, verbose=0)
        pred = pred1.argmax()
        brain_labels = {0: "glioma", 1: "meningioma", 2: "no tumor", 3: "pituitary"}
        x = brain_labels.get(pred, "Unknown")
        y = x + ", " + (pred1[0][pred] * 100).astype("str")[:6] + "%"
        self.brain_result.setText(y)
        self.brain_result.setAlignment(QtCore.Qt.AlignCenter)
        # tumor segmentation
        img1 = cv2.imread(self.brain_path[0])
        img2 = cv2.imread(self.brain_path[0])

        if x != "no tumor" and len(img1.shape) == 3:
            seg_model = load_model(
                "brain_seg_best_model.h5",
                custom_objects={
                    "iou_coef": self.iou_coef,
                    "dice_coef": self.dice_coef,
                    "dice_loss": self.dice_loss,
                },
            )
            img1 = cv2.resize(img1, (256, 256))
            img1 = img1 / 255
            seg = seg_model.predict(
                img1.reshape(1, 256, 256, 3), batch_size=1, verbose=0
            )
            seg = np.squeeze(seg)
            img2 = cv2.resize(img2, (256, 256))
            img2[seg > 0.5] = (0, 0, 255)
            cv2.imwrite("images/seg.jpg", img2)
            seg_result = QPixmap("images/seg.jpg")
            self.brain_seg_result.setPixmap(seg_result)
        elif x != "no tumor" and len(img1.shape) == 2:
            self.brain_seg_result.setText(
                "unable to segment this image as it is not RGB"
            )
            self.brain_seg_result.setFont(QFont("Arial", 26))

    def diagnose_covid(self):
        img = cv2.imread(self.covid_path[0])
        if len(img.shape) == 3:
            img = cv2.resize(img, (224, 224))
            img = img / 255
            model = load_model("covid_best_model.h5", custom_objects={"AFR": AFR})
            pred1 = model.predict(img.reshape(1, 224, 224, 3), batch_size=1, verbose=0)
            pred = pred1.argmax()

            covid_labels = {0: "COVID-19", 1: "Normal", 2: "Pneumonia"}
            x = (
                covid_labels.get(pred, "Unknown")
                + ", "
                + (pred1[0][pred] * 100).astype("str")[:6]
                + "%"
            )
            # set final prediction to the result label in covid page
            self.covid_result.setText(x)
            self.covid_result.setAlignment(QtCore.Qt.AlignCenter)

            if covid_labels.get(pred, "Unknown") != "Normal":
                gmodel = load_model("covid_best_model.h5", custom_objects={"AFR": AFR})
                gmodel.layers[-1].activation = None
                last_conv_layer_name = "afr_3"
                img1 = cv2.imread(self.covid_path[0])
                img1 = cv2.resize(img1, (224, 224))
                self.gradcam(img1, gmodel, last_conv_layer_name)
                self.covid_gradcam.setPixmap(QPixmap("images/cam.jpg"))

        elif len(img.shape) == 2:
            self.covid_gradcam.setText("unable to diagnose this image as it is not RGB")
            self.covid_gradcam.setFont(QFont("Arial", 26))

    def gradcam(
        self,
        img,
        model,
        last_conv_layer_name,
        alpha=0.4,
        cam_path="images/cam.jpg",
        pred_index=None,
    ):
        array = img_to_array(img)
        img_array = np.expand_dims(array, axis=0)
        img_array = img_array / 255

        grad_model = Model(
            model.inputs, [model.get_layer(last_conv_layer_name).input, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., newaxis]
        heatmap = squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = maximum(heatmap, 0) / math.reduce_max(heatmap)

        # img1 = load_img(img_path)
        img1 = array  # img_to_array(img1)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img1.shape[1], img1.shape[0]))
        jet_heatmap = img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img1
        superimposed_img = array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)

    def diagnose_ecg(self):
        self.ecg_beat.setText("Diagnosing signal")
        self.ecg_beat.setAlignment(QtCore.Qt.AlignCenter)
        self.ecg_beat.setFont(QFont("Arial", 26))

        # new_fs = 125
        # resampled_signal = resample(self.ecg_signal, int(len(self.ecg_signal) * new_fs / self.fs))
        # window_size = 4 * new_fs
        # window_start = 0
        # window_end = window_start + window_size
        # ecg_window = resampled_signal[window_start:window_end]

        # ecg_norm = (self.ecg_signal - np.min(self.ecg_signal)) / (
        #     np.max(self.ecg_signal) - np.min(self.ecg_signal)
        # )
        out = ecg.hamilton_segmenter(signal=self.ecg_signal, sampling_rate=125)
        rpeaks = ecg.correct_rpeaks(
            signal=self.ecg_signal, rpeaks=out["rpeaks"], sampling_rate=125, tol=0.08
        )
        rr_intervals = np.diff(rpeaks)
        T = np.median(rr_intervals)

        rpeak = rpeaks[0][0]

        start = rpeak
        end = int(rpeak + T * 1.2)

        beat = self.ecg_signal[start:end]

        fixed_length = 187
        padding_length = fixed_length - len(beat)
        padded_beat = np.concatenate((beat, np.zeros(padding_length)))

        plt.ioff()
        fig = plt.figure(figsize=(28, 12))
        plt.plot(padded_beat, linewidth=4, color="red")
        plt.axis("off")
        plt.savefig("images/beat.jpg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        idecies = []
        values = []

        plt.ioff()
        fig = plt.figure(figsize=(28, 12))
        plt.plot(self.ecg_signal, linewidth=2)
        for i in range(start, end, 1):
            idecies.append(i)
            values.append(self.ecg_signal[i])
        plt.plot(idecies, values, color="red")
        plt.axis("off")
        plt.savefig("images/sig.jpg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        beat = QPixmap("images/beat.jpg")
        self.ecg_beat.setPixmap(beat)

        new_sig = QPixmap("images/sig.jpg")
        self.ecg_choose_image.setPixmap(new_sig)

        model = load_model(
            "ecg_data_best_model.h5", custom_objects={"AFRLayer1D": AFRLayer1D}
        )
        pred1 = model.predict(padded_beat.reshape(1, 187, 1), batch_size=1, verbose=0)
        pred = pred1.argmax()

        ecg_labels = {
            0: "Normal Beat",
            1: "Supraventricular premature beat",
            2: "Premature ventricular contraction",
            3: "Fusion of ventricular",
            4: "Unknown beat",
        }

        x = (
            ecg_labels.get(pred, "Unknown")
            + ", "
            + (pred1[0][pred] * 100).astype("str")[:6]
            + "%"
        )
        self.ecg_result.setText(x)
        self.ecg_result.setAlignment(QtCore.Qt.AlignCenter)

    def diagnose_hw_ecg(self):
        self.ecg_choose_image.clear()
        self.ecg_result.setText("Result")
        self.ecg_beat.clear()
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        local_ip = gethostbyname(gethostname())
        client.connect(local_ip, 1883)

        # Start the MQTT client loop to listen for messages
        client.loop_forever()
        print("Received messages:", self.recieved_ecg)
        print("message length:", len(self.recieved_ecg))

        new_fs = 125
        fs = int((len(self.recieved_ecg) / 4))
        print(fs)
        self.recieved_ecg = resample(
            self.recieved_ecg, int(len(self.recieved_ecg) * new_fs / fs)
        )

        ecg_norm = (self.recieved_ecg - np.min(self.recieved_ecg)) / (
            np.max(self.recieved_ecg) - np.min(self.recieved_ecg)
        )
        out = ecg.hamilton_segmenter(signal=ecg_norm, sampling_rate=125)
        rpeaks = ecg.correct_rpeaks(
            signal=ecg_norm, rpeaks=out["rpeaks"], sampling_rate=125, tol=0.08
        )
        rr_intervals = np.diff(rpeaks)
        T = np.median(rr_intervals)
        print(T)
        rpeak = rpeaks[0][0]

        start = rpeak
        end = int(rpeak + T * 1.2)

        beat = ecg_norm[start:end]

        fixed_length = 187
        padding_length = fixed_length - len(beat)
        padded_beat = np.concatenate((beat, np.zeros(padding_length)))

        plt.ioff()
        fig = plt.figure(figsize=(28, 12))
        plt.plot(padded_beat, linewidth=5, color="red")
        plt.axis("off")
        plt.savefig("images/beat.jpg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        beat = QPixmap("images/beat.jpg")
        self.ecg_beat.setPixmap(beat)

        idecies = []
        values = []

        plt.ioff()
        fig = plt.figure(figsize=(28, 12))
        plt.plot(ecg_norm, linewidth=3)
        for i in range(start, end, 1):
            idecies.append(i)
            values.append(ecg_norm[i])
        plt.plot(idecies, values, linewidth=3, color="red")
        plt.axis("off")
        plt.savefig("images/sig.jpg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        sig = QPixmap("images/sig.jpg")
        self.ecg_choose_image.setPixmap(sig)

        model = load_model(
            "ecg_data_best_model.h5", custom_objects={"AFRLayer1D": AFRLayer1D}
        )
        pred1 = model.predict(padded_beat.reshape(1, 187, 1), batch_size=1, verbose=0)
        pred = pred1.argmax()

        ecg_labels = {
            0: "Normal Beat",
            1: "Supraventricular premature beat",
            2: "Premature ventricular contraction",
            3: "Fusion of ventricular",
            4: "Unknown beat",
        }

        x = (
            ecg_labels.get(pred, "Unknown")
            + ", "
            + (pred1[0][pred] * 100).astype("str")[:6]
            + "%"
        )
        self.ecg_result.setText(x)
        self.ecg_result.setAlignment(QtCore.Qt.AlignCenter)
        self.ecg_hw_btn.setChecked(False)

    def on_connect(self, client, userdata, flags, rc):
        self.rc = rc
        if rc == 0:
            print("Connected to MQTT broker!")
            client.subscribe("m")
            self.recieved_ecg = []
        else:
            print("Failed to connect to MQTT broker")

    def on_message(self, client, userdata, msg):
        print(f"Received message: {msg.payload.decode('utf-8')}")
        try:
            self.recieved_ecg.append(float(msg.payload.decode("utf-8")))
        except ValueError:
            print("not a number")

        if msg.payload.decode("utf-8") == "a":
            client.disconnect()

    def diagnose_chest(self):
        img = cv2.imread(self.chest_path[0])
        if len(img.shape) == 3:
            img = cv2.resize(img, (640, 640))

            yolo = hub.load("ultralytics/yolov5", "custom", path="best.pt")
            results = yolo(img)

            boxes = results.xyxy[0].numpy()
            class_indices = results.pred[0][:, -1].long().cpu().tolist()
            class_labels = [results.names[i] for i in class_indices]
            scores = results.pred[0][:, 4].cpu().tolist()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box[:4].astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{class_labels[i]}: {scores[i]:.2f}"

                # Get the size of the label text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # Draw a white rectangle behind the label text
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1 - 5), (255, 255, 255), -1)
                # Draw the label text in red color
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            cv2.imwrite("images/chest.png", img)
            chest_result = QPixmap("images/chest.png")
            self.chest_result_image.setPixmap(chest_result)
        elif len(img.shape) == 2:
            self.chest_result_image.setText(
                "unable to diagnose this image as it is not RGB"
            )
            self.chest_result_image.setFont(QFont("Arial", 26))

    def diagnose_skin(self):
        img = cv2.imread(self.skin_path[0])
        if len(img.shape) == 3:
            img = cv2.resize(img, (224, 224))
            img = img / 255
            model = load_model("skin_best_model.h5", custom_objects={"AFR": AFR})
            pred1 = model.predict(img.reshape(1, 224, 224, 3), batch_size=1, verbose=0)
            pred = (pred1 > 0.5).astype(int)
            skin_labels = {0: "Benign", 1: "Malignant"}
            if pred1 > 0.5:
                x = (
                    skin_labels.get(pred[0][0], "Unknown")
                    + ", "
                    + (pred1[0][0] * 100).astype("str")[:6]
                    + "%"
                )
            else:
                x = (
                    skin_labels.get(pred[0][0], "Unknown")
                    + ", "
                    + ((1 - pred1[0][0]) * 100).astype("str")[:6]
                    + "%"
                )

            # set final prediction to the result label in covid page
            self.skin_result.setText(x)
            self.skin_result.setAlignment(QtCore.Qt.AlignCenter)
        elif len(img.shape) == 2:
            self.skin_result_image.setText(
                "unable to diagnose this image as it is not RGB"
            )
            self.chest_result_image.setFont(QFont("Arial", 26))

    ###############################################
    # custom objects in the brain segmentation model
    ###############################################

    def dice_coef(self, y_true, y_pred, smooth=100):
        y_true_flatten = K.flatten(y_true)
        y_pred_flatten = K.flatten(y_pred)

        intersection = K.sum(y_true_flatten * y_pred_flatten)
        union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
        return (2 * intersection + smooth) / (union + smooth)

    def dice_loss(self, y_true, y_pred, smooth=100):
        return -self.dice_coef(y_true, y_pred, smooth)

    def iou_coef(self, y_true, y_pred, smooth=100):
        intersection = K.sum(y_true * y_pred)
        sum = K.sum(y_true + y_pred)
        iou = (intersection + smooth) / (sum - intersection + smooth)
        return iou


# Initialize The App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
