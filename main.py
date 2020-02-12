import imutils
import cv2 as cv
from joblib import dump, load
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import numpy
import glob
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance as dist


from classificationModels import getKNN, getNetwork

getNetwork()

# KONSTANTE
winSize = (64, 64)
new_dataset_location = './datasets/converted'
# koliko frejmova se preskace u videu (da bi program brze radio, i da bi se bolje pokret detektovao)
skipFrames = 23
# kooridnate linije
lineStart = (50, 500)
lineEnd = (1300, 500)

# udaljenost od linija za koju se registruje da je vozilo preslo preko linije
distanceTreshold = 50

# PRACENJE KONTURA
ID = 0  # id konture
vehicles = dict()  # kljuc je id konture, vrednost su podaci o konturi (x,y,w,h)
centers = dict()  # izracunate kooridnate centara za sve konture
counted = set()  # skup id koji su presli preko linija
lastSeen = dict()  # kad su konture poslednji put vidjene

# Dodaje novu konturu u spisak onih koje se prate

TOTAL = 0
TRUCK_KNN = 0
CAR_KNN = 0

TRUCK_NET = 0
CAR_NET = 0

# Za racunanje hog desktiptora
HOG = cv.HOGDescriptor(winSize, (32, 32), (8, 8), (8, 8), 9)


def checkLineCrossing(p):
    p = [p[1], p[0]]
    t1 = numpy.array([lineStart[1], lineStart[0]])
    t2 = numpy.array([lineEnd[1], lineEnd[0]])
    return abs(numpy.cross(t2-t1, p-t1)/numpy.linalg.norm(t2-t1)) < distanceTreshold


knn = getKNN()
network = getNetwork()


def checkCrossing(gray):
    global TOTAL, CAR_KNN, TRUCK_KNN, CAR_NET, TRUCK_NET
    for id in centers:
        if id in counted:
            continue
        if checkLineCrossing(centers[id]):
            det = vehicles[id]
            img = gray[det[1]: det[3]+det[1], det[0]:  det[0]+det[2]]
            img = cv.resize(img, winSize, cv.INTER_CUBIC)
            h = HOG.compute(img)
            counted.add(id)
            img = reshape_data(numpy.array([h]))
            knnPrediction = knn.predict(img)
            netPrediction = network.predict(img)[0]
            if knnPrediction[0] == 0:
                CAR_KNN += 1
            else:
                TRUCK_KNN += 1
            if netPrediction.argmax() == 0:
                CAR_NET += 1
            else:
                TRUCK_NET += 1
            TOTAL += 1


def addContour(c):
    global ID
    vehicles[ID] = c
    centers[ID] = (int(c[0]+c[2]/2), int(c[1]+c[3]/2))
    lastSeen[ID] = 0
    ID = ID+1

# Uklanja konture koje dugo nisu bile primecene


def removeCnts():
    toRemove = []
    for key in lastSeen:
        if lastSeen[key] >= skipFrames*5:
            toRemove.append(key)
    for key in toRemove:
        del lastSeen[key]
        del vehicles[key]
        del centers[key]

# Povezuje konture koje su detektovane u trenutnom frejmu sa onima od ranije


def trackVehicles(cnts):
    # nema nijedno vozilo koje je prethodno detektovano, pa se samo dodaju nova
    if len(vehicles) == 0:
        for cnt in cnts:
            addContour(cnt)
    # ako nema novih vozila, proveravamo koja nisu vidjena dugo vremena i izbacimo te konture
    elif len(cnts) == 0:
        for key in lastSeen:
            lastSeen[key] += skipFrames
        removeCnts()
    # ima i novih i starih, povezuju se
    else:
        allCenters = list(centers.values())
        oldMatched = set()
        newAdded = set()
        newCenters = []
        for cnt in cnts:
            newCenters.append((int(cnt[0]+cnt[2]/2), int(cnt[1]+cnt[3]/2)))
        newCenters = numpy.array(newCenters, dtype="int")
        allIDs = list(vehicles.keys())

        dists = dist.cdist(numpy.array(allCenters), newCenters)

        old = dists.min(axis=1).argsort()
        newCnts = dists.argmin(axis=1)[old]

        for (om, nc) in zip(old, newCnts):

            if om not in oldMatched and nc not in newAdded:

                id = allIDs[om]
                centers[id] = newCenters[nc]
                lastSeen[id] = 0
                vehicles[id] = cnts[nc]
                oldMatched.add(om)
                newAdded.add(nc)
        if dists.shape[0] >= dists.shape[1]:
            leftUnmatched = set(
                range(0, dists.shape[0])).difference(oldMatched)
            for om in leftUnmatched:
                id = allIDs[om]
                lastSeen[id] += skipFrames
            removeCnts()
        else:
            leftNew = set(
                range(0, dists.shape[1])).difference(newAdded)
            for nc in leftNew:

                addContour(cnts[nc])


"""
Funkcija prima putanju do slike, ucitava je, konvertuje u grayscale i radi resize
"""


def loadGray(path):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, winSize)
    return image


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


# Labele slika vozila i pozadine
labels = []
# HOG deskriptori slika vozila i pozadine
images = []

SVM_classifier = None


def SVM(x, y):
    ros = RandomOverSampler(random_state=283)
    print('SVM training started...')
    classifier = SVC(gamma='scale', degree=3, probability=True, kernel='poly')
    x = reshape_data(x)
    # resample
    x, y = ros.fit_resample(x, y)
    classifier.fit(x, y)
    return classifier


try:
    SVM_classifier = load('./svm.cl')
except:

    for image_path in glob.glob(new_dataset_location+'/positive/' + "*.jpg"):
        img = loadGray(image_path)
        computedHOG = HOG.compute(img)
        images.append(computedHOG)
        labels.append('vehicle')
    for image_path in glob.glob(new_dataset_location+'/negative/' + "*.png"):
        img = loadGray(image_path)
        computedHOG = HOG.compute(img)
        images.append(computedHOG)
        labels.append('background')

    SVM_classifier = SVM(numpy.array(images), numpy.array(labels))
    try:
        dump(SVM_classifier, './svm.cl')
    except:
        pass


video = cv.VideoCapture('validation.mp4')

# cascade_src = 'cascade.xml'
# car_cascade = cv.CascadeClassifier(cascade_src)

cv.namedWindow('frame') #pozicija prozora
cv.moveWindow('frame',0,0)


# Frejmovi iz videa cija se razlika uzima prilikom detekcije pokreta
first = second = None

i = 0  # broj frejma, uzima se svaki gde je i%skipFrames==0
while(video.isOpened()):
    i += 1
    if i == skipFrames:
        i = 0
    else:
        continue
    _, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    img = frame.copy()
    img2 = frame.copy()

    try:
        first = second.copy()
    except:
        pass
    second = gray.copy()
    try:
        f = cv.GaussianBlur(first, (13, 13), 0)
        s = cv.GaussianBlur(second, (13, 13), 0)
    except:
        continue
    gray2 = cv.absdiff(f, s)

    gray2 = cv.GaussianBlur(gray2, (21, 21), 0)
    # plt.imshow(img)
    # plt.show()

    cv.imshow('diff',gray2)

    # thresh = cv.threshold(gray2, 20, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    thresh = cv.threshold(gray2, 6, 255, cv.THRESH_BINARY)[1]

    # plt.imshow(thresh, 'gray')
    # plt.show()

    thresh = cv.morphologyEx(
        thresh, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)
    thresh = cv.morphologyEx(
        thresh, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (8, 8)), iterations=3)
    thresh = cv.morphologyEx(
        thresh, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)
    thresh = cv.morphologyEx(
        thresh, cv.MORPH_DILATE, cv.getStructuringElement(
            cv.MORPH_RECT, (2, 2)), iterations=2)

    cnts, hier = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE)
    cv.imshow('thresh', thresh)
#    cv.namedWindow('thresh')
#    cv.moveWindow('thresh',0,0)

    detected = []
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        if w < 69 or h < 69:
            continue
        added = False
        if w > 160 or h > 160:
            t = gray2[y:y+h, x:x+w]
            t = cv.threshold(t, 25, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
            t = cv.morphologyEx(
                t, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=1)
            t = cv.morphologyEx(
                t, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (8, 8)), iterations=2)
            cnts2, hier = cv.findContours(t, cv.RETR_EXTERNAL,
                                             cv.CHAIN_APPROX_SIMPLE)
            for c in cnts2:
                (x2, y2, w2, h2) = cv.boundingRect(c)
                if w2 > 69 and h2 > 69:
                    added = True
                    detected.append((x+x2, y+y2, w2, h2))

            if(added):
                continue

        detected.append((x, y, w, h))
    new = []
    for i in range(len(detected)): #delim dva spojena, crno-beli kvadrat kad nije dovoljno simetrican da se podeli
        d = detected[i]
        if d[2] > 140:
            left = thresh[d[1]:d[1]+d[3], d[0]:d[0]+int(d[2]/2)]
            right = numpy.fliplr(
                thresh[d[1]:d[1]+d[3], d[0]+int(d[2]/2):d[0]+d[2]])
            right = cv.resize(right, (left.shape[1], left.shape[0]))
            diff = left != right
            diff = numpy.array(diff).astype('int')
            diff = diff.sum()
            if float(diff)/float(right.shape[0]*right.shape[1]) > 0.5:
                detected[i] = (d[0], d[1], int(d[2]/2), d[3])
                new.append((d[0]+int(d[2]/2), d[1], int(d[2]/2), d[3]))
    for i in range(len(new)):
        detected.append(new[i])
    filtered = []

    for det in detected:
        img = gray[det[1]: det[3]+det[1], det[0]:  det[0]+det[2]]
        img = cv.resize(img, winSize)
        hog_d = reshape_data(numpy.array([HOG.compute(img)]))
        if SVM_classifier.predict_proba(hog_d)[0][1] > 0.6:
            filtered.append(det)
    trackVehicles(filtered)
    checkCrossing(gray)

    for key in vehicles:
        c = vehicles[key]
        if key in counted:
            cv.rectangle(img2, (c[0], c[1]),
                         (c[0]+c[2], c[1]+c[3]), (255, 0, 0), 2)
        else:
            cv.rectangle(img2, (c[0], c[1]),
                         (c[0]+c[2], c[1]+c[3]), (0, 0, 255), 2)

    cv.line(img2, lineStart,
            lineEnd, (255, 255, 0), 2)
    cv.putText(img2, 'KNN:      CAR:  '+str(CAR_KNN)+'    TRUCK:  '+str(TRUCK_KNN), (0, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.putText(img2, 'NETWORK: CAR:  '+str(CAR_NET)+'    TRUCK:  '+str(TRUCK_NET), (0, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow('frame', img2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
