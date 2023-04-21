import tkinter as tk
from tkinter import ttk
import pyautogui as pg
import cv2
import os
import numpy as np
from PIL import Image
import yaml
from base64 import b64encode, b64decode
import hashlib
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from cryptography.fernet import Fernet
from tkinter import Tk 
from tkinter.filedialog import askopenfilename
import os
import glob


root=tk.Tk()
root.geometry('640x480')
root.resizable(True,True)
root.title('Facial File Security v1.0')


def trainerfunc(): # Save Data
  
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

   
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids


    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    
    recognizer.write('trainer/trainer.yml')
    path = os.getcwd()
    files = glob.glob(path + '/datasset')
    for f in files:
        os.remove(f)

def encrypterprimaryfunc(): #Encrypter Function
    key = Fernet.generate_key()
 
    
    with open('filekey.key', 'wb') as filekey:
        filekey.write(key)
    
    with open('filekey.key', 'rb') as filekey:
        key = filekey.read()
    Tk().withdraw() 
    filename = askopenfilename() 
    print(filename)

    fernet = Fernet(key)
    
    
    with open(filename, 'rb') as file:
        original = file.read()
    with open('filekey.key','rb') as keyfile:
        pw = keyfile.read()
    
    encrypted = fernet.encrypt(original)
    

    with open(filename, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)

    with open('trainer/trainer.yml','rb') as trainerog:
        og = trainerog.read()
        encrypted2 = fernet.encrypt(og)
    
    print(encrypted2)
    with open('trainer/trainer.yml','wb') as trainer:
        trainer.write(encrypted2)
    pg.alert(text="Your file has been encrypted!",title='Alert')
    pg.alert(text="Password for your file is stored in a file named FILEKEY.KEY.",title='Alert')

def recognizer(): #Function to recongize faces
    Tk().withdraw() 
    filename = askopenfilename() 
    print(filename)
    with open('filekey.key','rb') as decryptkey:
        x = decryptkey.read()
        fernet = Fernet(x)
    with open('trainer/trainer.yml','rb') as trainerdecrypt:
        z = trainerdecrypt.read()
        trainerans = fernet.decrypt(z)
    with open('trainer/trainer.yml','wb') as trainfinal:
        trainfinal.write(trainerans)


    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX


    id = 0




    cam = cv2.VideoCapture(0)
    cam.set(3, 1920) # set video widht
    cam.set(4, 1080) # set video height


    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        ret, img =cam.read()
        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            
            if (confidence < 100):
                id = id
                confidence = "  {0}%".format(round(100 - confidence))
                with open(filename,'rb') as decryptedfile:
                    y = decryptedfile.read()
                    originalfile = fernet.decrypt(y)
                with open(filename,'wb') as originalwriter:
                    originalwriter.write(originalfile)
                pg.alert(text="File decrypted!",title="Decryption success")

                exit()
                 
            else:
                id = "Unknown Entity"
                confidence = "  {0}%".format(round(100 - confidence))
                pg.alert(text="Unknown Entity trying to access the file. \n Exiting program...",title="WARNING")
                exit()
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('Face recognizer',img) 

        k = cv2.waitKey(10) & 0xff 
        if k == 27:
            break
    
    cam.release()
    cv2.destroyAllWindows()

def recorfunc(): #Function for training data

    cam = cv2.VideoCapture(0)
    cam.set(3, 1920) # set video width
    cam.set(4, 1080)# set video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    face_id = pg.prompt(text='Enter the user id, for first user enter 1', title='Face Trainer' , default='')

    pg.alert(text="\n Capturing images, Click OK to continue and look at the camera",title='Face Trainer',button='OK')
    
    count = 0

    while(True):

        ret, img = cam.read()
        img = cv2.flip(img, 1) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('Face Trainer', img)

        k = cv2.waitKey(100) & 0xff 
        if k == 27:
            break
        elif count >= 30: # o
            break


    pg.alert(text="\n Exiting Program",title='Face Trainer',button='OK')

    cam.release()
    cv2.destroyAllWindows()
    trainerfunc()

photo_button = ttk.Button(
    root,text='Train Data',
    command=lambda:recorfunc()
)
photo_button.pack(
    ipadx=1.5,
    ipady=1.5,
    expand=True
)

encrypt_button = ttk.Button(
    root,text='Encrypt',
    command=lambda:encrypterprimaryfunc()
)
encrypt_button.pack(
    ipadx=1.5,
    ipady=1.5,
    expand=True
)

encrypt_button = ttk.Button(
    root,text='Decrypt',
    command=lambda:recognizer()
)
encrypt_button.pack(
    ipadx=1.5,
    ipady=1.5,
    expand=True
)

exit_button=ttk.Button(
root,
text='Exit',
command=lambda:root.quit()
)

exit_button.pack(
ipadx=1.5,
ipady=1.5,
expand=True
)

root.mainloop()