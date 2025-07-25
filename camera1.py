import tensorflow as tf
import cv2
import imutils
import numpy as np
import keras.utils
import pyttsx3
import tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
# global variables
r = Tk()
#r.geometry("700x500")
sent=[]
labels = []
bg = None    
def openfile():
    global labels
    d1={'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'J': 9,'K': 10,'L': 11,'M': 12,'N': 13,'O': 14,'P': 15,
     'Q': 16,'R': 17,'S': 18,'T': 19,'U': 20,'V': 21,'W': 22,'X': 23,'Y': 24,'Z': 25}
    f1=filedialog.askopenfilename()
    image=cv2.imread(f1)
    res=tf.image.resize(image,(64,64))
    new_model=load_model('modelslr.h5')
    pred=new_model.predict(np.expand_dims(res/255,0))
    y=np.argmax(pred)
    y1=list(filter(lambda x: d1[x] == y,d1))[0]
    #print(y1)
    global sent 
    sent.append(y1)
    #label=Label(r,text="")
    #label.config(text=y1)
    #label = Label(r,text=y1)
    label=Label(r,text="The character predicted is:"+y1,font=('Helvetica',14,'bold'))
    labels.append(label)
    label.pack()
    #label.place(x=950,y=550)
    engine = pyttsx3.init()
    engine.say("The character predicted is:"+y1)
    engine.runAndWait()
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def dtect_hand():
    try:
        # initialize weight for running average
        from keras.preprocessing import image
        import cv2
        import imutils
        aWeight = 0.5

        # get the reference to the webcam
        camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        # region of interest (ROI) coordinates
        top, right, bottom, left = 10, 350, 225, 590

        # initialize num of frames
        num_frames = 0

        # keep looping, until interrupted
        while(True):
            # get the current frame
            (grabbed, frame) = camera.read()

            # resize the frame
            frame = imutils.resize(frame, width=700)

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our running average model gets calibrated
            if num_frames < 30:
                run_avg(gray, aWeight)
            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    cv2.imshow("Thesholded", thresholded)

            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

            # increment the number of frames
            num_frames += 1

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF
            if keypress%256 == 32:
                # SPACE pressed
                img_name = "Clicked_image.png"
                cv2.imwrite(img_name, thresholded)
                new_model=load_model('modelslr.h5')
                test_image = keras.utils.load_img('./Clicked_image.png')
                test_image1 = keras.utils.img_to_array(test_image)
                res=tf.image.resize(test_image1,(64,64))
                pred=new_model.predict(np.expand_dims(res/255,0))
                y=np.argmax(pred)
                d1={'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'J': 9,'K': 10,'L': 11,'M': 12,'N': 13,'O': 14,'P': 15,
                     'Q': 16,'R': 17,'S': 18,'T': 19,'U': 20,'V': 21,'W': 22,'X': 23,'Y': 24,'Z': 25}
                y1=list(filter(lambda x: d1[x] == y,d1))[0]
                global sent
                global labels
                sent.append(y1)
                #print("The character predicted is" " " +y1)
                out=Label(r,text="The character predicted is:" +y1,font=('Helvetica',14,'bold'))
                labels.append(out)
                out.pack(side=TOP)
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(frame,"The Character predicted is: " +y1,(200, 50),font, 1,(255,0,0),2,cv2.LINE_4)
                #cv2.imshow('ssa',frame)
                engine = pyttsx3.init()
                engine.say("The character predicted is:"+y1)
                engine.runAndWait()
                camera.release()
                cv2.destroyAllWindows()
                break
            if keypress == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                break
    except:
        messagebox.showwarning("WARNING", "PLEASE SHOW YOUR HAND INSIDE THE BOUNDING BOX!")
        camera.release()
        cv2.destroyAllWindows()
        
def delete_label(label):
    global labels
    global sent
    labels.remove(label)
    del sent[-1]
    #sent.remove(y1)
    label.destroy()

def output():
    new_window=Toplevel()
    new_window.geometry('400x300')
    global labels
    global sent
    #lf=Label(new_window,text="SIGN LANGUAGE RECOGNITION SYSTEM",font=('Helvetica',24,'bold'))
    #lf.pack()
    str1 = ""
    for ele in sent:
        str1 += ele
    show1=Label(new_window,text="The Word is: " +str1,font=('Helvetica',17,'bold'))
    show1.pack()
    engine1 = pyttsx3.init()
    engine1.say("The Word is:"+str1)
    engine1.runAndWait()
    frame1 = Frame(new_window,width=300,height=400)
    frame1.pack()
    frame1.place()

    img1 = ImageTk.PhotoImage(Image.open("C:/Users/ABDUL MUEED SOUDAGAR/Downloads/dataset-cover.png"),master=new_window)

    labelf = Label(frame1, image = img1)
    labelf.pack()
    new_window.mainloop()
    

l1=Label(r,text="SIGN LANGUAGE RECOGNITION SYSTEM",font=('Helvetica',24,'bold'))
l1.pack()

bs=Button(r,text='OPEN CAMERA',width=27,bg='yellow',command=dtect_hand,font=('Helvetica',10,'bold'))
bs.pack(side=TOP)
bs1=Button(r, text="OPEN FILE",width=27,bg='yellow',command=openfile,font=('Helvetica',10,'bold'))
bs1.pack(side=TOP)
#bs2=Button(r,text='CLOSE',width=27,bg='yellow',command=r.destroy,font=('Helvetica',10,'bold'))
#bs2.pack(side=TOP)
bs3=Button(r,text='SHOW',width=27,bg='yellow',command=output,font=('Helvetica',10,'bold'))
bs3.pack(side=TOP)
bs4=Button(r,text='DELETE',width=27,bg='yellow',command=lambda: delete_label(labels[-1]),font=('Helvetica',10,'bold'))
bs4.pack(side=TOP)
frame = Frame(r,width=300,height=400)
frame.pack()
frame.place()

img = ImageTk.PhotoImage(Image.open("C:/Users/ABDUL MUEED SOUDAGAR/Downloads/dataset-cover.png"),master=r)

label = Label(frame, image = img)
label.pack()

r.mainloop()
