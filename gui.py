from tkinter import *
from PIL import ImageTk, Image
import pyttsx3
from tkinter import filedialog
r = Tk()
#r.geometry("700x500")
r.title('Sign Language Recognition System')
#label=None
#widget=None
def clear_widget(widget):
    widget.destroy()
def openfile():
    global label
    d1={'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'J': 9,'K': 10,'L': 11,'M': 12,'N': 13,'O': 14,'P': 15,
     'Q': 16,'R': 17,'S': 18,'T': 19,'U': 20,'V': 21,'W': 22,'X': 23,'Y': 24,'Z': 25}
    f1=filedialog.askopenfilename()
    image=cv2.imread(f1)
    res=tf.image.resize(image,(200,200))
    pred=new_model.predict(np.expand_dims(res/255,0))
    y=np.argmax(pred)
    y1=list(filter(lambda x: d1[x] == y,d1))[0]
    #print(y1)
    label=Label(r,text="The character predicted is:"+y1,font=('Helvetica',14,'bold'))
    label.pack(side=TOP)
    engine = pyttsx3.init()
    engine.say("The character predicted is:"+y1)
    engine.runAndWait()

    
l1=Label(r,text="SIGN LANGUAGE RECOGNITION SYSTEM",font=('Helvetica',24,'bold'))
l1.pack()

bs=Button(r,text='OPEN FILE',width=27,bg='yellow',command=openfile,font=('Helvetica',10,'bold'))
bs.pack(side=TOP)
bs1=Button(r,text='CLOSE',width=27,bg='yellow',command=r.destroy,font=('Helvetica',10,'bold'))
bs1.pack(side=TOP)
bs3=Button(r, text="Clear",command=lambda : clear_widget(label))
bs3.pack(side=TOP)
    
frame = Frame(r,width=300,height=400)
frame.pack()
frame.place()

img = ImageTk.PhotoImage(Image.open("C:/Users/ABDUL MUEED SOUDAGAR/Downloads/dataset-cover.png"),master=r)

labelf = Label(frame, image = img)
labelf.pack()

r.mainloop()
