from tensorflow.keras.models import load_model
classifier = load_model('modelslr.h5')
import numpy as np
import keras.utils
test_image = keras.utils.load_img("C:/Users/ABDUL MUEED SOUDAGAR/Downloads/sign language dataset/dataset/test_set/S/7.png", target_size=(64, 64))
test_image = keras.utils.img_to_array(test_image)
test_image1 = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image1)
y=np.argmax(result)
d1={'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'J': 9,'K': 10,'L': 11,'M': 12,'N': 13,'O': 14,'P': 15,
                 'Q': 16,'R': 17,'S': 18,'T': 19,'U': 20,'V': 21,'W': 22,'X': 23,'Y': 24,'Z': 25}
y1=list(filter(lambda x: d1[x] == y,d1))[0]
print("The character predicted is" " " +y1)
