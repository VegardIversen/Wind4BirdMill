from PIL import Image # Bildebehandling
import numpy as np #nyttig støtte bibliotek
import os #filbehandling
import cv2 #bildelesing
from tensorflow.keras import datasets, layers, models #for modellen 
from keras.models import Sequential #modellen
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout #modellen
import random 
from sklearn.model_selection import train_test_split #splitting av dataset


i = 0 #Variabel for label av fugl
def prepros():
    #setter variabler 
    global i 
    data = [] 
    labels = []
    #itererer gjennom mapper og filer og legger til i lister.
    for folder in os.listdir('./bilder/'): 
        print(folder+' har label {}'.format(i)) 
        for bird in os.listdir(f'bilder/{folder}/'):
            try: 
            
                image = cv2.imread(f'bilder/{folder}/{bird}')
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((50, 50))
                data.append(np.array(size_image))
                labels.append(i)
            #hvis feil med fil
            except AttributeError:
                print(f'feil: {bird}')#printer hvilken fil som er feil
        i += 1
    #python liste->numpy liste
    df = np.array(data)
    labels = np.array(labels)
    #Slaar sammen lister
    dataset = list(zip(df,labels))
    #stokker datasettet
    random.shuffle(dataset)
    
    return dataset


df, labels = zip(*prepros()) #unzipper datasett
#endrer til numpy liste
df = np.array(df) 
labels = np.array(labels)
#splitter dataset for trening og testing
X_train, X_test, y_train, y_test = train_test_split(df,labels,test_size=0.1)
s=np.arange(X_train.shape[0])
#stokker om igjen
np.random.shuffle(s)
#normaliserer 
X_train=X_train[s]
y_train=y_train[s]
X_train = X_train/255.0


def create_model():
    global i
    model = models.Sequential()
    model.add(layers.Conv2D(16,3, padding='same', activation='relu', input_shape=(50,50,3)))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(32,3,padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(64,3,padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(i+1,activation='softmax'))#2 represent output layer neurons 
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model


#lager nettverket
model = create_model()
#printer nettverk strukturen
model.summary()
#trener modellen, her med 13 epochs
model.fit(X_train,y_train, epochs=13, validation_data=(X_test,y_test))
#lagrer modell i samme mappe som python filen. 
model.save('model.hdf5')