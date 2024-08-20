# STEP 1: Generate Two Large Prime Numbers (p,q) randomly
from random import randrange, getrandbits
from tkinter import *
from tkinter import ttk  
from tkinter import Menu  
from tkinter import messagebox as mbox  
# import filedialog module
from tkinter import filedialog
flg=0;
import tkinter as tk
import tkinter
from tkinter import *
from PIL import Image, ImageTk
# Create a photoimage object of the image in the path


import tkinter as tk
from PIL import ImageTk, Image
from tkintertable import TableCanvas, TableModel

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image

import seaborn as sns
def train():
    print("training")
    
    
    from tensorflow.keras.preprocessing import image
    
    model = Sequential()   ## creating a blank model

    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))    ### reduce the overfitting

    model.add(Flatten())    ### input layer

    model.add(Dense(256,activation='relu'))    ## hidden layer of ann

    model.add(Dropout(0.5))

    model.add(Dense(512,activation='relu'))    ## hidden layer of ann

    model.add(Dropout(0.5))

    model.add(Dense(2,activation='softmax'))   ## output layer


    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.summary()

    #Moulding train images
    train_datagen = image.ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)

    test_dataset = image.ImageDataGenerator(rescale=1./255)

    #Reshaping test and validation images 
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size = (224,224),
        batch_size = 15,
        class_mode = 'categorical')
    validation_generator = test_dataset.flow_from_directory(
        'dataset/val',
        target_size = (224,224),
        batch_size = 7,
        class_mode = 'categorical')
    print("Training Started")
    #### Train the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=7,
        epochs = 10,
        validation_data = validation_generator
    )

    print("Training Ended")
    test_loss, test_acc = model.evaluate(validation_generator)
    print('Accuracy:', test_acc)
    model.save("trained_model.h5");

    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer : Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    out = "Accuracy Score : "
    out += str(test_acc)
    
    app = tk.Tk()
    app.title("Scores")
    ttk.Label(app, text=out).grid(column=0,row=0,padx=20,pady=30)  
    menuBar = Menu(app)
    app.config(menu=menuBar)


    

def test():
    print("testing")
    import tkinter as tk
    import tkinter
    from PIL import Image, ImageTk
    from tensorflow.keras.preprocessing import image
    def upload_file():
        global img
        global filename
        f_types = [('ALL', '*')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        image=Image.open(filename)

        # Resize the image in the given (width, height)
        imgs=image.resize((234, 234))
        img = ImageTk.PhotoImage(imgs)
        b2 =tk.Button(my_w,image=img) # using Button 
        b2.grid(row=9,column=1, padx=5, pady=5)
        print(filename)
        
    def predict():
        
            
        ft=0
        st=0
        lt=0
        rt=0
        ut=0

        h=""
        out=""
        outv=5
        model1 = load_model("trained_model.h5")
        img = image.load_img(filename,target_size=(224,224))
        img = image.img_to_array(img, dtype='uint8')
        
        
        img = np.expand_dims(img,axis=0)   ### flattening
        ypred1 = model1.predict_classes(img)
        ypred1=ypred1.round()
        print(ypred1)
        if(ypred1[0]==0):
            out = "Result for the given Image: Drowning"
            outv=0
        elif(ypred1[0]==1):
            out = "Result for the given Image: Non Drowning"
            outv=1  
        ft=0
        st=0
        lt=0
        rt=0
        ut=0
        
        print(out)
             
        from tkinter import messagebox  
                 
        my_w.geometry("100x100")      
          
        messagebox.showinfo("Result",out)  
          
        print(" ")
        
    
        
    my_w = tk.Toplevel()
    my_w.geometry("400x400")  # Size of the window 
    my_w.title('Drowning Detection System')
    my_font1=('times', 18, 'bold')


    l1 = tk.Label(my_w,text='Give Images',width=30,font=my_font1)  
    l1.grid(row=1,column=1)
    b1 = tk.Button(my_w, text='Upload File', 
       width=20,command = lambda:upload_file())
    b1.grid(row=2,column=1, padx=5, pady=5) 

    b3 = tk.Button(my_w, text='Predict Output', 
       width=20,command = lambda:predict())
    b3.grid(row=6,column=1, padx=5, pady=5)
    my_w.mainloop()


    

    my_w.mainloop()  # Keep the window open





if __name__ == '__main__':    
    # Create the main window
    root = tk.Tk()
    root.title("Drowning Detection System")
    root.geometry("600x400")

    # Load the background image
    bg_image = ImageTk.PhotoImage(Image.open("dr.jpg"))
    
    # Load the image to be displayed
    image = Image.open("dr.png")
    resized_image = image.resize((120, 120))
    tk_image = ImageTk.PhotoImage(resized_image)

    # Create a canvas to display the background image
    canvas = tk.Canvas(root, width=500, height=300)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_image, anchor="nw")

    # Create a title in the center of the GUI with a bigger font
    title = tk.Label(canvas, text="Drowning Detection System", font=("Arial", 24))
    title.place(relx=0.5, rely=0.1, anchor="center")

    # Display the image in the center of the GUI
    image_label = tk.Label(canvas, image=tk_image)
    image_label.place(relx=0.5, rely=0.4, anchor="center")

    # Create two big buttons for Train and Test
    train_button = tk.Button(canvas, text="Train", bg="blue", fg="white", font=("Arial", 20), command = train)
    train_button.place(relx=0.3, rely=0.7, anchor="center")
    test_button = tk.Button(canvas, text="Test", bg="blue", fg="white", font=("Arial", 20), command= test)
    test_button.place(relx=0.7, rely=0.7, anchor="center")

    # Start the GUI
    root.mainloop()

