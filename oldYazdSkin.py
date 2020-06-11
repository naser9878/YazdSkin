import streamlit as st
import numpy as np
import pandas as pd
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras import backend as K
import os
import time
import io
from PIL import Image
import plotly.express as px


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,Activation
from keras.layers.normalization import BatchNormalization


MODELSPATH = './models/'
DATAPATH = './data/'


def render_header():
    st.write("""
        <p align="center"> 
            <H1> Skin cancer Analyzer 
			<H2> Negin Shakeri 

        </p>

    """, unsafe_allow_html=True)


def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


@st.cache
def load_mekd():
    img = Image.open(DATAPATH + '/ISIC_0024312.jpg')
    return img



@st.cache
def data_gen(x):
    img = np.asarray(Image.open(x).resize((100, 75)))
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)

    return x_validate


@st.cache
def data_gen_(img):
    img = img.reshape(100, 75)
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)

    return x_validate


def load_models():

    model = load_model(MODELSPATH + 'model.h5')
    model.summary()
    return model


@st.cache
def predict(x_test, model):
    Y_pred = model.predict(x_test)
    ynew = model.predict_proba(x_test)
    ynew2 = model.predict_classes(x_test)

    K.clear_session()
    ynew = np.round(ynew, 2)
    ynew = ynew*100
    y_new = ynew[0].tolist()


    Y_pred_classes = np.argmax(Y_pred, axis=1)
    K.clear_session()
    return ynew2,y_new, Y_pred_classes


@st.cache
def display_prediction(y_new):
    """Display image and preditions from model"""

    result = pd.DataFrame({'Probability': y_new}, index=np.arange(7))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {2: 'Benign keratosis-like lesions', 4: 'Melanocytic nevi', 3: 'Dermatofibroma',
                        5: 'Melanoma', 6: 'Vascular lesions', 1: 'Basal cell carcinoma', 0: 'Actinic keratoses'}
    result["Classes"] = result["Classes"].map(lesion_type_dict)
    return result


def main():
    from urllib.parse import quote
    from urllib.request import urlopen
    import streamlit as st


    img = Image.open(DATAPATH + '/logo.jpg')
    st.image(img, caption='', width=None)
    st.title("Ms. Negin Shakeri Thesis")
    st.sidebar.header('Ms. Negin Shakeri Thesis')
    st.sidebar.title('Skin cancer Analyzers')
    ## st.sidebar.image(img)

    st.sidebar.subheader('Choose a page to proceed:')
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
		h1{color: red;}
		body {text-align: center !important}
        </style>
        """, unsafe_allow_html=True)
		
		

    page = st.sidebar.selectbox("", ["Sample Data", "Upload Your Image"])

    if page == "Sample Data":
        st.header("Sample Data Prediction for Skin Cancer")
        st.markdown("""
        **Now, this is probably why you came here. Let's get you some Predictions**

        You need to choose Sample Data
        """)


        mov_base = ['Sample Data I']
        movies_chosen = st.multiselect('Choose Sample Data', mov_base)

        if len(movies_chosen) > 1:
            st.error('Please select Sample Data')
        if len(movies_chosen) == 1:
            st.success("You have selected Sample Data")
        else:
            st.info('Please select Sample Data')

        if len(movies_chosen) == 1:
            if st.checkbox('Show Sample Data'):
                st.info("Showing Sample data---->>>")
                image = load_mekd()
                st.image(image, caption='Sample Data', use_column_width=True)
                st.subheader(" Training Algorithm!")

                if st.checkbox('Convolutional Neural Network'):
                    st.subheader(" Set some hyperparameters")
                    batch_size = st.selectbox('Select batch size', [32, 64, 128, 256])
                    epochs=st.selectbox('Select number of epochs', [3, 10, 25, 50])
                    loss_function = st.selectbox('Loss function', ['mean_squared_error', 'mean_absolute_error', 'categorical_crossentropy'])
                    optimizer = st.selectbox('Optimizer', ['SGD', 'RMSprop', 'Adam'])

                    st.subheader('Building your CNN')
                    model = Sequential()

                    act1 = st.selectbox('Activation function for first layer: ', ['relu', 'tanh', 'softmax'])

                    model.add(Conv2D(filters = 16,activation=act1,kernel_size = 2,input_shape=(75,100,3),padding='same'))

                    model.add(MaxPool2D(pool_size=2))

                    model.add(Conv2D(filters = 32,kernel_size = 2,activation= act1,padding='same'))
                    model.add(MaxPool2D(pool_size=2))

                    model.add(Conv2D(filters = 64,kernel_size = 2,activation=act1,padding='same'))
                    model.add(MaxPool2D(pool_size=2))

                    model.add(Conv2D(filters = 128,kernel_size = 2,activation=act1,padding='same'))
                    model.add(MaxPool2D(pool_size=2))

                    drop2=st.selectbox('Which drop rate?', [0.1, 0.25,0.3, 0.5],key=1)
                    model.add(Dropout(drop2))

                        

                    model.add(Flatten())
                    model.add(Dense(150))

                    if st.checkbox('Add drop layer?'):
                        drop1=st.selectbox('Which drop rate?', [0.1, 0.25,0.3, 0.5],key=0)
                        model.add(Dropout(drop1))


                    act3 = st.selectbox('Activation function for Dense layer: ', ['relu', 'tanh', 'softmax'])
                    model.add(Dense(7,activation = act3))
                    model.summary()


                    if st.checkbox('Submit CNN'):
                        model = load_models()
                        st.success("Hooray !! Keras Model Loaded!")
                        model_summary_string = get_model_summary(model)
                        st.info(model_summary_string)
                        
                        if st.checkbox('Show Prediction Probablity on Sample Data'):
                            x_test = data_gen(DATAPATH + '/ISIC_0024312.jpg')
                            y_new2,y_new, Y_pred_classes = predict(x_test, model)
                            result = display_prediction(y_new)
                            st.write(result)

                            if st.checkbox('Display Probability Graph'):
                                fig = px.bar(result, x="Classes",
                                            y="Probability", color='Classes')
                                st.plotly_chart(fig, use_container_width=True)

    if page == "Upload Your Image":

        st.header("Upload Your Image")

        file_path = st.file_uploader('Upload an image', type=['png', 'jpg'])

        if file_path is not None:
            x_test = data_gen(file_path)
            image = Image.open(file_path)
            img_array = np.array(image)

            st.success('File Upload Success!!')
        else:
            st.info('Please upload Image file')

        if st.checkbox('Show Uploaded Image'):
            st.info("Showing Uploaded Image ---->>>")
            st.image(img_array, caption='Uploaded Image',
                     use_column_width=True)
            if st.checkbox('Convolutional Neural Network'):
                    st.subheader(" Set some hyperparameters")
                    batch_size = st.selectbox('Select batch size', [32, 64, 128, 256])
                    epochs=st.selectbox('Select number of epochs', [3, 10, 25, 50])
                    loss_function = st.selectbox('Loss function', ['mean_squared_error', 'mean_absolute_error', 'categorical_crossentropy'])
                    optimizer = st.selectbox('Optimizer', ['SGD', 'RMSprop', 'Adam'])

                    st.subheader('Building your CNN')
                    model = Sequential()

                    act1 = st.selectbox('Activation function for first layer: ', ['relu', 'tanh', 'softmax'])

                    model.add(Conv2D(filters = 16,activation=act1,kernel_size = 2,input_shape=(75,100,3),padding='same'))

                    model.add(MaxPool2D(pool_size=2))

                    model.add(Conv2D(filters = 32,kernel_size = 2,activation= act1,padding='same'))
                    model.add(MaxPool2D(pool_size=2))

                    model.add(Conv2D(filters = 64,kernel_size = 2,activation=act1,padding='same'))
                    model.add(MaxPool2D(pool_size=2))

                    model.add(Conv2D(filters = 128,kernel_size = 2,activation=act1,padding='same'))
                    model.add(MaxPool2D(pool_size=2))

                    drop2=st.selectbox('Which drop rate?', [0.1, 0.25,0.3, 0.5],key=1)
                    model.add(Dropout(drop2))

                        

                    model.add(Flatten())
                    model.add(Dense(150))

                    if st.checkbox('Add drop layer?'):
                        drop1=st.selectbox('Which drop rate?', [0.1, 0.25,0.3, 0.5],key=0)
                        model.add(Dropout(drop1))


                    act3 = st.selectbox('Activation function for Dense layer: ', ['relu', 'tanh', 'softmax'])
                    model.add(Dense(7,activation = act3))
                    model.summary()


                    if st.checkbox('Submit CNN'):
                        model = load_models()
                        st.success("Hooray !! Keras Model Loaded!")
                        model_summary_string = get_model_summary(model)
                        st.info(model_summary_string)
                        
                        if st.checkbox('Show Prediction Probablity on Sample Data'):
                            x_test = data_gen(DATAPATH + '/ISIC_0024312.jpg')
                            y_new2,y_new, Y_pred_classes = predict(x_test, model)
                            result = display_prediction(y_new)
                            st.write(result)

                            if st.checkbox('Display Probability Graph'):
                                fig = px.bar(result, x="Classes",
                                            y="Probability", color='Classes')
                                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()


