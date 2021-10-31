from flask import Flask,render_template, request, send_file
from flask import request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import cv2
from skimage.color import rgb2lab, lab2rgb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Size of the image required to train our model
img_size = 256

def get_generator_model():
    
    inputs = tf.keras.layers.Input( shape=( img_size , img_size , 1 ) )

    conv1 = tf.keras.layers.Conv2D( 64, kernel_size=(4, 4), strides=(2, 2), padding='same')( inputs )

    conv1 = tf.keras.layers.LeakyReLU(0.2)( conv1 )
    conv1 = tf.keras.layers.Conv2D( 128 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same')( conv1 )
    conv1 = tf.keras.layers.BatchNormalization()( conv1 )

    conv2 = tf.keras.layers.LeakyReLU(0.2)( conv1 )
    conv2 = tf.keras.layers.Conv2D( 256 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same')( conv2 )
    conv2 = tf.keras.layers.BatchNormalization()( conv2 )

    conv3 = tf.keras.layers.LeakyReLU(0.2)( conv2 )
    conv3 = tf.keras.layers.Conv2D( 512 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same')( conv3 )
    conv3 = tf.keras.layers.BatchNormalization()( conv3 )

    conv3 = tf.keras.layers.LeakyReLU(0.2)( conv3 )
    conv3 = tf.keras.layers.Conv2D( 512 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same')( conv3 )
    conv3 = tf.keras.layers.BatchNormalization()( conv3 )

    conv3 = tf.keras.layers.LeakyReLU(0.2)( conv3 )
    conv3 = tf.keras.layers.Conv2D( 512 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same')( conv3 )
    conv3 = tf.keras.layers.BatchNormalization()( conv3 )

    conv4 = tf.keras.layers.LeakyReLU(0.2)( conv3 )
    conv4 = tf.keras.layers.Conv2D( 512 , kernel_size=( 4 , 4 ), strides=(2, 2), padding='same')( conv4 )
    conv4 = tf.keras.layers.ReLU()( conv4 )
    conv4 = tf.keras.layers.Conv2DTranspose( 512 , kernel_size=( 4 , 4 ), strides=(2, 2), padding='same' )( conv4 )
    conv4 = tf.keras.layers.BatchNormalization()( conv4 )

    conv5 = tf.keras.layers.ReLU()( conv4 )
    conv5 = tf.keras.layers.Conv2DTranspose( 512 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same' )( conv5 )
    conv5 = tf.keras.layers.BatchNormalization()( conv5 )

    conv5 = tf.keras.layers.ReLU()( conv5 )
    conv5 = tf.keras.layers.Conv2DTranspose( 512 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same' )( conv5 )
    conv5 = tf.keras.layers.BatchNormalization()( conv5 )



    conv5 = tf.keras.layers.ReLU()( conv5 )
    conv5 = tf.keras.layers.Conv2DTranspose( 256 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same' )( conv5 )
    conv5 = tf.keras.layers.BatchNormalization()( conv5 )


    conv6 = tf.keras.layers.ReLU()( conv5 )
    conv6 = tf.keras.layers.Conv2DTranspose( 128 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same' )( conv6 )
    conv6 = tf.keras.layers.BatchNormalization()( conv6 )

    conv7 = tf.keras.layers.ReLU()( conv6 )
    conv7 = tf.keras.layers.Conv2DTranspose( 64 , kernel_size=( 4 , 4 ) , strides=(2, 2), padding='same' )( conv7 )
    conv7 = tf.keras.layers.BatchNormalization()( conv7 )

    conv8 = tf.keras.layers.ReLU()( conv7 )
    conv8 = tf.keras.layers.Conv2DTranspose( 2 , kernel_size=( 4 , 4 ) ,activation='tanh', strides=(2, 2), padding='same' )( conv8 )
    model = tf.keras.models.Model( inputs , conv8 )
    return model


generator_optimizer = tf.keras.optimizers.Adam( 0.0005 )
generator = get_generator_model()
generator.load_weights('./model/modeGEN256.h5')
print("Loaded model from disk")

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = tf.concat([L, ab], axis=-1)
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def predict(link):
    im = cv2.imread(str(link))
    im = cv2.resize(im, (img_size, img_size))  
    # rgb_img_array = (np.asarray( rgb_image ) ) / 255
    im = np.array(im)
    img_lab = rgb2lab(im).astype("float32")
    L = img_lab[...,[0] ]/ 50. - 1.
    ab = img_lab[...,[1, 2] ]/ 110. 
    ab_pre = generator.predict(L.reshape((1,img_size,img_size,1)))
    im_pre = lab_to_rgb(L.reshape((1,img_size,img_size,1)), ab_pre)
    # im_pre = im_pre.astype('uint64')
    im_pre = im_pre[0]*255.0
    im_pre = cv2.cvtColor(im_pre, cv2.COLOR_RGB2BGR )
    return im_pre


import random, string

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

# Khai báo port của server
my_port = '5000'

app = Flask(__name__)
CORS(app)
PEOPLE_FOLDER = os.path.join('static', 'people_photo')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
# Khai bao ham xu ly request index
@app.route('/')
@cross_origin()
def index():
    return render_template('client.html')

@app.route('/upload', methods=['POST'])
def my_form_post():
    # print('hello')
    # text=request.form['u']
    if request.method == "POST":
        file = request.files["file"]
        file.save(os.path.join("static", file.filename))
        result = predict(os.path.join("static", file.filename))
        st = randomword(5)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'anh'+st+'.jpg'), result)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'anh'+st+'.jpg')
        
        return send_file(full_filename, mimetype='image/gif')
        # return render_template('client.html',user_image = full_filename)
    return render_template("client.html")
# Thuc thi server
if __name__ == '__main__':
    app.run(debug=True, host='localhost',port=my_port)