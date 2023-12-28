from aiogram import types, Bot, executor, Dispatcher
import requests
import io
from PIL import Image, ImageFilter
import os
import secrets
import pickle
import numpy as np
import sklearn
from io import BytesIO
import torchvision.transforms as transforms
import torch
import tensorflow as tf
from keras.models import load_model


TOKEN_API = ""
bot = Bot(TOKEN_API)
dp = Dispatcher(bot)

URI_INFO = f"https://api.telegram.org/bot{TOKEN_API}/getFile?file_id="
URI = f"https://api.telegram.org/file/bot{TOKEN_API}/"

model = load_model('my_model.h5')
img_width = 227
img_height = 227
class_names = ['съедобный гриб', 'ядовитый гриб']

def fit_range(x):
    k = (1920 - 144) / (5 - 1)
    d = 144 - 5 * k
    return x * k + d

@dp.message_handler(content_types=['photo'])
async def process_photo(msg: types.Message) -> None:
    file_id = ""
    try:
        file_id = msg.photo[3].file_id
    except:
        file_id = msg.photo[2].file_id
    try:
        uri = URI_INFO + file_id
        resp = requests.get(uri)
        img_path = resp.json()['result']['file_path']
        img = requests.get(URI + img_path)
        img = Image.open(io.BytesIO(img.content))
        path = f'static/{secrets.token_hex(8)}.png'
        if not os.path.exists('static'):
            os.mkdir('static')
        img.save(path, format='PNG')
        img = tf.keras.utils.load_img(path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        await msg.answer("На изображении скорее всего {} ({:.2f}% вероятность)".format(class_names[np.argmax(score)], 100 * np.max(score)))
    except:
        await msg.answer('С вашей фотографией что-то не так, попробуйте использовать другую')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def to_prepaate_img(img):
    # load image in RGB mode (png files contains additional alpha channel)
    img = img.convert('RGB')

    # set up transformation to resize the image
    resize = transforms.Resize([70, 70])
    img = resize(img)
    to_tensor = transforms.ToTensor()

    # apply transformation and convert to Pytorch tensor
    img_tenzor = to_tensor(img)
    # torch.Size([3, 70, 70])
    img_norm = normalize(img_tenzor)
    # to_normalize = transforms.Normalize()

    # add another dimension at the front to get NCHW shape
    tensor = img_tenzor.unsqueeze(0)
    # torch.Size([1, 3, 70, 70])
    return tensor


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)