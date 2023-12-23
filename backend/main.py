from aiogram import types, Bot, executor, Dispatcher
import requests
import io
from PIL import Image, ImageFilter
import os
import secrets
import pickle
import numpy as np
import sklearn

TOKEN_API = "6841493307:AAFEZptlHBiv7uMLqOIoejZINWTDyEVSAk8"
bot = Bot(TOKEN_API)
dp = Dispatcher(bot)

URI_INFO = f"https://api.telegram.org/bot{TOKEN_API}/getFile?file_id="
URI = f"https://api.telegram.org/file/bot{TOKEN_API}/"

model = pickle.load(open("model.pkl", "rb"))

def fit_range(x):
    k = (1920 - 144) / (5 - 1)
    d = 144 - 5 * k
    return x * k + d

@dp.message_handler(content_types=['photo'])
async def process_photo(msg: types.Message) -> None:
    try:
        file_id = msg.photo[3].file_id
        uri = URI_INFO + file_id
        resp = requests.get(uri)
        img_path = resp.json()['result']['file_path']
        img = requests.get(URI + img_path)
        img = Image.open(io.BytesIO(img.content))
        img = img.filter(ImageFilter.GaussianBlur(radius=20))
        path = f'static/{secrets.token_hex(8)}.png'
        if not os.path.exists('static'):
            os.mkdir('static')
        img.save(path, format='PNG')
        img = Image.open(path)
        width, height = img.size
        print(f"Width: {width}, Height: {height}")
        float_features = [fit_range(width), fit_range(height), fit_range(width / 3), fit_range(height / 5)]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        await msg.answer(prediction)
    except:
        await msg.answer('С вашей фотографией что-то не так, попробуйте использовать другую')


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)