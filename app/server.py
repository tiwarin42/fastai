import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import pickle
from PIL import Image
import tensorflow as tf
import numpy as np

# export_file_url = 'https://www.dropbox.com/s/3y5xorm7rq8fzby/model_Lgt.pkl?raw=1'
# export_file_name = 'model_Lgt.pkl'

classNames = {4: 'class4',
              5: 'class5',
              6: 'class6',
              7: 'class7',
              8: 'class8',
              9: 'class9',
              10: 'class10',
              11: 'class11',
              12: 'class12',
              13: 'class13',
              14: 'class14',
              15: 'class15'}

# classes = ['class4','class5','class6','class7','class8','class9','class10','class11','class12','class13','class14','class15']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


# async def download_file(url, dest):
#     if dest.exists(): return
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             data = await response.read()
#             with open(dest, 'wb') as f:
#                 f.write(data)


async def setup_learner():
    with open('app/models/test.pkl', 'rb') as file:
         model = pickle.load(file)
    try:
        models = model
        return models
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    
    img = Image.open(BytesIO(img_bytes)).resize((64, 64)).convert('RGB')
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = open_cv_image

    img_array = tf.keras.preprocessing.image.img_to_array(img).reshape(1, -1)
    prediction = model.predict(img_array)
    category = classNames[prediction[0]]
    return JSONResponse({'result': category})

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
