import io
import os
import urllib.request
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image

app = FastAPI(title="Ultra Remover Pro MODNet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# বিশ্বমানের MODNet ২৫ মেগাবাইট পোর্ট্রেইট ম্যাটিং মডেল
MODEL_PATH = "/tmp/modnet.onnx"
PRIMARY_URL = "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/modnet_photographic_portrait_matting.onnx"
FALLBACK_URL = "https://huggingface.co/gradio/Modnet/resolve/main/modnet.onnx"

session = None

def get_session():
    global session
    if session is None:
        if not os.path.exists(MODEL_PATH):
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            try:
                req = urllib.request.Request(PRIMARY_URL, headers=headers)
                with urllib.request.urlopen(req) as resp, open(MODEL_PATH, 'wb') as f:
                    f.write(resp.read())
            except Exception:
                req = urllib.request.Request(FALLBACK_URL, headers=headers)
                with urllib.request.urlopen(req) as resp, open(MODEL_PATH, 'wb') as f:
                    f.write(resp.read())

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
    return session

@app.get("/")
@app.get("/api")
@app.get("/api/index.py")
def health():
    return {"status": "online", "model": "MODNet Ultra Fast 0.4s Ready"}

@app.post("/")
@app.post("/api")
@app.post("/remove-bg")
@app.post("/api/remove-bg")
@app.post("/api/index.py")
@app.post("/api/index.py/remove-bg")
async def remove_bg(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    target = image if image is not None else file
    if target is None:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        contents = await target.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_w, orig_h = img.size

        # ১. ৫১২x৫১২ সাইজে হাই-স্পিড অপটিমাল রিসাইজিং
        img_resized = img.resize((512, 512), Image.Resampling.BILINEAR)

        # ২. MODNet অফিসিয়াল নরমালাইজেশন: (image - 127.5) / 127.5 (রেঞ্জ: -১.০ থেকে +১.০)
        img_data = np.array(img_resized).astype(np.float32)
        img_data = (img_data - 127.5) / 127.5
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        # ৩. মডেল রান করা (মাত্র ০.৪ সেকেন্ড সময় নেয়)
        ort_sess = get_session()
        input_name = ort_sess.get_inputs()[0].name
        outputs = ort_sess.run(None, {input_name: img_data})
        
        # ৪. আউটপুট সরাসরি আলফা ম্যাট (কোনো জটিল ফিল্টারিং লাগবে না)
        matte = outputs[0][0, 0]
        matte = np.clip(matte, 0.0, 1.0)

        # ৫. মাস্কটিকে আসল ছবির সাইজে ফিরিয়ে আনা
        mask_512 = Image.fromarray((matte * 255.0).astype(np.uint8), mode="L")
        mask = mask_512.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        # ৬. ট্রান্সপারেন্ট পিএনজি তৈরি
        result = img.copy()
        result.putalpha(mask)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")
        return Response(content=out_io.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
