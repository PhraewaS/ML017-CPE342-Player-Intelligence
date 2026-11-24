import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import sobel, gaussian_filter1d

def upright(img):
    thumb=img.convert("L").resize((115,60),Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(.7))
    pixels=np.asarray(thumb,dtype=np.float32)
    gx=sobel(pixels,axis=1)
    gy=sobel(pixels,axis=0)
    angle=(np.degrees(np.arctan2(gy,gx))+45)%90
    energy=(gx*gx+gy*gy)**.8
    energy[[0,-1],:]=0
    energy[:,[0,-1]]=0
    hist=np.bincount(angle.astype(int).clip(0,89).ravel(),weights=energy.ravel(),minlength=90)
    smooth=gaussian_filter1d(hist,2,mode="wrap")
    tilt=(np.argmax(smooth[12:79])+12)-44.5
    # Preserve scene aspect ratio when estimating rotation, then restore contrast.
    denoised=img.filter(ImageFilter.SMOOTH)
    turned=denoised.rotate(float(tilt),resample=Image.Resampling.BILINEAR)
    w,h=turned.size
    crop=turned.crop((int(.05*w),int(.05*h),int(.95*w),int(.95*h)))
    gray=ImageOps.autocontrast(crop.convert("L"),cutoff=1).convert("RGB")
    color=ImageOps.autocontrast(crop,cutoff=1)
    return Image.blend(gray,color,.30)
