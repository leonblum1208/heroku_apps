

from copy import Error
import fastbook

fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
from IPython.display import display

try:
    learn_inf = load_learner('bird_mmodel.pkl', cpu=True)
except:
    print(Error)
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()

def on_click(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
    
btn_upload.observe(on_click, names=['data'])

display(VBox(
    [widgets.Label('Select your bird!'), 
    btn_upload, out_pl, lbl_pred]
    ))


