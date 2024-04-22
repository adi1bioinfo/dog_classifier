from fastai.vision.all import *
import gradio as gr

learn = load_learner('dog_model.pkl')

categories = ('Beagle', 'Chihuahua', 'Golden Retriever', 'Husky', 'Maltipoo', 'Poodle', 'Shis Tzu')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return f'Prediction: {pred}; Probability: {probs[idx]:.04f}'


image = gr.Image(height=192, width=192)
label = gr.Label()
examples = ['dog.jpeg', 'Cute_dog.jpg', 'dog2.jpeg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False, share=True)


