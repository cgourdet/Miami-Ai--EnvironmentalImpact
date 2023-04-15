import base64
import json
import random
import requests
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import pipeline
from transformers import YolosFeatureExtractor, YolosForObjectDetection

from flask import Flask, render_template


# TODO: Limit to a few examples
mocked_chatgpt = {
    'gray snapper': '''
    In Florida, the legal size limit for gray snapper (also known as mangrove snapper) is a minimum total length of 10 inches measured from the most forward point of the head with the mouth closed to the farthest tip of the tail. The daily bag limit for gray snapper is five fish per person within the 10-inch minimum and 14-inch maximum size range. It is important to check for any updates or changes to these regulations, as they may vary depending on the time of year or location. Additionally, it is always important to follow catch-and-release guidelines for any fish that are not within the legal size range or are unwanted.
    ''',
}


def load_image_annos():
    annos = json.loads(Path('static/img/_annotations.coco.json').read_text())
    # Set a dict of categories
    categories = annos['categories']
    categories = {cate['id']: cate['name'] for cate in categories}
    # Copy over the list the images
    images = list(annos['images'])
    # For each image, add an empty list of annotations
    for img in images:
        img['annos'] = list()
    # Turn the list of images into a dict keyed on image id
    images = {img['id']: img for img in images}
    # Add the annotations for each image
    for anno in annos['annotations']:
        img = images[anno['image_id']]
        assert img
        img['annos'].append({
            'category': categories[anno['category_id']],
            'bbox': anno['bbox'],
        })
    # The images with annotations are changed back to a list
    return list(images.values())


def render_image_annos(image, img_with_annos):
    # Draw the bounding boxes
    draw = ImageDraw.Draw(image, 'RGBA')
    for anno in img_with_annos['annos']:
        bbox = anno['bbox']
        xmin, ymin, w, h = tuple(bbox)
        draw.rectangle((xmin, ymin, xmin+w, ymin+h), outline='red', width=1)
        draw.text((xmin, ymin), anno['category'], fill='white')
    # Encode the image into a base64 string
    buffered = BytesIO()
    image.save(buffered, format='png')
    image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_str


def ask_chatgpt(img_with_annos):
    answers = list()
    cat_names = set()
    for anno in img_with_annos['annos']:
        cat = anno['category']
        if cat in cat_names:
            continue
        cat_names.add(cat)
        if cat in mocked_chatgpt:
            answer = mocked_chatgpt[cat]
        else:
            # TODO: Remove this hack
            answer = mocked_chatgpt['gray snapper']
        answers.append(answer)
    return answers


# YOLOS Start

coco_examples = [
    'http://images.cocodataset.org/test-stuff2017/000000000001.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000019.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000128.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000171.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000184.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000300.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000333.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000416.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000442.jpg',
]

feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
pipe = pipeline('object-detection', model=model, feature_extractor=feature_extractor)

def yolos_annotate(img_url):
    image = Image.open(requests.get(img_url, stream=True).raw)
    annos = pipe(image)
    # Convert to the format that we can render
    converted_annos = list()
    for anno in annos:
        converted_annos.append({
            'category': anno['label'],
            'bbox': (
                anno['box']['xmin'],
                anno['box']['ymin'],
                anno['box']['xmax']-anno['box']['xmin'],
                anno['box']['ymax']-anno['box']['ymin'],
            )
        })
    return image, {'annos': converted_annos}


# YOLOS End


image_annos = load_image_annos()


app = Flask(__name__)


@app.route('/')
def index():
    annotated_img = random.choice(image_annos)
    image = Image.open(Path('static/img') / annotated_img['file_name'])
    image_str = render_image_annos(image, annotated_img)
    answers = ask_chatgpt(annotated_img)
    return render_template('index.html', image_str=image_str, answers=answers)


@app.route('/yolos')
def yolos():
    img_url = random.choice(coco_examples)
    print(img_url)
    image, annotated_img = yolos_annotate(img_url)
    image_str = render_image_annos(image, annotated_img)
    answers = ask_chatgpt(annotated_img)
    return render_template('index.html', image_str=image_str, answers=answers)
