from PIL import Image
from transformers import AutoImageProcessor, TFAutoModelForImageClassification, pipeline, ViTFeatureExtractor


def get_pipeline(model_name, model_paths):
    image_processor = AutoImageProcessor.from_pretrained(model_paths[model_name], from_pt=True)
    model = TFAutoModelForImageClassification.from_pretrained(model_paths[model_name], from_pt=True)
    pipeline_ = pipeline('image-classification', model=model, image_processor=image_processor, device=None)
    return pipeline_


def predict(filepath, pipeline_):
    image = Image.open(filepath)
    pred = pipeline_(image)
    max_score_entry = max(pred, key=lambda x: x['score'])

    highest_score_label = max_score_entry['label']

    return highest_score_label
