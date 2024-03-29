background_path = r'C:\Users\mdhim\Videos\PDD\bg2.jpg'
model_state_dict_path = 'vision_transformer_image_classifier.pth'

disease_remedies = {
    'Apple__Apple_scab': 'Apply fungicide and prune infected areas.',
    'Apple_Black_rot': 'Prune and destroy infected parts; apply fungicide.',
    'Apple_Cedar_apple_rust': 'Remove galls; apply fungicide.',
    'Apple_healthy': 'No remedy needed, plant is healthy.',
    'Blueberry_healthy': 'No remedy needed, plant is healthy.',
    'Cherry(including_sour)Powdery_mildew': 'Apply fungicide, improve air circulation.',
    'Cherry(including_sour)healthy': 'No remedy needed, plant is healthy.',
    'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot': 'Use disease-resistant varieties, apply fungicides.',
    'Corn(maize)Common_rust': 'Plant resistant varieties, apply fungicides early.',
    'Corn_(maize)Northern_Leaf_Blight': 'Rotate crops, use resistant varieties, apply fungicides.',
    'Corn(maize)healthy': 'No remedy needed, plant is healthy.',
    'Grape_Black_rot': 'Prune affected areas, apply fungicide.',
    'Grape_Esca(Black_Measles)': 'Prune and destroy infected parts, apply fungicide.',
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': 'Prune affected leaves, apply fungicide.',
    'Grape__healthy': 'No remedy needed, plant is healthy.',
    'Orange_Haunglongbing(Citrus_greening)': 'Remove infected trees, control psyllid vectors.',
    'Peach__Bacterial_spot': 'Prune affected parts, apply copper-based fungicide.',
    'Peach_healthy': 'No remedy needed, plant is healthy.',
    'Pepper,_bell_Bacterial_spot': 'Remove infected plants, apply copper-based fungicides.',
    'Pepper,_bell_healthy': 'No remedy needed, plant is healthy.',
    'Potato_Early_blight': 'Remove affected leaves, apply fungicides.',
    'Potato_Late_blight': 'Remove and destroy affected plants, apply fungicides.',
    'Potato_healthy': 'No remedy needed, plant is healthy.',
    'Raspberry_healthy': 'No remedy needed, plant is healthy.',
    'Soybean_healthy': 'No remedy needed, plant is healthy.',
    'Squash_Powdery_mildew': 'Apply fungicides, improve air circulation.',
    'Strawberry_Leaf_scorch': 'Remove infected leaves, apply fungicides.',
    'Strawberry_healthy': 'No remedy needed, plant is healthy.',
    'Tomato_Bacterial_spot': 'Remove and destroy infected plants, apply copper-based fungicides.',
    'Tomato_Early_blight': 'Remove affected leaves, apply fungicides.',
    'Tomato_Late_blight': 'Remove and destroy infected plants, apply fungicides.',
    'Tomato_Leaf_Mold': 'Improve air circulation, apply fungicides.',
    'Tomato_Septoria_leaf_spot': 'Remove infected leaves, apply fungicides.',
    'Tomato_Spider_mites Two-spotted_spider_mite': 'Use insecticidal soap, neem oil; improve humidity levels.',
    'Tomato_Target_Spot': 'Remove infected leaves, apply fungicides.',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies, remove infected plants.',
    'Tomato_Tomato_mosaic_virus': 'Remove infected plants, control aphids.',
    'Tomato__healthy': 'No remedy needed, plant is healthy.'
}