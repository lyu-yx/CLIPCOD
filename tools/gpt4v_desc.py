import base64
import requests
import json
import os
from glob import glob
# OpenAI API Key
api_key = "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"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

images_list = glob(os.path.join('dataset', 'TrainDataset', 'Imgs', '*.jpg'))
# images_list[480:] images_list[480:980]
startnum = 3872

for image_path in images_list[startnum:]:
    # Getting the base64 string
    # print(image_path)
    base64_image = encode_image(image_path)
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": 
            '''
            Here is an image contains camouflaged object(s), answer the following questions one by one.
            1. Provide a detailed 30-word description of the image focusing on the camouflaged objects and their immediate environment.
            2. Describe how the camouflaged object(s) achieve camouflage. Please provide a detailed description within 30 words.
            3. Here are some camouflaged strategies or reasons which make this camouflage succuss. Surrounding Reasons: Background Matching, Surrounding Pattern Disruption, Environmental Motion Dazzle, Environmental Shading, Environmental Textures. Camouflaged Object-Self Reasons: Color Matching, Shape Mimicry, Behavior Mimicry, Texture, Shadow Minimization, Edge Diffusion. Imaging Quality Reasons: Blur Issue, Low Resolution, Improper Exposure, Compression Artifacts, Object Size Matters, Object Placement. Among them, Surrounding Reasons, Camouflaged Object-Self Reasons, and Imaging Quality Reasons are main classes while others are finer classes. 
            Calculate the exact contribution proportions of each finer class to the success of a camouflage in that image which make human hard to detect. Ensure that the total sum of these proportions equals 1. Include all relevant finer classes, even if their contribution is 0. Allocate proper contribution to the classes in “Imaging Quality Reasons” to make sure it is not always 0. Provide the results as a list, without any additional explanations. The focus is on precision and completeness in representing each finer class's contribution.
            '''
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
      "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    filename = os.path.basename(image_path).split('.')[0]
    file_path = os.path.join('dataset', 'TrainDataset', 'Desc_raw', filename + '.json')
    msg = response.json()
    startnum += 1
    with open(file_path, 'w') as file:
        json.dump(response.json(), file, indent=4)
    if 'error' in msg and msg["error"]["code"] == "rate_limit_exceeded":
       print("Rate limit exceeded. Waiting 5 seconds...")
       print(filename)
       print('starnum:', startnum)
       break
    
