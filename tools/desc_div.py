import os
import json
from glob import glob
import re
# mkdir 
dir_name = ['overall_description', 'attribute_description', 'attribute_contribution']
for name in dir_name:
    dir_name = os.path.join('dataset/TrainDataset/Desc_raw', name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)



class_names = [
    "Background Matching", "Surrounding Pattern Disruption", "Environmental Motion Dazzle",
    "Environmental Shading", "Environmental Textures", "Color Matching", "Shape Mimicry",
    "Behavior Mimicry", "Texture", "Shadow Minimization", "Edge Diffusion", "Blur Issue",
    "Low Resolution", "Improper Exposure", "Compression Artifacts", "Object Size Matters", "Object Placement"
]

import re

def extract_weights(text, class_names):
    weights = {class_name: 0.0 for class_name in class_names}  # Initialize weights for each class
    found_classes = set()  # To keep track of classes found
    flag = True
    # Split the text into lines
    lines = text.split('\n')

    # Iterate through each class name
    for class_name in class_names:
        # Handle variations for "Texture" and "Blur Issue"
        if class_name == "Texture":
            pattern = r"Texture( \w+)?( \([^)]+\))?:?\s*([0-9.]+)"
        elif class_name == "Blur Issue":
            pattern = r"Blur( \w+)?( \([^)]+\))?:?\s*([0-9.]+)"
        else:
            # Regex to handle additional qualifiers in class names
            pattern = re.escape(class_name) + r"( \([^)]+\))?:?\s*([0-9.]+)"

        # Check each line for the class name
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            match = re.search(pattern, line)
            if match:
                # Save the number after the colon
                weights[class_name] = float(match.group(3) if class_name in ["Texture", "Blur Issue"] else match.group(2))
                found_classes.add(class_name)
                break  # Stop searching after the first match for this class

    # Check if all 17 classes were found
    if len(found_classes) != 17:
        print("Not all classes were found. Only found:", len(found_classes))
        flag = False

    # Print class names not found in the text
    not_found_classes = [class_name for class_name in class_names if class_name not in found_classes]
    if not_found_classes:
        print("Classes not found or mismatched:", not_found_classes)

    # Convert the dictionary to a list of weights in the order of class_names
    weights_list = [weights[class_name] for class_name in class_names]

    # Normalize the weights if their sum is not 1
    total_weight = sum(weights_list)
    if total_weight != 1 and total_weight != 0:
        weights_list = [weight / total_weight for weight in weights_list]

    return weights_list, flag




desc_list = glob('dataset/TrainDataset/Desc_raw/*.json')
fail_list = []
cnt = 0
for desc_path in desc_list:
    with open(desc_path) as json_file:
        data = json.load(json_file)
        # read the content key in the json file
        try:
            content = data["choices"][0]["message"]["content"]
            # save the first part of the content to overall_description
            
            overall_description = re.split(r"\n\d\.\s", content)[0]
            if overall_description.startswith("1. "):
                overall_description = overall_description[3:]
            with open('dataset/TrainDataset/Desc_raw/overall_description/' + desc_path.split("\\")[-1][:-5] + '.txt', 'w') as f:
                f.write(overall_description)

            # save the second part of the content to attribute_description
            attribute_description = re.split(r"\n\d\.\s", content)[1]
            if attribute_description.startswith("2. "):
                attribute_description = attribute_description[3:]
            with open('dataset/TrainDataset/Desc_raw/attribute_description/' + desc_path.split("\\")[-1][:-5] + '.txt', 'w') as f:
                f.write(attribute_description)
            # save the third part of the content to attribute_contribution
            attribute_contribution = re.split(r"\n\d\.\s", content)[2]
            weights_single_line, correct = extract_weights(attribute_contribution, class_names)
            if correct:
                cnt += 1
            else:
                fail_list.append(desc_path)
            with open('dataset/TrainDataset/Desc_raw/attribute_contribution/' + desc_path.split("\\")[-1][:-5] + '.txt', 'w') as f:
                for weight in weights_single_line:
                    f.write(f"{weight}\n")
        except Exception as e:
            fail_list.append(desc_path.split("\\")[-1][:-5])
            print(f"Error processing {desc_path}: {e}")
print("correct counted:", cnt)
print(f'total length:{len(fail_list)}')
print(fail_list)