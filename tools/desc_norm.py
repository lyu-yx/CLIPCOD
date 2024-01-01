import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
from glob import glob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

dir_name = ['overall_description', 'attribute_description', 'attribute_contribution']
descriptions = []


dir_name = os.path.join('dataset/TrainDataset/Desc_raw', 'attribute_description')
desc_list = glob(dir_name + '/*.txt')

for cur_file in desc_list:
    with open(cur_file, 'r') as f:
        descriptions.append(f.read())

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to normalize text
def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    normalized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Rejoin into a single string
    return ' '.join(normalized_tokens)

# Apply normalization to each description
normalized_descriptions = [normalize_text(desc) for desc in descriptions]

wordcloud_text = ' '.join(normalized_descriptions)




# mask generation
# Generate a circular mask
radius = 400  # Radius of the circle
center = (radius, radius)
x, y = np.ogrid[:radius * 2, :radius * 2]
mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2
mask = 255 * mask.astype(int)
tensor_mask = np.array(mask, dtype=np.int32)

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = ['.', ',', 'The', 'the', 'a', 'to', "a", "about", "above", "after", "again", "against", "all", 
                             "am", "an", "and", "any", "are", "aren't", "as", "at", 
                            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "can't", "cannot", "could", 
                            "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", 
                            "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
                            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", 
                            "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", 
                            "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
                            "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
                            "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
                            "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", 
                            "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", 
                            "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
                            "yours", "yourself", "yourselves", 'colors'], 
                min_font_size = 10,
                mask=tensor_mask,
                max_words=200).generate(wordcloud_text)

# Display word cloud
plt.figure(figsize = (6, 6), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show()