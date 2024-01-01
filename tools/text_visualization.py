# I'll create some simple visualizations as examples for the user's request.
# These will be basic demonstrations of the types of visualizations mentioned earlier.

import matplotlib.pyplot as plt
import numpy as np
import os
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from glob import glob

dir_name = ['overall_description', 'attribute_description', 'attribute_contribution']
descriptions = []


dir_name = os.path.join('dataset/TrainDataset/Desc_raw', 'attribute_description')
desc_list = glob(dir_name + '/*.txt')

for cur_file in desc_list:
    with open(cur_file, 'r') as f:
        descriptions.append(f.read())

# Example word cloud
wordcloud_text = ' '.join(descriptions)



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


'''
# Example PCA or
#  t-SNE (using dummy data)
# Generating some random data for demonstration
data = np.random.rand(8, 100)  # 8 descriptions, 100 features (dummy data)

# PCA Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
for i, txt in enumerate(descriptions):
    plt.annotate(txt.split()[0], (pca_result[i, 0], pca_result[i, 1]))
plt.title("PCA of Descriptions (Dummy Data)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_result = tsne.fit_transform(data)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
for i, txt in enumerate(descriptions):
    plt.annotate(txt.split()[0], (tsne_result[i, 0], tsne_result[i, 1]))
plt.title("t-SNE of Descriptions (Dummy Data)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
'''