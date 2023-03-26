import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def create_data(path):
    data = []
    labels = []
    try:
        for fruit_folder in os.listdir(path):
            for image_file in os.listdir(os.path.join(path, fruit_folder)):
                if image_file.endswith(".jpg") or image_file.endswith(".png"):
                    img = cv2.imread(os.path.join(path, fruit_folder, image_file))
                    if img.shape[0] > 0 and img.shape[1] > 0:
                        # Add color features
                        red_channel = img[:,:,0]
                        green_channel = img[:,:,1]
                        blue_channel = img[:,:,2]
                        red_mean = np.mean(red_channel)
                        green_mean = np.mean(green_channel)
                        blue_mean = np.mean(blue_channel)
                        color_features = [red_mean, green_mean, blue_mean]

                        # Resize and convert to grayscale
                        img = cv2.resize(img, (100, 100))
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # Extract HOG features
                        win_size = (100, 100)
                        block_size = (20, 20)
                        block_stride = (10, 10)
                        cell_size = (10, 10)
                        nbins = 9
                        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
                        hog_features = hog.compute(gray).flatten()

                        # Combine color and HOG features
                        all_features = np.concatenate((color_features, hog_features))

                        data.append(np.array(all_features))
                        labels.append(fruit_folder)
                    else:
                        continue

        return np.array(data), np.array(labels)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Load the dataset
data, labels = create_data('c:/Users/Ani/ImgRecProj/fruit-dataset/archive/train-zip/train')
if data is not None and labels is not None:
    print(f"Successfully loaded {len(data)} images and {len(labels)} labels")
else:
    print("Failed to load dataset")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Extract HOG features from the training images
hog_features = []
for img in X_train:
    hog_features.append(img)

# Train a SVM classifier on the extracted features with regularization
clf = SVC(kernel='linear', C=0.1)
clf.fit(hog_features, y_train)

# Evaluate the classifier on the testing set
test_features = []
for img in X_test:
    test_features.append(img)

# Make a prediction
y_pred = clf.predict(test_features)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Save the model
joblib.dump(clf, 'svm_model.joblib')

