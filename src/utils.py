import os
import gzip
import numpy as np
import urllib.request

URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

def download_mnist(data_dir):
    os.makedirs(data_dir,exist_ok=True)
    for name , url in URLS.items():
        filepath = os.path.join(data_dir, url.split("/")[-1])
        if not os.path.exists(filepath):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"{name} already exists, skipping.")

def read_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        data = f.read()
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(-1, 784) 

def read_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8, offset=8)

def normalize(X):
    return X / 255.0

def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded

def load_mnist(data_dir="data/"):
    download_mnist(data_dir)

    X_train = read_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = read_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test  = read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test  = read_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    X_train = normalize(X_train)
    X_test  = normalize(X_test)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train[:5])
    print("X_test:",  X_test.shape)
    print("y_test:",  y_test.shape)
    print("One-hot sample:", one_hot_encode(y_train[:3]))