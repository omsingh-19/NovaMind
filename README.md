# NovaMind

okay so I built a neural network completely from scratch. no PyTorch or TensorFlow, just pure numpy and typing out every math equation by hand. after a million "shape not aligned" errors, it actually WORKS. it guesses hand-drawn numbers (MNIST) right 97.91% of the time!
what's inside
it takes 784 pixels, squishes them to 128 neurons, then 64, then spits out a guess (0-9). I had to code the dense layers, relu/softmax math, and write my own messy script to load the weird binary image files.
my stats
trained for 20 epochs with batches of 64. hit 99.69% training accuracy and 97.91% on the test set!
crazy math realization
doing the calculus by hand hurts, but the backward math for Softmax and Cross Entropy actually cancels out into just (predictions - actual) / batch_size. you literally don't realize this until you try to write it yourself.
there is a web demo!
there's a browser demo in the docs/ folder where you can draw a number and it guesses it. full disclosure: i used AI for the frontend part because javascript is scary. i just exported my python weights to JSON so it works in the browser.
