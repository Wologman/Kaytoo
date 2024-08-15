# Kaytoo
New Zealand Bird Call Classification

- This is the repo for my bird call classifier.   This work is still in proof of concept stage.

- The model architecture is currently the same as my BirdCLEF 2024 submission.  Image classification on Mel Spectrograms, but with an attention layer applied along the frequency axis.  Training on 10 second samples, but inferring on 5 seconds by seperating the feature maps before global pooling.

- It is trained on internal DOC data, currently 93 species, 250,000 training samples.  

- Infers on 9 hours of .wav files per minute on my Dell G7 using 2 cores and an NVIDIA 1060. 
