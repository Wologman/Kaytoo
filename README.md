# Kaytoo
New Zealand Bird Call Classification

- This is the repo for my bird call classifier.   This work is still in proof of concept stage.

- The model architecture is currently the same as my BirdCLEF 2024 submission.  Image classification on Mel Spectrograms, but with an attention layer applied along the frequency axis.  Training on 10 second samples, but inferring on 5 seconds by seperating the feature maps before global pooling.

- It is trained on internal DOC data, currently 87 species, 290,000 training samples.

- I haven't quite finished my analysis yet, but it is looking like the performance is good on the 39 most populous classes.  Anything with over 500 training samples has a recall of over 0.79 on my validation split.  But performance is inconsistent on the rest.  There is room for improvement in both the training methods and the dataset quality, that I hope to look into in coming months.

- Infers on 9 hours of .wav files per minute on my Dell G7 using 2 cores and an NVIDIA 1060 (Linux).
- It also runs (slowly) on a more typical corporate Dell Latitude 3420, on  Windows with a MX450 GPU. 
