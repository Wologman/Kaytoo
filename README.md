# Kaytoo
New Zealand Bird Call Classification

- This is the repo for my bird call classifier for New Zealand bird species.   This work is still in proof of concept stage, supported by the Predator Science and NPCP programs at the Department of Conservation.

- The current SED model architecture is based on my [BirdCLEF 2025](https://www.kaggle.com/competitions/birdclef-2025) submission.  Training on 12-second samples, with inference on concatenated 6-second samples with a window that hops along in 5-second increments.

- The model is trained on internal DOC data, currently 87 species, 290,000 training samples.  The data is *weakly labelled*, like most such birdcall datasets.  I have reasonable confidence in the *primary* bird for each training sample, but lower confidence in the list of *secondary* labels typically present for daytime birds in New Zealand.

- The model architecture will soon be replaced with one that works with time-frequency annotated soundscapes, both for the underlying dataset and at inference.  The predictions will then be visualisable in [Raven](https://www.ravensoundsoftware.com/software/raven-pro/), and can contribute to human-in-loop improvement of the underlying traning data.
  
- I am building a new data standard (Anqa), and will update the [New Zealand bird-sound dataset](https://www.kaggle.com/datasets/ollypowell/new-zealand-bird-sound)  into that format.  I hope to publish the new dataset on [Zenodo](https://zenodo.org/) by mid 2026.  The methods that I'm building up to support the Anqa data standard will be a part of the [wildpytools](https://github.com/Wologman/wildpytools) package.

- The underlying motivation for the focus on data standards it to enable *Regional Datasets* and accompanying *Regional Models* in a compatible format, so that methods become sharable, and models for different regions can be easily built from the same training code-base.  Also the model outputs will in a form that ecologists can more easily use for their statistical models.

## License

Apache License 2.0
