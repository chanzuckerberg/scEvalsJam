## Model Training Demo

We provide the following basic demo as a starting point to training perturbation models using this repo.

To train a SAMS-VAE model on the Replogle dataset, run:
`python ../train.py --config sams_vae_replogle.yaml`

The example config file, `sams_vae_replogle.yaml`, has been annotated with additional explanation regarding config structure
and can be used as a starting point for setting up new training runs.

As the model trains, the training metrics and the model checkpoints can be visualized using `visualize_results.ipynb`

For examples of training model sweeps, see the sweep configs and instructions in `../paper/experiments/`
