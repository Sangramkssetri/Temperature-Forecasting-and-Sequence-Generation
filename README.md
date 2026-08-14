# Deep Learning for Temperature Forecasting and Sequence Generation

<div align="justify">
    
This project investigates the application of deep learning to multivariate temperature forecasting and generative time-series modelling. Three forecasting architectures based on gated recurrent units (GRUs) are developed to predict monthly mean minimum and maximum temperatures over multiple forecast horizons. The evaluated approaches comprise a direct multi-step forecaster, a recursive single-step forecaster, and an encoder–decoder sequence-to-sequence model. Their predictive performance is assessed using mean absolute error (MAE) on a held-out test set.

In addition, a variational autoencoder (VAE) is trained to generate synthetic maximum-temperature sequences. Diagnostic analysis of the learned latent representation indicates that only a small subset of latent dimensions contributes substantially to the generated output. This underutilisation of the latent space explains the limited diversity and excessive smoothness observed in the synthetic sequences.

All data processing, model development, evaluation, and visualisation are in 
[`Notebooks/temperature_forecasting.ipynb`](Notebooks/temperature_forecasting.ipynb).

## Project Objectives

The project addresses the following objectives:

1. Develop recurrent neural-network models for forecasting monthly minimum and maximum temperatures.
2. Compare direct, recursive, and sequence-to-sequence forecasting strategies across several prediction horizons.
3. Examine the effect of autoregressive error accumulation in recursive forecasting.
4. Train a VAE capable of generating plausible maximum-temperature sequences.
5. Investigate the structure and utilisation of the VAE latent space.

## Repository Structure

```text
.
├── Data/
│   └── temperatures-2025.pkl                  # Pickled (train_seq, val_seq, test_seq) sequences
├── Models/
│   ├── model1.keras                           # Model 1 — direct multi-step GRU forecaster
│   ├── model2.keras                           # Model 2 — recursive single-step GRU forecaster
│   ├── model3.keras                           # Model 3 — GRU encoder–decoder forecaster
│   ├── vae.keras                              # Full VAE (encoder + decoder)
│   ├── vae_encoder.keras                      # VAE encoder
│   └── vae_decoder.keras                      # VAE decoder (generates synthetic sequences)
├── Notebooks/
│   └── temperature_forecasting.ipynb          # Complete analysis notebook
├── requirement.txt
└── README.md
```

The `Models/` directory contains the trained models saved by the notebook.

## Dataset and Pre-processing

The dataset, `temperatures-2025.pkl`, contains pre-partitioned training, validation, and test sequences stored as the tuple:

```python
(train_seq, val_seq, test_seq)
```

Each timestep represents one month and contains two variables:

* monthly mean minimum temperature; and
* monthly mean maximum temperature.

Minimum temperatures are approximately distributed between 5 °C and 15 °C, while maximum temperatures generally range from 15 °C to 35 °C.

The `split_seq()` function transforms each complete sequence into an input window of length `input_len` and a corresponding prediction target of length `target_len`. This formulation permits the same data-processing pipeline to be applied across multiple forecasting horizons. The `display_temperatures()` function is used to compare historical inputs, ground-truth targets, and model predictions graphically.

## Forecasting Methodology

All forecasting models are implemented using TensorFlow and Keras. Training is conducted with the Nadam optimiser and MAE loss. To reduce overfitting and improve convergence, early stopping and `ReduceLROnPlateau` learning-rate scheduling are applied. Each model is trained for a maximum of 100 epochs with a batch size of 32.

MAE is reported in degrees Celsius:

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| y_i - \hat{y}_i \right|,
$$

where $y_i$ denotes the observed temperature and $\hat{y}_i$ denotes the corresponding model prediction.

### Model 1: Direct Multi-step GRU Forecaster

Model 1 predicts the complete forecast horizon in a single forward pass. For the principal 12-month experiment, the model receives 72 months of historical observations and directly produces predictions for the following 12 months.

Its architecture consists of:

* a GRU layer with 8 units;
* a GRU layer with 16 units;
* a fully connected output layer; and
* a reshape operation that produces the required temporal output structure.

The model contains 1,944 trainable parameters. Because all future timesteps are predicted simultaneously, the model does not feed its predictions back into subsequent inputs. It is therefore not directly affected by recursive error accumulation.

### Model 2: Recursive Single-step GRU Forecaster

Model 2 uses the same two-layer GRU backbone but predicts only the next month. Multi-month forecasts are generated recursively: each predicted observation is appended to the input sequence and used to predict the following month.

The model contains 1,570 trainable parameters and achieved a single-step test MAE of 1.0568 °C. Its relatively accurate one-step predictions support strong performance over short horizons. Nevertheless, because later forecasts depend on earlier predictions, estimation errors may accumulate as the forecast horizon increases.

### Model 3: GRU Encoder–Decoder Forecaster

Model 3 adopts an encoder–decoder, or sequence-to-sequence, architecture. Two stacked encoder GRUs with 8 and 16 units compress the 72-month input sequence into hidden-state representations. These states initialise a corresponding two-layer GRU decoder, which generates the 12-month output sequence through a final fully connected layer.

Teacher forcing is used during training by supplying the decoder with observed previous values. The complete model contains 3,106 trainable parameters, comprising 1,536 encoder parameters and 1,570 decoder parameters.

## Comparison Across Forecast Horizons

The `comparision()` routine retrains and evaluates all three models under several combinations of input and target lengths. The resulting test-set MAEs are presented below.

| Prediction horizon (months) | Model 1 MAE (°C) | Model 2 MAE (°C) | Model 3 MAE (°C) |
| --------------------------: | ---------------: | ---------------: | ---------------: |
|                          12 |           1.3332 |       **1.1942** |           1.3842 |
|                          18 |           1.4861 |       **1.3267** |           1.6801 |
|                          24 |       **1.2673** |           1.3319 |           1.4232 |
|                          30 |       **1.3382** |           2.1154 |           1.5509 |
|                          36 |           1.3314 |       **1.3154** |           1.4759 |

Model 2 achieves the lowest MAE at the 12-, 18-, and 36-month horizons. Its performance at shorter horizons is consistent with its strong single-step accuracy. However, its MAE increases substantially to 2.1154 °C at the 30-month horizon, illustrating the potential instability of recursive forecasting when prediction errors are repeatedly reintroduced into the input.

Model 1 achieves the lowest MAE at the 24- and 30-month horizons. Its errors remain within a comparatively narrow range of 1.2673–1.4861 °C, indicating greater stability across the evaluated horizons. Model 3 produces the highest MAE in every configuration, suggesting that the additional complexity of the encoder–decoder structure does not improve predictive accuracy for this dataset.

These values are point estimates for the evaluated experimental configurations. Repeated training runs or confidence intervals would be required to determine whether small performance differences—particularly the 0.016 °C difference between Models 1 and 2 at 36 months—are statistically meaningful.

## Variational Autoencoder

A VAE is trained on 84-month maximum-temperature sequences to model the underlying data distribution and generate synthetic observations.

### Encoder

The encoder applies the following transformations:

```text
Input (84)
    → min–max normalisation
    → Dense (64)
    → Dense (32)
    → latent mean (10) and latent log-variance (10)
    → sampled latent vector (10)
```

A custom `Sampling` layer implements the reparameterisation procedure:

$$
z = \mu + \exp\left(\tfrac{1}{2}\log\sigma^2\right)\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I).
$$

The layer also contributes the Kullback–Leibler divergence term required to regularise the approximate posterior distribution.

### Decoder

The decoder reconstructs an 84-month sequence according to:

```text
Latent vector (10)
    → Dense (32)
    → Dense (64)
    → Dense (84)
    → de-normalisation to degrees Celsius
```

The complete VAE contains 16,104 trainable parameters. Its objective combines MAE reconstruction loss with KL-divergence regularisation.

Six randomly sampled latent vectors produce smooth, periodic sequences within a plausible temperature range. Although these outputs reproduce broad seasonal structure, they exhibit limited variation and appear more regular than the observed sequences.

## Latent-space Analysis

Three diagnostic experiments are conducted to investigate the limited diversity of the generated data.

### Distribution of Latent Means

Kernel density estimates and per-dimension standard deviations show that most encoded latent means are concentrated near zero. This result indicates that the encoder maps observations into a relatively restricted region of the latent space and that several dimensions carry little information.

### Zero-vector Decoding

Decoding an all-zero latent vector produces a generic, low-amplitude sequence of approximately 22.5–24.5 °C. This output does not reproduce the full seasonal range of the observed maximum-temperature data.

### Selective Latent Injection

When non-zero values are introduced only into latent dimensions 3 and 5, the decoder produces a full-range periodic sequence. This finding demonstrates that these dimensions exert substantially greater influence over the generated output than the remaining latent variables.

Collectively, the results are consistent with latent-space underutilisation, potentially representing a form of partial posterior collapse. The decoder appears to rely predominantly on a small number of active dimensions, which accounts for the similarity, smoothness, and limited diversity of sequences generated from random latent samples.

## Conclusions

The experiments demonstrate that model performance depends on both the forecasting architecture and the prediction horizon. Recursive single-step forecasting provides the strongest performance for several evaluated horizons, particularly when the underlying one-step model is accurate. However, its susceptibility to cumulative prediction error can produce unstable long-range forecasts. The direct multi-step model offers more consistent performance across horizons, while the encoder–decoder architecture does not provide a measurable advantage for the present dataset.

The VAE successfully learns broad seasonal temperature patterns but does not fully exploit its ten-dimensional latent representation. Its synthetic outputs are plausible at a qualitative level but lack the variability observed in real temperature sequences. Future work could investigate alternative KL-divergence weighting, KL annealing, reduced latent dimensionality, more expressive decoder architectures, and quantitative measures of synthetic-data fidelity.

## Reproducibility

The project requires Python 3, TensorFlow 2.x, and the principal scientific Python libraries. Dependencies can be installed using:

```bash
pip install numpy pandas matplotlib seaborn tensorflow
```

The dataset is loaded within the notebook via the provided `load_data()` helper, which reads the pickled training, validation, and test sequences from the `Data/` directory:

```python
(train_seq, val_seq, test_seq) = load_data("./Data/temperatures-2025.pkl")
```

The complete analysis can then be executed with:

```bash
jupyter notebook Notebooks/temperature_forecasting.ipynb
```

</div>
