# PMLDL. Project Presentation

Anna Boronina

---

# The idea

<div class="row">
<div>

~~GANs for point clouds generation~~

~~VAE for point clouds generation~~

AE for point clouds
</div>
<img src="images/data/pointcloud.png">
</div>

<style>
    p {
        opacity: 1;
    }
    .row {
        margin-left: 0.5em;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    img {
        width: 70%;
    }
</style>

---

# Voxels and Their Problem

<div class="row">
<div>

- Structured

- You can apply 3D convolution

**BUT**

- Too large to work with

- Imagine 64x64x64 model - 262144 **voxels**

</div>
<img src="images/voxels.jpg">
</div>

<style>
    .row {
        margin-top: 5em;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-left: -1em;
    }
    img {
        width: 50%;
        margin-left: 1em;
        border-radius: 5px;
    }
</style>

---

# Point Clouds and Their Problem

<div class="row">
<div>

- You can sample N points

- Lightweighted

**BUT**

- Unstructured

- You cannot apply 3D convolution

</div>

<img src="images/pointcloud.png">
</div>


<style>
    .row {
        margin-top: 5em;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-left: -1em;
    }
    img {
        width: 50%;
        margin-left: 5.5em;
        border-radius: 5px;
    }
</style>

---

# PointNet

Permutation and rotation invariant!

<img src="images/pointnet.jpg">

<style>
    p {
        opacity: 1;
    }
</style>

---

# The dataset

<img src="images/data/shapenet.png">

<style>
    img {
        width: 97%;
    }
</style>

---

# Data Augmentation

1. `PointSampler`
2. `ToSorted`
3. `Normalize`
4. `RandomNoise`
5. `ToTensor` (obviously)

<div class="column">
    <img src="images/data/aug1.png">
    <img src="images/data/aug2.png">
</div>

<style>
    .column {
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
    img {
        width: 75%;
        margin: -0.5em;
        margin-right: 1em;
        margin-top: 0.05em;
        border-radius: 5px;
    }
</style>

---

# After Augmentation

<div class="column">
<div class="row">
    <img src="images/data/original_beds/bed1.png">
    <img src="images/data/original_beds/bed2.png">
</div>

<div class="row">
    <img src="images/data/original_beds/bed3.png">
    <img src="images/data/original_beds/bed4.png">
</div>
</div>

<style>
    .column {
        margin-top: 2em;
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
    .row {
        margin-bottom: 3.5em;
        margin-left: 0.5em;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    img {
        width: 30%;
        margin: -0.5em;
        margin-right: 1em;
        border-radius: 5px;
    }
</style>

---

# Architectures

<div class="row">
<div class="column"> 
    Encoder + Dense Decoder
    <img src="images/architecture/decoder_linear.png">
</div>
<div class="column">
    Encoder + Convolutional Decoder
    <img src="images/architecture/decoder_points.png">
</div>
</div>

<style>
    p {
        opacity: 1;
    }
    .column {
        margin-top: 1em;
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
    .row {
        margin-bottom: 3.5em;
        margin-left: 0.5em;
        display: flex;
        justify-content: center;
        align-items: flex-start;
    }
    img {
        width: 90%;
        padding-top: 2em;
        margin-right: 1em;
        border-radius: 5px;
    }
</style>

---

# Encoder

<img src="images/architecture/encoder.png">

<style>
    img {
        width: 100%;
        padding-top: 4em;
    }
</style>
---

# Intermediate Recap :)

1. **We saw the encoder**

2. **Two augmentations**
    - fully static
    - static + dynamic

3. **Two architectures**
    - dense decoder
    - convolutional decoder

---

# Dense Decoder

<img src="images/architecture/decoder_linear.png">

---

# Dense Decoder. Static augmentation

<div class="column">
    <img src="images/stats/dec1aug1train.png">
    <img src="images/stats/dec1aug1valid.png">
</div>

<style>
    p {
        opacity: 1;
    }
    .column {
        padding-top: 0.5em;
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
    img {
        width: 65%;
        border-radius: 5px;
    }
</style>

---

# Dense Decoder. Dynamic augmentation

<div class="column">
    <img src="images/stats/dec1aug3train.png">
    <img src="images/stats/dec1aug3valid.png">
</div>


<style>
    p {
        opacity: 1;
    }
    .column {
        padding-top: 0.5em;
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
    img {
        width: 65%;
        border-radius: 5px;
    }
</style>

---

# Dense Decoder. Comparison

<div class="column">
    Static Augmentation: train and valid batches
    <img src="images/stats/dec1aug1_trainvalidbatches.png">
    Dynamic Augmentation: train and valid batches
    <img src="images/stats/dec1aug3_trainvalidbatches.png">
</div>

<style>
    img {
        width: 70%;
    }
    .column {
        padding-top: 0.5em;
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
</style>

---

# Convolutional Decoder

<img src="images/architecture/decoder_points.png">

<style>
    img {
        width: 90%;
        padding-left: 4.5em;
        padding-top: 1em;
    }
</style>

---

# Convolutional Decoder. Static augmentation

<div class="column">
    <img src="images/stats/dec3aug1train.png">
    <img src="images/stats/dec3aug1valid.png">
</div>

<style>
    p {
        opacity: 1;
    }
    .column {
        padding-top: 0.5em;
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
    img {
        width: 65%;
        border-radius: 5px;
    }
</style>

---

# Convolutional Decoder. Dynamic augmentation

<div class="column">
    <img src="images/stats/dec3aug3train.png">
    <img src="images/stats/dec3aug3valid.png">
</div>

<style>
    p {
        opacity: 1;
    }
    .column {
        padding-top: 0.5em;
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
    img {
        width: 65%;
        border-radius: 5px;
    }
</style>

---

# Convolutional Decoder. Comparison

<div class="column">
    Static Augmentation: train and valid batches
    <img src="images/stats/dec3aug1_trainvalidbatches.png">
    Dynamic Augmentation: train and valid batches
    <img src="images/stats/dec3aug3_trainvalidbatches.png">
</div>

<style>
    img {
        width: 70%;
    }
    .column {
        padding-top: 0.5em;
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
</style>

---

# Main points

1. Purely static augmentation is evil (especially when there is no dropout)
2. Use 1D convolution to increase the number of dimensions
3. Dense and Convolutional Ecnoder have nearly the same performance on dynamically augmented data

<style>
    p {
        opacity: 1;
    }
</style>

---

# Results. Train Set

<div class="column">
    Loss: 2.4
    <img src="images/results/train_2.3.png">
    Loss: 4.3
    <img src="images/results/train_4.png">
    Loss: 52.1
    <img src="images/results/train_52.png">
</div>

<style>
    img {
        width: 40%;
    }
    .column {
        /* padding-top: 0.5em; */
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
</style>

---

# Results. Valid Set

<div class="column">
    Loss: 80.1
    <img src="images/results/valid_80.png">
    Loss: 84.9
    <img src="images/results/valid_84.png">
    Loss: 128.0
    <img src="images/results/valid_128.png">
</div>

<style>
    img {
        width: 40%;
    }
    .column {
        /* padding-top: 0.5em; */
        display: flex;
        justify-content: center;
        flex-flow: column;
        align-items: center;
    }
</style>

---

# Loss Function: Chamfer Distance

<img src="images/architecture/chamfer_dist.png">

<style>
    img {
        padding-top: 6em;
    }
</style>

---

# A discovery

**YAY**: use 1D convolution to increase the number of points

**NAY**: do not use it to increase the number of dimensions

<div class="row">
    <img src="images/architecture/decoder_channels.png" class="arch">
    <img src="images/results/badbad.png">
</div>

<style>
    p {
        opacity: 1;
    }
    .row {
        padding-top: 0.5em;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    img {
        width: 30%;
        border-radius: 5px;
    }
    .arch {
        width: 70%
    }
</style>

---

# Possible Further Work

1. Train for longer

2. Graph Neural Network

3. Variational Autoencoder

4. Change size of latent space

5. Use this architecture to train a GAN

---

# Variational Autoencoder

<img src="images/architecture/vaencoder.png">

<style>
    img {
        width: 65%;
        border-radius: 5px;
        margin-left: 7em;
    }
</style>

