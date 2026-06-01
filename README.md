🎯 Project Objective
This project explores the use of an autoencoder for feature extraction on the Fashion‑MNIST dataset, followed by a latent‑space classifier to perform image classification.
The goal is to evaluate whether compressed latent representations can achieve competitive accuracy compared to a direct CNN baseline trained on raw pixels.

⚙️ Pipeline Overview
Autoencoder: Learns to compress Fashion‑MNIST images into a low‑dimensional latent space and reconstruct them.

Latent Classifier: Predicts clothing categories from the latent vectors.

CNN Baseline: A standard convolutional neural network trained directly on raw images, used as a benchmark.

📊 Results
Autoencoder + LatentClassifier	0.0060	0.2968	0.8926 (~89%)
CNN Baseline (raw pixels)	–	0.0443	0.9849 (~98%)

🚀 How to Run

git clone https://github.com/chong2006s/Fashion-Autoencoder-Classifer.git

cd Fashion-Autoencoder-Classifer

pip install -r requirement.txt

python train_pipeline.py

python evaluate.py

python base_line.py


📝 Note
Training for more epochs can steadily improve accuracy as the models converge toward optimal weights. Results may also vary slightly across different PCs depending on whether training is run on CPU or GPU, due to differences in computation speed and floating‑point precision.
