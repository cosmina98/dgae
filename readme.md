# Vector-Quantized Graph Auto-Encoder

## Welcome to the official repository of the Vector-Quantized Graph Auto-Encoder (VQ-GAE)!

![Autoencoder image](autoencoder.png)

This repository contains the source code for the VQ-GAE, a powerful autoencoder for graph data that uses vector quantization to learn discrete representations. To get started with the VQ-GAE, please make sure that your system meets the requirements listed in the requirements.txt file.

Once you have met the requirements, you can run the VQ-GAE by executing the main.py file. The --work_type argument allows you to specify the type of work you want to do with the VQ-GAE. The --dataset argument specify the dataset, you want to use (ego, community, qm9 or zinc). You can choose to train the autoencoder, train the prior, or sample from a trained model.

You need to specify the configuration in a yaml file and put it in the 'config' folder. Files with the default parameters are available in the folder. 

If you choose to train the prior or sample from a model, you will need to specify a folder using the --model_folder argument. This folder should contain the configuration for your model, which we currently store using Weight and Bias.

Please note that the VQ-GAE downloads the required data by itself. However, be aware that the preprocessed (kekulized) zinc dataset is 5.4 GB in size.

Thank you for using the VQ-GAE.

