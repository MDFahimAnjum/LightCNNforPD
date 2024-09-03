# LightCNN: A Single-Layer Model Outperforming Deep-CNN and CRNN in PD Classification
In this work, we propose LightCNN, a lightweight Convolutional Neural Network (CNN) architecture designed for efficient and effective Parkinson's disease (PD) classification using EEG data. LightCNN features a simple architecture with a single convolutional layer followed by pooling and fully connected layers, emphasizing computational efficiency without sacrificing performance.

We benchmarked LightCNN against several established deep learning architectures known for their effectiveness in EEG-based PD classification, including a Convolutional Recurrent Neural Network (CRNN) that combines CNN and Gated Recurrent Unit (GRU) layers. Our results show that LightCNN outperforms all existing methods by significant margins across all key metrics: a 2.3% improvement in recall, a 4.6% increase in precision, a 0.1% advantage in AUC, a 4% boost in F1-score, and a 3.3% higher accuracy compared to the closet competitor. These findings highlight LightCNN's ability to deliver high performance while maintaining computational simplicity, making it a suitable candidate for deployment in resource-limited environments, such as mobile or embedded systems for EEG analysis.

In summary, LightCNN represents a significant step forward in EEG-based PD classification, demonstrating that a well-designed, lightweight model can achieve superior performance over more complex architectures. This work underscores the potential for simple, efficient models to meet the demands of modern healthcare applications, particularly in scenarios where resources are constrained.

## List of contents
1. [Introduction](#introduction)
2. [LightCNN: A Simple yet High-Performing CNN Architecture](#lightcnn-a-simple-yet-high-performing-cnn-architecture)
3. [Experiments](#experiments)
4. [Results](#results)
5. [Discussion & Conclusion](#discussion--conclusion)
6. [Appendix: LightCNN Model Parameters](#lightcnn-model-parameters)
6. [Appendix: Dataset](#dataset)
7. [Appendix: Steps to Run Codebase](#steps-to-run-codebase)

## Introduction
Parkinson's disease (PD) is a debilitating neurodegenerative disorder that demands early and accurate diagnosis to improve patient outcomes. Electroencephalography (EEG) is a promising tool for non-invasive monitoring of brain activity, offering potential for PD diagnosis. However, the complexity of EEG data presents significant challenges that require advanced machine learning models. Most existing deep-learning methods for PD classification using EEG data are complex and computationally expensive, which can hinder their practical application, especially in resource-constrained environments.

In this work, we propose **LightCNN**, a lightweight Convolutional Neural Network (CNN) architecture designed for efficient and effective classification of EEG data. The architecture of LightCNN is purposefully simple, featuring a single convolutional layer followed by pooling and fully connected layers. This streamlined design prioritizes computational efficiency without sacrificing performance, making it ideal for applications where resource constraints are a concern.

To benchmark the performance of LightCNN, we compared it against several established deep learning architectures that have shown strong results in PD classification using EEG data: the 13-layer Deep CNN [Oh et al. (2018)](https://link.springer.com/article/10.1007/s00521-018-3689-5), ShallowConvNet [Schirrmeister et al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), DeepConvNet [Schirrmeister et al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), and EEGNet [Lawhern et al. (2018)](http://stacks.iop.org/1741-2552/15/i=5/a=056013). Additionally, we included the Convolutional Recurrent Neural Network (CRNN) [Lee et al. (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8528094/), which combines CNN layers with Gated Recurrent Unit (GRU) layers to capture both spatial and temporal features.

Among these architectures, CRNN emerged as the closest competitor to LightCNN, demonstrating strong overall performance. However, our LightCNN model outperformed CRNN by significant margins: a 2.3% improvement in recall, a 4.6% increase in precision, a slight but notable 0.1% advantage in AUC, a 4% boost in F1-score, and a 3.3% higher accuracy. These results underscore LightCNN's capability to achieve a balanced and robust performance across key classification metrics, despite its simplicity. Our results demonstrate that a well-designed CNN architecture can effectively capture the necessary features for PD classification from EEG data, eliminating the need for more complex recurrent layers and such lightweight models can be both efficient and powerful.

The experiments were conducted using EEG data from 46 participants, comprising 22 individuals with PD and 24 healthy controls. The results demonstrate that LightCNN not only rivals but exceeds the performance of more complex architectures, offering a powerful and computationally efficient alternative for PD classification. This makes LightCNN a promising candidate for real-time applications and resource-constrained environments, where both accuracy and efficiency are critical.

## LightCNN: A Simple yet High-Performing CNN Architecture

### Architecture
The proposed model is a lightweight CNN architecture designed for efficient classification tasks, particularly in the context of EEG data. Its architecture is straightforward yet effective, featuring a single convolutional layer followed by pooling and fully connected layers. The design emphasizes simplicity and computational efficiency, making it suitable for applications where a balance between performance and resource constraints is necessary. The architecture is composed of the following layers:

1. **Convolutional Layer**: The input to the model is 1D signal from multiple EEG channels ($59\times 5F_s$). The first layer is a 1D convolutional layer with the same number of output channels (59). The kernel size was 11. The layer applies padding to ensure the output has the same length as the input. This layer captures local dependencies in the signal by sliding the convolutional kernel across the time dimension of each channel.

2. **Activation and Dropout**: Following the convolution, the Rectified Linear Unit (ReLU) activation function is applied to introduce non-linearity into the model. A dropout layer with a dropout rate of 0.1 is then used to prevent overfitting by randomly setting a fraction of the input units to zero during training.

3. **Pooling Layer**: The output from the convolutional layer is passed through an average pooling layer with kernel size same as signal length ($5F_s$), which reduces the dimensionality of the data by taking the average over the entire length of the signal for each channel, resulting in a condensed representation of size ($59\times 1$).

4. **Fully Connected Layer**: The pooled output is flattened and fed into a fully connected layer with two output node. Finally a softmax function is applied to obtain classification output.

Model parameters are given in [Appendix: LightCNN Model Parameters](#lightcnn-model-parameters).

### Training parameters
During the training, the batch size was set to 2 with a learning rate of $1\times 10^{-4}$. Adam optimizer was utilized with a total of 80 epochs for the training.

## Experiments
### EEG Dataset
For our experiments, we used an EEG dataset of 54 participants from a study at the University of New Mexico (UNM; Albuquerque, New Mexico) where 27 had PD and the rest of the participants were healthy which was previously described in [Anjum et. al. (2020)](https://www.sciencedirect.com/science/article/abs/pii/S1353802020306672). Upon manual inspection, we utilized EEG data from 46 participants (22 PD and 24 healthy subjects). EEG data were recorded with a sampling rate of 500 Hz on a 64-channel Brain Vision system. PD patients were in OFF medication state.

### Data preprocessing
In this work, we utilize EEG data from 59 channels out of 63 based on average channel data quality. The data from each channel were high-pass filtered at 1 Hz to remove noise. No other pre-processing was implemented. Only the first one minute of EEG data from each participant was utilized, which corresponds to eyes closed resting state EEG.
The multi-channel data ($5n$ seconds) for each subject ($\mathbb{R}^{59\times 5n F_s}$) were converted into 5-second segments ($\mathbb{R}^{n\times 59\times 5F_s}$).

### Experimental setup
First, we randomly shuffled data at the subject level and split the dataset into training (60\%), validation (20\%), and test (20\%) datasets. We utilized the training data for training the models. The validation data were used for evaluating the model's performance against overfitting and the best-performing model on the validation set was selected. To measure the classification performance, we utilized five metrics: precision, recall, accuracy, F1-score, and AUC. The classification performance was evaluated on the test dataset. 

### State-of-the-art methods
To benchmark the performance of our proposed approach, we utilized five deep-learning architectures that have been shown to perform well in PD classification using EEG data: 13-layer Deep CNN [Oh et. al. (2018)](https://link.springer.com/article/10.1007/s00521-018-3689-5), ShallowConvNet [Schirrmeister et. al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), DeepConvNet [Schirrmeister et. al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), EEGNet [Lawhern et. al. (2018)](http://stacks.iop.org/1741-2552/15/i=5/a=056013) and Convolutional Recurrent Neural Network (CRNN) [Lee et. al. (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8528094/). All methods were deep CNN architectures execpt for CRNN which utilized CNN and Gated Recurrent Unit (GRU) layers. We chose these methods as they were shown to be very effective neural network architectures tailored for EEG-based PD classification in the literature. Model performances were evaluated on the test dataset while training and validation datasets were utilized for the training stage.

## Results
Our results show that our simple single layer CNN model outperformed the state-of-the-art architectures in all metrics. Among the four architectures compared in this study, CRNN provided the best overall performance. Our LightCNN model outperformed CRNN by 2.3% in recall, 4.6% in precision, 0.1% in AUC, 4% in F1-score, and 3.3% in accuracy. Recall that CRNN employs GRU layer which is computationally expensive.

| Method                         | Architecture   | Precision | Recall | F1-score | AUC  | Accuracy |
|--------------------------------|----------------|-----------|--------|----------|------|----------|
| [Oh et. al. (2018)](https://link.springer.com/article/10.1007/s00521-018-3689-5)                    | DeepCNN        | 60.0      | 40.9   | 0.49     | 0.629 | 58.7     |
| [Schirrmeister et. al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)             | ShallowConvNet | 80.5      | 75.0   | 0.78     | 0.831 | 79.3     |
| [Schirrmeister et. al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)                | DeepConvNet    | 87.8      | 81.8   | 0.85     | 0.917 | 85.9     |
| [Lawhern et. al. (2018)](http://stacks.iop.org/1741-2552/15/i=5/a=056013)                 | EEGNet-8,2     | 88.6      | 88.6   | 0.89     | 0.967 | 89.1     |
| [Lee et. al. (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8528094/)                       | CNN+GRU        | 95.4      | 95.4   | 0.95     | 0.997 | 95.6     |
| **LightCNN (Ours)**            | LightCNN    | **100**   | **97.7** | **0.99** |**0.998**| **98.9** |




## Discussion & Conclusion
In this study, we introduced LightCNN, a streamlined yet highly effective CNN architecture tailored for PD classification using EEG data. Despite its simplicity, our single-layer CNN model demonstrated remarkable performance, surpassing state-of-the-art (SOTA) architectures across all evaluation metrics. This achievement underscores the potential of minimalist architectures to deliver superior results without the computational complexity often associated with more sophisticated models.

Among the four architectures compared, CRNN emerged as the closest competitor, providing strong overall performance. However, our LightCNN model outperformed CRNN by significant margins: a 2.3% improvement in recall, a 4.6% increase in precision, a 4% boost in F1-score, and a 3.3% higher accuracy. These results highlight LightCNN's effectiveness in achieving a balanced and robust performance across key classification metrics. CRNN's use of a GRU layer, while effective, introduces substantial computational demands and challenges in scalability, particularly for large-scale or real-time applications. In contrast, LightCNN's architecture avoids these complexities, offering a more efficient and scalable alternative without compromising on performance. The simplicity of our approach not only facilitates easier implementation but also makes it more adaptable to scenarios where computational resources are constrained.

These findings have important implications. First, they demonstrate that a well-designed CNN architecture can effectively capture the necessary features for PD classification from EEG data, eliminating the need for more complex recurrent layers. Second, the performance gains achieved by LightCNN suggest that lightweight models can be both efficient and powerful, making them suitable for deployment in resource-limited environments, such as mobile or embedded systems.

Future research could explore the generalizability of LightCNN to other neurodegenerative disorders or broader EEG-based classification tasks. Additionally, integrating techniques like model quantization or pruning could further enhance LightCNN's efficiency, making it an even more attractive option for real-time and edge computing applications.

In conclusion, LightCNN represents a compelling approach to EEG-based PD classification, combining simplicity, efficiency, and high performance. Its ability to outperform more complex architectures while maintaining computational efficiency, positions LightCNN as a valuable tool for both research and clinical applications in the field of neurodegenerative disease detection.

## Appendix

### LightCNN Model Parameters

| Layer                | Output Size       | Parameters                                |
|----------------------|--------------------|-------------------------------------------|
| **Input**            | (59, 2500)       | -                                         |
| **Conv1D**           | (59, 2500)       | 38,350                                     |
| **AvgPool1D**        | (59, 1)          | -                                         |
| **Dropout**          | (59, 1)          | -                                         |
| **Flatten**          | (59)             | -                                         |
| **Linear (FC)**      | (2)              | 120                                       |  

### Dataset 
We use EEG dataset of 28 PD and 28 control participants.
- Original dataset can be found at [link](http://predict.cs.unm.edu/downloads). The data are in .mat formats and you need Matlab to load them. (No need for this unless you are interested into the original EEG data)
- Raw CSV dataset used for this repo can be found at [link](https://www.dropbox.com/scl/fi/xinqn33vof0bnb9rlvmdh/raw.zip?rlkey=jb4dyumh7v82vbj36wsb53x13&dl=0). Download this before running all steps in this [repo](https://github.com/MDFahimAnjum/LightCNNforPD).


### Steps to Run Codebase
1. Clone/Fork/Download the codebase from this [Repository](https://github.com/MDFahimAnjum/LightCNNforPD)
2. Download raw CSV dataset (can be found at [link](https://www.dropbox.com/scl/fi/xinqn33vof0bnb9rlvmdh/raw.zip?rlkey=jb4dyumh7v82vbj36wsb53x13&dl=0)) and place them in the `data/raw` folder
3. Next, the data must be processed. Run `data_processing` notebook which loads raw data and prepares training, validation and test dataset.

4. Run the models and evaluate performance
    1. `cnn_classifier` notebook uses CNN model as described in [Oh et. al. (2018)](https://link.springer.com/article/10.1007/s00521-018-3689-5)
    2. `deepnet_classifier` notebook uses Deep Convolutional Network as described in [Schirrmeister et. al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
    3. `shallownet_classifier` notebook uses Shallow Convolutional Network as described in [Schirrmeister et. al. (2017)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
    4. `eegnet_classifier` notebook uses EEGNet as described in [Lawhern et. al. (2018)](http://stacks.iop.org/1741-2552/15/i=5/a=056013)
    5. `crnn_classifier` notebook uses Convolutional-Recurrent Neural Network as described in [Lee et. al. (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8528094/)
    6. `light_cnn_classifier` notebook uses our proposed lightweight CNN architecture. 

5. If you want to generate the CSV files yourself, use the Matlab script `matlab_process.m` in `MatLab Codes` folder. You will need Matlab and [EEGLab](https://sccn.ucsd.edu/eeglab/index.php). The script will generate CSV files and plots which you can manually inspect to see which data were too corrupted with noise and you can ignore those. 