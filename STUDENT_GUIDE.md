# Fine-tuned Domain-adapted Infant Pose (FiDIP) - Student Guide

Welcome to the FiDIP project! This guide will walk you through a step-by-step process to learn about infant pose estimation using the FiDIP model. The project is based on the paper: "Invariant Representation Learning for Infant Pose Estimation with Small Data" by Xiaofei Huang, Nihang Fu, Shuangjun Liu, and Sarah Ostadabbas.

## Table of Contents
- [Introduction](#introduction)
- [Learning Path Overview](#learning-path-overview)
- [Part 1: Environment Setup](#part-1-environment-setup)
- [Part 2: Data Acquisition and Preparation](#part-2-data-acquisition-and-preparation)
- [Part 3: Understanding the Model Architecture](#part-3-understanding-the-model-architecture)
- [Part 4: Training the Model](#part-4-training-the-model)
- [Part 5: Testing and Evaluation](#part-5-testing-and-evaluation)
- [Part 6: Running Demos](#part-6-running-demos)
- [Part 7: Project Extensions](#part-7-project-extensions)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Introduction

The Fine-tuned Domain-adapted Infant Pose (FiDIP) estimation model transfers knowledge of adult poses to infant pose estimation using domain adaptation techniques on the Synthetic and Real Infant Pose (SyRIP) dataset. This project is particularly valuable for studying pose estimation with limited data, a common challenge in medical and healthcare applications.

## Learning Path Overview

This guide breaks down the learning process into manageable parts that can be completed over several weeks:

- **Week 1**: Environment setup and project understanding
- **Week 2**: Data acquisition and preparation
- **Week 3**: Understanding the model architecture
- **Week 4**: Training the model
- **Week 5**: Testing and evaluation
- **Week 6**: Running demos and visualization
- **Week 7+**: Project extensions and improvements

## Part 1: Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/yourusername/Infant-Pose-Estimation.git
cd Infant-Pose-Estimation
```

### 1.2 Create and Activate the Conda Environment

The project requires specific dependencies that are managed using Conda. Choose one of the following options based on your system configuration:

**Option 1**: Create the environment using the provided YAML file:

```bash
conda env create -f fidip_env.yml
conda activate fidip_cuda12.2
```

**Option 2**: If you encounter issues with the provided environment file, you can create a simplified environment:

```bash
conda env create -f fidip_simple.yml
conda activate fidip_simple
```

**Option 3**: For systems with CUDA 12.2, use the specific environment file:

```bash
conda env create -f fidip_cuda12.2.yml
conda activate fidip_cuda12.2
```

### 1.3 Verify GPU Support

To ensure your GPU is properly configured with PyTorch:

```bash
python gpu_test.py
```

You should see output confirming that CUDA is available and showing your GPU information.

### 1.4 Build the Project Libraries

```bash
cd lib
make
cd ..
```

## Part 2: Data Acquisition and Preparation

### 2.1 Download the SyRIP Dataset

1. Visit the [SyRIP dataset page](https://coe.northeastern.edu/Research/AClab/SyRIP/)
2. Download the following files:
   - `SyRIP.zip` (original SyRIP data)
   - `syrip_for_train.zip` (data for FiDIP model training)

Note: You will need to contact the authors for the password to access the dataset.

### 2.2 Extract and Organize the Data

```bash
mkdir -p data
unzip SyRIP.zip -d data/
unzip syrip_for_train.zip -d data/
```

Verify that your data directory structure matches the following:

```
${PROJECT_ROOT}
|-- data
`-- |-- syrip
    `-- |-- annotations
        |   |-- person_keypoints_train_pre_infant.json
        |   |-- person_keypoints_train_infant.json   
        |   `-- person_keypoints_validate_infant.json
        `-- images
            |-- train_pre_infant
            |   |-- train00001.jpg
            |   |-- ...
            |-- train_infant
            |   |-- train00001.jpg
            |   |-- ...
            |   |-- train10001.jpg
            |   |-- ...
            `-- validate_infant
                |-- test0.jpg
                |-- test1.jpg
                |-- ...
```

### 2.3 Download Pre-trained Models

1. Download DarkPose pre-trained models from [TGA_models](https://drive.google.com/drive/folders/14kAA1zXuKODYgrRiQmKnVcipbY7RedVV)
2. Download FiDIP pre-trained models from [FiDIP_models](https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC?usp=sharing)
3. Place them in the models directory with the following structure:

```
${PROJECT_ROOT}
`-- models
    |-- hrnet_fidip.pth
    |-- mobile_fidip.pth
    `-- coco
        `-- w48_384x288.pth
```

## Part 3: Understanding the Model Architecture

### 3.1 Project Structure Overview

Take time to understand the project structure:

- `lib/`: Core libraries and model implementations
  - `config/`: Configuration files
  - `core/`: Core functions for evaluation and training
  - `dataset/`: Dataset handling code
  - `models/`: Model architecture definitions
- `tools/`: Scripts for training, testing, and demos
- `experiments/`: Configuration files for different experiments
- `syn_generation/`: Code for synthetic data generation

### 3.2 Key Components

Study the following key components:

1. **HRNet Architecture**: The backbone network used for pose estimation
2. **Domain Adaptation**: How the model transfers knowledge from adult to infant poses
3. **Data Handling**: How the SyRIP dataset is processed and used

## Part 4: Training the Model

### 4.1 Understanding the Training Process

Before running the training, review the training script (`tools/train_adaptive_model_hrnet.py`) to understand:
- How data is loaded and processed
- The loss function used
- How the model is optimized
- How checkpoints are saved

### 4.2 Training Configuration

Review the configuration file for training:

```bash
cat experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
```

Understand the key parameters:
- Learning rate and optimizer settings
- Batch size and number of epochs
- Data augmentation settings
- Model architecture settings

### 4.3 Run Training

Start the training process:

```bash
python tools/train_adaptive_model_hrnet.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
```

Training will take several hours depending on your GPU. Monitor the training progress in the terminal output and the log files generated in the `log/` directory.

### 4.4 Monitor Training Progress

During training, you can monitor:
- Loss values
- Validation metrics
- GPU utilization (`nvidia-smi`)

### 4.5 Identifying Overfitting and Underfitting

Monitoring for overfitting and underfitting is crucial to develop a well-performing model:

#### Signs of Overfitting:
- **Training loss continues to decrease** while **validation loss starts to increase**
- **Large gap** between training and validation performance metrics
- Model performs exceptionally well on training data but poorly on validation data
- **Validation AP (Average Precision)** starts decreasing after initially improving

**How to address overfitting:**
- Implement early stopping (save the model when validation performance is best)
- Add regularization (L1/L2)
- Increase data augmentation
- Reduce model complexity
- Add dropout layers

#### Signs of Underfitting:
- **Both training and validation loss remain high** and don't decrease significantly
- **Poor performance metrics** on both training and validation sets
- Model predictions look overly simplified or miss important patterns

**How to address underfitting:**
- Train for more epochs
- Increase model complexity
- Reduce regularization strength
- Adjust learning rate
- Use a more powerful architecture

#### Monitoring Tools:
- Plot training and validation loss curves (should be available in `log/syrip/adaptive_pose_hrnet/`)
- Monitor AP and AR metrics during validation steps
- Visualize predictions on a small set of validation images periodically

## Part 5: Testing and Evaluation

### 5.1 Test the Pre-trained Model

To evaluate the pre-trained FiDIP model on the SyRIP validation dataset:

```bash
python tools/test_adaptive_model.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True
```

### 5.2 Test Your Trained Model

If you've trained your own model, you can test it using:

```bash
python tools/test_adaptive_model.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    TEST.MODEL_FILE output/syrip/adaptive_pose_hrnet/model_best.pth TEST.USE_GT_BBOX True
```

### 5.3 Understanding Evaluation Metrics

The test script will output several metrics:
- Mean Average Precision (mAP)
- AP50 and AP75 (Average Precision at IoU thresholds of 0.5 and 0.75)
- Average Recall (AR)

Take time to understand what these metrics mean and how they reflect model performance.

## Part 6: Running Demos

### 6.1 Command-Line Demo

The `demo.py` script allows you to run pose estimation on a single image:

```bash
python tools/demo.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    --image demo_test_data/00002.jpg \
    --model models/hrnet_fidip.pth \
    --output demo_output.jpg
```

Parameters:
- `--cfg`: Configuration file path
- `--image`: Input image path
- `--model`: Model file path
- `--output`: Output image path
- `--conf-threshold`: Confidence threshold for displaying keypoints (default: 0.3)
- `--output-size`: Size of the output image (default: 512,512)

### 6.2 Interactive Streamlit Demo

The project includes a Streamlit-based interactive demo:

1. Install Streamlit if not already installed:
   ```bash
   pip install streamlit
   ```

2. Run the Streamlit demo:
   ```bash
   streamlit run tools/streamlit_demo.py
   ```

3. Access the demo in your web browser at `http://localhost:8501`

4. Using the Streamlit interface:
   - Upload an image using the file uploader
   - Adjust confidence threshold using the slider
   - View the pose estimation results and confidence values

### 6.3 Using Your Own Images

Try the demo with your own infant images:
1. Place your images in the `temp_uploads/` directory
2. Run the demo with your image path
3. Analyze the results and confidence scores

## Part 7: Project Extensions

Once you've completed the basic workflow, consider these extensions:

### 7.1 Synthetic Data Generation

Explore the synthetic data generation pipeline in the `syn_generation/` directory. Follow the instructions in `syn_generation/README.md` to generate your own synthetic infant poses.

### 7.2 Model Improvements

Consider ways to improve the model:
- Try different backbone networks
- Experiment with different hyperparameters
- Implement data augmentation techniques

### 7.3 Application Development

Develop applications using the trained model:
- Real-time pose estimation with a webcam
- Batch processing of videos
- Integration with other healthcare applications

## Troubleshooting

### Common Issues and Solutions

1. **CUDA out of memory**
   - Reduce batch size in the configuration file
   - Use a model with fewer parameters

2. **Missing dependencies**
   - Ensure all required packages are installed: `pip install -r requirements.txt`
   - Check for compatible versions of PyTorch and CUDA

3. **Data loading errors**
   - Verify the data directory structure matches the expected format
   - Check file permissions

4. **Training not converging**
   - Check learning rate and optimizer settings
   - Verify data preprocessing steps

## References

1. Huang, X., Fu, N., Liu, S., & Ostadabbas, S. (2021). Invariant Representation Learning for Infant Pose Estimation with Small Data. IEEE International Conference on Automatic Face and Gesture Recognition (FG).

2. [Distribution-Aware Coordinate Representation for Human Pose Estimation](https://github.com/ilovepose/DarkPose)

3. [Learning from Synthetic Humans](https://github.com/gulvarol/surreal)

4. [SyRIP Dataset](https://coe.northeastern.edu/Research/AClab/SyRIP/)

5. [Expanded SyRIP Dataset](https://coe.northeastern.edu/Research/AClab/Expanded_SyRIP/)

---

## Acknowledgments

This guide is based on the FiDIP project developed by the Augmented Cognition Lab at Northeastern University. For questions or assistance, please contact the lab through their [website](http://www.northeastern.edu/ostadabbas/).
