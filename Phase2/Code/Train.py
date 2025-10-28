#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm


def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, ModelType="Sup"):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    ModelType - "Sup" for supervised, "Unsup" for unsupervised
    Outputs:
    For supervised: I1Batch, CoordinatesBatch
    For unsupervised: P_A, P_B, C_A, I_A, Combined
    """
    if ModelType == "Unsup":
        return GenerateUnsupervisedBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize)
    else:
        return GenerateSupervisedBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize)

def GenerateSupervisedBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    """Generate batch for supervised learning"""
    I1Batch = []
    CoordinatesBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)
        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        ImageNum += 1

        I1 = np.float32(cv2.imread(RandImageName))
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I1 = cv2.resize(I1, (ImageSize[0], ImageSize[1]))
        
        # For supervised: duplicate image as both channels
        I1_two_channel = np.stack([I1, I1], axis=0)
        Coordinates = TrainCoordinates[RandIdx]

        I1Batch.append(torch.from_numpy(I1_two_channel).float())
        CoordinatesBatch.append(torch.tensor(Coordinates).float())

    return torch.stack(I1Batch), torch.stack(CoordinatesBatch)

def GenerateUnsupervisedBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    """Generate batch for unsupervised learning with proper patch pairs"""
    P_A_batch = []
    P_B_batch = []
    C_A_batch = []
    I_A_batch = []
    Combined_batch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)
        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        ImageNum += 1

        # Load original image
        I_A = np.float32(cv2.imread(RandImageName))
        I_A = cv2.cvtColor(I_A, cv2.COLOR_BGR2GRAY)
        
        # Use original image size or resize to appropriate size for patch extraction
        # We need a larger image to extract 128x128 patches with margins
        h, w = I_A.shape[:2]
        if h < 200 or w < 200:
            # Resize to provide enough room for patches
            I_A = cv2.resize(I_A, (320, 240))
        
        h, w = I_A.shape[:2]
        
        # Generate patch pair with homography
        patch_size = 128
        pixel_shift_limit = 32
        
        # Extract random patch from safe region
        min_x = pixel_shift_limit
        min_y = pixel_shift_limit
        max_x = w - patch_size - pixel_shift_limit
        max_y = h - patch_size - pixel_shift_limit
        
        # Ensure valid range
        if max_x <= min_x or max_y <= min_y:
            # If image is too small, resize it to ensure we have room
            if w < 200 or h < 200:
                I_A = cv2.resize(I_A, (320, 240))
                h, w = I_A.shape[:2]
                max_x = w - patch_size - pixel_shift_limit
                max_y = h - patch_size - pixel_shift_limit
        
        # Fallback to smaller patches if still too small
        if max_x <= min_x or max_y <= min_y:
            patch_size = min(64, h - 2*pixel_shift_limit, w - 2*pixel_shift_limit)
            max_x = w - patch_size - pixel_shift_limit
            max_y = h - patch_size - pixel_shift_limit
        
        patch_x = random.randint(min_x, max_x)
        patch_y = random.randint(min_y, max_y)
        
        P_A = I_A[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
        
        # Define corner coordinates for patch A
        C_A = np.array([
            [patch_x, patch_y],                           # top-left
            [patch_x + patch_size, patch_y],              # top-right  
            [patch_x, patch_y + patch_size],              # bottom-left
            [patch_x + patch_size, patch_y + patch_size]  # bottom-right
        ], dtype=np.float32)
        
        # Generate perturbed corners for patch B
        common_translation_x = random.randint(-pixel_shift_limit, pixel_shift_limit)
        common_translation_y = random.randint(-pixel_shift_limit, pixel_shift_limit)
        
        C_B = np.zeros_like(C_A)
        for i in range(4):
            individual_perturbation_x = random.randint(-pixel_shift_limit, pixel_shift_limit)
            individual_perturbation_y = random.randint(-pixel_shift_limit, pixel_shift_limit)
            
            C_B[i][0] = C_A[i][0] + individual_perturbation_x + common_translation_x
            C_B[i][1] = C_A[i][1] + individual_perturbation_y + common_translation_y
        
        # Calculate homography and warp image
        H_AB = cv2.getPerspectiveTransform(C_A, C_B)
        I_B = cv2.warpPerspective(I_A, H_AB, (w, h))  # Use actual image dimensions
        
        # Extract patch B from warped image
        P_B = I_B[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
        
        # Resize patches to 128x128 if they're smaller
        if patch_size != 128:
            P_A = cv2.resize(P_A, (128, 128))
            P_B = cv2.resize(P_B, (128, 128))
        
        # Stack patches for combined input
        Combined = np.stack([P_A, P_B], axis=0)
        
        # Convert to tensors
        P_A_batch.append(torch.from_numpy(P_A).float().unsqueeze(0))  # Add channel dimension
        P_B_batch.append(torch.from_numpy(P_B).float().unsqueeze(0))
        C_A_batch.append(torch.from_numpy(C_A).float())
        I_A_batch.append(torch.from_numpy(I_A).float())
        Combined_batch.append(torch.from_numpy(Combined).float())

    return (torch.stack(P_A_batch), torch.stack(P_B_batch), 
            torch.stack(C_A_batch), torch.stack(I_A_batch), 
            torch.stack(Combined_batch))


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
    HomographyModel,
    LossFn,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel()

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            if ModelType == "Unsup":
                # Unsupervised learning: use full pipeline
                P_A, P_B, C_A, I_A, Combined = GenerateBatch(
                    BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, ModelType
                )
                
                # Use unsupervised forward pass
                P_A_warped, H4pt = model.unsupervised_forward(P_A, P_B, C_A)
                # P_A_warped and P_B are both (batch_size, 1, 128, 128)
                LossThisBatch = LossFn(P_A_warped, P_B)
            else:
                # Supervised learning: use standard approach
                I1Batch, CoordinatesBatch = GenerateBatch(
                    BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, ModelType
                )
                
                # Predict output with forward pass
                PredicatedCoordinatesBatch = model(I1Batch)
                LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            # Log training loss to tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                LossThisBatch.item(),
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="/Users/rohin/Documents/CV/AutoPano/Phase2/Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=1,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Dynamically import the appropriate model based on ModelType
    if ModelType == "Unsup":
        from Network.Network_Unsupervised import HomographyModel, LossFn
        print("Using Unsupervised Network")
    else:  # Default to Supervised
        from Network.Network_Supervised import HomographyModel, LossFn
        print("Using Supervised Network")

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
        HomographyModel,
        LossFn,
    )


if __name__ == "__main__":
    main()
