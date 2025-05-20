import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_model(model, train_loader, val_loader, criterion, optimizer):
  EPOCHS = 15
  train_samples_num = 45000
  val_samples_num = 5000
  train_epoch_loss_history, val_epoch_loss_history = [], []

  for epoch in range(EPOCHS): # loop for each epoch

    train_running_loss = 0
    correct_train = 0

    model.train()
    model.to(device)

    for inputs, labels in train_loader: # loop for each batch
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()

      # forward pass
      prediction0, aux_pred1, aux_pred2 = model(inputs)

      # backward pass
      real_loss = criterion(prediction0, labels)
      aux_loss1 = criterion(aux_pred1, labels)
      aux_loss2 = criterion(aux_pred2, labels)

      loss = real_loss + 0.3 * aux_loss1 + 0.3 * aux_loss2

      # backward pass
      loss.backward()
      optimizer.step()

      # update the correct values
      _, predicted = torch.max(prediction0.data, 1) # dim=1 means across the rows
      correct_train += (predicted == labels).float().sum().item()

      # uptil now we have calculated the avg loss
      # so no we will have to calculate the batch loss as well
      # for that we multiply avg batch loss with the batch length
      train_running_loss += loss.data.item() * inputs.shape[0] # 0 ele of inputs is always the batch size

    train_epoch_loss = train_running_loss / train_samples_num
    train_epoch_loss_history.append(train_epoch_loss)

    train_acc = correct_train / train_samples_num

    val_loss = 0
    correct_val = 0

    model.eval()
    model.to(device)

    with torch.no_grad(): # computign the val accuracy so we switch off the gradient calculcation
      for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass.
        prediction0, aux_pred_1, aux_pred_2 = model(inputs)

        # Compute the loss
        real_loss = criterion(prediction0, labels)
        aux_loss_1 = criterion(aux_pred_1, labels)
        aux_loss_2 = criterion(aux_pred_2, labels)

        loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2

        # Compute training accuracy
        _, predicted = torch.max(prediction0.data, 1)
        correct_val += (predicted == labels).float().sum().item()

        # Compute batch loss
        val_loss += loss.data.item() * inputs.shape[0]

      val_loss /= val_samples_num
      val_epoch_loss_history.append(val_loss)
      val_acc = correct_val / val_samples_num

    info = "[For Epoch {}/{}]: train-loss = {:0.5f} | train-acc = {:0.3f} | val-loss = {:0.5f} | val-acc = {:0.3f}"

    print(info.format(epoch + 1, EPOCHS, train_epoch_loss, train_acc, val_loss, val_acc))

    torch.save(model.state_dict(), "/content/sample_data/checkpoint{}".format(epoch + 1))

  torch.save(model.state_dict(), "/content/sample_data/googlenet_model")

  return train_epoch_loss_history, val_epoch_loss_history
