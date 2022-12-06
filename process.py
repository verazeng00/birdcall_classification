# process.py
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np


def training(model, train_dl, valid_dl, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in tqdm(enumerate(train_dl), desc = f'Training {epoch}/{num_epochs}'):

            # Get the input features and target labels, and put them on the GPU
            idx, inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs['logits'], labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs['logits'],1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    #         if i % 10 == 0:    # print every 10 mini-batches
    #             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                # Run inference on trained model with the validation set


        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        if epoch % 10 == 0:
            valid(model, valid_dl)

        print('Finished Training')


def valid(model, valid_dl):
    correct_prediction = 0
    total_prediction = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Disable gradient updates
    with torch.no_grad():
        for data in tqdm(valid_dl, desc = 'Valid'):
            # Get the input features and target labels, and put them on the GPU
            idx, inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)


            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs['logits'],1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction/total_prediction
    print(f'Validation Accuracy: {acc:.2f}, Total items: {total_prediction}')



def prediction(model, test_dl, INV_BIRD_CODE, threshold = 0.5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    prediction_dict = {}
    all_events = set()
    for i, data in tqdm(enumerate(test_dl), desc = 'Test'):  # test_dl has only one row_id
                    # Normalize the inputs

        image, row_id, site = data  # return image.shape = 6, 3, 313, 313
        # row_id = a set with shape  = (,<= 16)

        image = image.squeeze(0)
        if site in {"site_1", "site_2"}:
            image = image.to(device)
            with torch.no_grad():
                prediction = model(image)
                proba = prediction["multilabel_proba"].detach().cpu().numpy().reshape(-1)

            events = proba >= threshold
            labels = np.argwhere(events).reshape(-1).tolist()
            for label in labels:
                all_events.add(label)


        else:
            # to avoid prediction on large batch
            batch_size = 16
            whole_size = image.size(0)
            if whole_size % batch_size == 0:
                n_iter = whole_size // batch_size
            else:
                n_iter = whole_size // batch_size + 1


            for batch_i in range(n_iter):
                batch = image[batch_i * batch_size:(batch_i + 1) * batch_size]
                if batch.ndim == 3:
                    batch = batch.unsqueeze(0)

                batch = batch.to(device)
                with torch.no_grad():
                    prediction = model(batch)
                    proba = prediction["multilabel_proba"].detach().cpu().numpy() # batch_size, num_classes

                events = proba >= threshold
                for i in range(events.shape[0]):
                    event = events[i, :]
                    labels = np.argwhere(event).reshape(-1).tolist()
                    for label in labels:
                        all_events.add(label)

    labels = list(all_events)

    if len(labels) == 0:
        label_string = "nocall"
    else:
        labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], labels))
        label_string = " ".join(labels_str_list)
    prediction_dict[row_id[0]] = label_string
    # print('prediction_dict', prediction_dict)
    return prediction_dict
