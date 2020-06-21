import torch


def train_model(model, train_loader, val_loader, criterion, optimizer, device, max_epoch, summary_writer=None):
    model.to(device)
    for epoch in range(1, max_epoch + 1):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        count = 0
        for step, sample in enumerate(train_loader, 1):
            # get the inputs; data is a list of [inputs, labels]
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image)
            loss = criterion(outputs, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            logits = torch.max(outputs, dim=1)[1]
            if label.dim() != 1:
                label = torch.max(label, dim=1)[1]
            count += outputs.size(0)
            running_accuracy += (logits == label).sum().item()

        # print statistics
        print('TRAIN - Epoch {} loss: {:.4f} accuracy: {:.4f}'.format(epoch,
                                                                      running_loss / step, running_accuracy / count))
        if summary_writer is not None:
            summary_writer.add_scalar('train', running_accuracy / count, epoch)

        running_accuracy = 0.0
        count = 0
        model.eval()
        for step, sample in enumerate(val_loader, 1):
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            with torch.no_grad():
                outputs = model(image)

                logits = torch.max(outputs, dim=1)[1]
                count += outputs.size(0)
                running_accuracy += (logits == label).sum().item()

        print('VAL - Epoch {}  accuracy: {:.4f}'.format(epoch,
                                                        running_accuracy / count))

        if summary_writer is not None:
            summary_writer.add_scalar(
                'val', running_accuracy / count, epoch)

    print('Finished Training')
