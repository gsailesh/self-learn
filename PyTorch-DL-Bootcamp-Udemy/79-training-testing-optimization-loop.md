```py
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
torch.cuda.manual_seed(42)
```

```py
epochs = 100

# Assumes data is already loaded in X_train, y_train, X_test, y_test.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


for epoch in range(epochs):
    model.train()

    y_logits = model(X_train).squeeze() # Get rid of the extra dimension
    y_pred = torch.round(torch.sigmoid(y_logits)) # Turn logits into probabilities and then round to get predictions
    loss = loss_fn(y_logits, y_train) # Assumes loss_fn is defined, e.g., nn.BCEWithLogitsLoss()
    # loss = loss_fn(torch.sigmoid(y_logits), y_train) # if nn.BCELoss is used
    acc = accuracy_fn(y_pred, y_train) # Assumes accuracy_fn is defined

    optimizer.zero_grad() # Zero the gradients
    loss.backward() # Backpropagation
    optimizer.step() # Update the weights

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | "
              f"Train Loss: {loss:.4f} | "
              f"Train Acc: {acc:.2f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}")
```