import torch
import matplotlib.pyplot as plt
import numpy as np

# Load model weights
model.load_state_dict(torch.load('best_cnn_model.pth'))
model.eval()

# Get a batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Make predictions
with torch.no_grad():
    outputs = model(images.to(device))
    probabilities = torch.softmax(outputs, dim=1)
    confs, preds = torch.max(probabilities, 1)

# Plot a 3x3 grid of sample results
fig = plt.figure(figsize=(12, 12))
class_names = test_dataset.classes

for i in range(min(9, len(images))):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    
    # Denormalize image for display
    img = images[i].cpu().numpy().transpose((1, 2, 0))
    img = np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    plt.imshow(img)
    
    pred_label = class_names[preds[i]]
    true_label = class_names[labels[i]]
    confidence = confs[i].item() * 100
    
    color = 'green' if preds[i] == labels[i] else 'red'
    ax.set_title(f"Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label}", color=color, fontsize=11)

plt.tight_layout()
plt.savefig('cnn_sample_predictions.png', dpi=150)
plt.show()
