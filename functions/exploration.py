import matplotlib.pyplot as plt
import numpy as np

def print_distribution(title, class_names, dataset):
  class_counts = [0] * len(class_names)
  for _, labels in dataset:
    for label in labels.numpy():
      class_counts[label] += 1
  plt.figure(figsize=(10, 6))
  plt.bar(class_names, class_counts)
  plt.xlabel('Class')
  plt.ylabel('Count')
  plt.title(title)
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
  plt.show()

def sneak_peek(dataset, class_names):
  plt.figure(figsize=(8, 8))
  for images, labels in dataset.take(1):
    for i in range(25):
      ax = plt.subplot(5, 5, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")