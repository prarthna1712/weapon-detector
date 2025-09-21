
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load existing training curve images
acc_img = mpimg.imread("results/train_accuracy.png")
loss_img = mpimg.imread("results/train_loss.png")

# Create figure with 2 rows (stacked)
fig, axs = plt.subplots(2, 1, figsize=(6,10))

axs[0].imshow(acc_img)
axs[0].axis('off')
axs[0].set_title("Train vs Val Accuracy")

axs[1].imshow(loss_img)
axs[1].axis('off')
axs[1].set_title("Train vs Val Loss")

plt.tight_layout()
plt.savefig("results/training_plot.png")
plt.show()

print("✅ Merged training plot saved as results/training_plot.png")
