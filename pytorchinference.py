import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# Configure the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Load the PyTorch model."""
    model = torch.load(model_path, map_location=device)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path, input_size):
    """Preprocess the image for inference."""
    transform = transforms.Compose([
        transforms.Resize(input_size),          # Resize the image
        transforms.ToTensor(),                 # Convert to a tensor
        transforms.Normalize(                  # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
    return transform(image).unsqueeze(0)          # Add batch dimension

def postprocess(outputs, class_names, conf_threshold=0.5):
    """Postprocess the outputs."""
    confidences, predictions = torch.max(outputs, dim=1)
    results = []
    for confidence, prediction in zip(confidences, predictions):
        if confidence > conf_threshold:
            results.append({
                "class": class_names[prediction.item()],
                "confidence": confidence.item()
            })
    return results

def draw_detections(image, detections):
    """Draw predictions on the image."""
    for det in detections:
        label = f"{det['class']}: {det['confidence']:.2f}"
        # Display the label
        cv2.putText(image, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

def main(model_path, image_path, class_names, input_size=(224, 224)):
    """Run inference."""
    # Load the model
    model = load_model(model_path)

    # Preprocess the image
    input_tensor = preprocess_image(image_path, input_size).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Postprocess the results
    detections = postprocess(outputs, class_names)

    # Load and annotate the image
    original_image = cv2.imread(image_path)
    annotated_image = draw_detections(original_image, detections)

    # Display the annotated image
    cv2.imshow("Inference", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    # Define argument parser
    parser = argparse.ArgumentParser(description="PyTorch Model Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the PyTorch model (.pth)")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--classes", type=str, required=True, help="Path to class names file")
    parser.add_argument("--size", type=int, default=224, help="Input size (default: 224)")
    args = parser.parse_args()

    # Load class names
    with open(args.classes, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Run the main function
    main(args.model, args.image, class_names, (args.size, args.size))
