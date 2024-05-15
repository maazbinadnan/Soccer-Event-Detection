import torch
import os
from torchvision.transforms import transforms
from PIL import Image
import cv2
import pandas as pd

df = pd.DataFrame(columns=['Frame No','timestamp','Classifier1','Classifier2'])
class_to_idx_mapping = {'Center': 0, 'Left': 1, 'Right': 2,'Free-Kick':3,'Penalty': 4, 'Tackle': 5,'To Substitue' : 6,'Cards':7,'Corner': 8}
class_to_idx_mapping_class1 = {'Event': 0, 'Soccer': 1}
path = os.getcwd()
path = os.path.join(path, 'sport-analysis/inference/event-detection/Event_Detection_Final')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from EfficientNet import EfficientNetB0ForCustomTask

def map_class_index(output_index, class_to_idx_mapping):
    for class_name, idx in class_to_idx_mapping.items():
        if idx == output_index:
            return class_name
    return None

def local_classifier1(num_classes=2):
    model = EfficientNetB0ForCustomTask(num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(path,'layer1_final.pth'),map_location=device))
    model.eval()

    return model

def local_classifier2(num_classes=9):
    model = EfficientNetB0ForCustomTask(num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(path,'layer2.pth'),map_location=device))
    model.eval()

    return model

def video_classifier(video_path):
    # Get the video
    video = cv2.VideoCapture(video_path)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frameduration = 1/fps
    framecount = 0
    # Initialize classifiers
    classifier1_model = local_classifier1()
    classifier2_model = local_classifier2()

    # Transformations definition
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Define output video writer
    output_path = os.path.join(path, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read the video and classify each frame
    for i in range(int(frames)):
        ret, frame = video.read()
        if not ret:
            break
        else:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(image).unsqueeze(0).to(device)
            timestamp = framecount * frameduration
            framecount += 1
            with torch.no_grad():
                output1 = classifier1_model(image_tensor)
                _, predicted1 = torch.max(output1, 1)
                if predicted1.item() == 1:
                    continue
                else:
                    _, predicted2 = torch.max(classifier2_model(image_tensor), 1)
                    # Draw classifier outputs on the frame
                    predicted_class1 = map_class_index(predicted1.item(),class_to_idx_mapping_class1)
                    predicted_class2 = map_class_index(predicted2.item(), class_to_idx_mapping)
                    cv2.putText(frame, f"Classifier1: {predicted_class1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Classifier2: {predicted_class2}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Write frame with text to the output video
                    output_video.write(frame)
                    # Show frame in a window
                    cv2.imshow('Frame', frame)
                    # Store results in DataFrame
                    df.loc[i] = [i, timestamp, predicted1.item(), predicted2.item()]
                    # Wait for key press
                    cv2.waitKey(1)
    
    # Close video window
    cv2.destroyAllWindows()

    # Save results to CSV
    df.to_csv(os.path.join(path,'output.csv'), index=False)

    # Release output video writer
    output_video.release()

video_classifier(os.path.join(path,'hattrick.mp4'))