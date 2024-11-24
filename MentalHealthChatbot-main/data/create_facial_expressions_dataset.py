import os
import pandas as pd

# Directory where images are stored
image_dir = 'data/processed/images'

# Get list of all images and their labels
data = {'image_path': [], 'label': []}
for expression in os.listdir(image_dir):
    expression_dir = os.path.join(image_dir, expression)
    for img_name in os.listdir(expression_dir):
        img_path = os.path.join(expression_dir, img_name)
        data['image_path'].append(img_path)
        data['label'].append(expression)

# Create a DataFrame
df = pd.DataFrame(data)

# Create directories if they don't exist
processed_data_dir = 'data/processed'
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

# Save the DataFrame to a CSV file
csv_path = os.path.join(processed_data_dir, 'facial_expressions.csv')
df.to_csv(csv_path, index=False)

print(f'Dataset created and saved to {csv_path}.')
