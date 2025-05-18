import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Example: Using transformers pipeline for VQA (replace with your model as needed)
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel
import torch
import os
import gdown
import zipfile

# Google Drive file ID
file_id = "16W-MdgTy9i8apeEAQfwBegpwwVAREiGG"
output_path = "blip-lora.zip"

# Download and extract
if not os.path.exists("blip-lora"):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

    # Extract
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall("blip-lora")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()
    # image_dir = "/home/snigdha/inference-setup/inference-setup/data"
    # Load metadata CSV
    # df = pd.read_csv("/home/snigdha/inference-setup/inference-setup/data/metadata.csv")
    df = pd.read_csv(args.csv_path)
    # Load model and processor, move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

    # Load base model
    base_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    # Load downloaded LoRA adapter (adjust path after unzip)
    lora_path = "blip-lora/16_16_no_dense"  # or extracted folder path
    model = PeftModel.from_pretrained(base_model, lora_path).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**encoding)
                # logits = outputs.logits
                # predicted_idx = logits.argmax(-1).item()
                # answer = model.config.id2label[predicted_idx]
                answer = processor.decode(outputs[0], skip_special_tokens=True)
                print(answer)
        except Exception as e:
            answer = "error"
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).strip().lower()
        generated_answers.append(answer)
        print(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()