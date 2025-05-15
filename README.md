# Visual Question and answering 

## Team Members

- Nikita Kiran - IMT2022028
- Sasi Snigdha Yadavalli - IMT2022571
- Akshaya Bysani - IMT2022579

## Introduction
This project focuses on multimodal Visual Question Answering (VQA) using the Amazon Berkeley Objects (ABO) dataset. The goal is to create a curated VQA dataset where each question is answerable by analyzing product images from the ABO dataset. This involves generating diverse single-word answer questions using either API-based or on-device multimodal models. The subsequent stages include evaluating baseline VQA models, applying Low-Rank Adaptation (LoRA) for efficient fine-tuning, and benchmarking performance using standard and custom evaluation metrics.

## Dataset curation

The provided code (samplelistings.ipynb) serves as the foundation for preprocessing and preparing the data from the ABO dataset to facilitate automatic question generation. Here's a breakdown of its functionality:
**Loading metadata:**
  - The code reads the ABO metadata CSV file, which contains structured information about each product such as its Title, Brand, Category, Description, Product ID, Image links (relative paths).
    
    ```
    df = pd.read_csv('/kaggle/input/abo-dataset-small/metadata.csv')
    ```
    
  - The dataset has over 147,000 listings, so only a representative subset is used for initial experimentation.
  - Pandas is used for efficient filtering, sampling, and data transformation.
  - The dataset includes multilingual metadata, but this implementation primarily focuses on English-based metadata.

**Sampling a subset**
  - To reduce processing overhead and allow fast iteration, a small number of listings is sampled from the full dataset.
    
    ```
    sampled_df = df.sample(n=500, random_state=42)
    ```
    
  - Sampling is randomized but reproducible via the random_state.
  - The subset ensures category and product-type diversity to allow a rich set of VQA questions.
  - This subset allows for manageable processing when generating VQA pairs using models like Gemini 2.0 or local Ollama-based models.

**Path Mapping for Images**
  - Each product listing is linked with its corresponding image path by resolving relative file paths into full paths.
  - Each listing can have multiple images, but this code selects one primary image per product.

    ```
    def get_image_path(product_id, image_id):
      return f"/kaggle/input/abo-dataset-small/images256/{product_id}/{image_id}.jpg"
    
    sampled_df['image_path'] = sampled_df.apply(lambda x: get_image_path(x['product_id'], x['image_id']), axis=1)
    ```
  
  - This linkage is essential for multimodal prompt generation, ensuring the model sees the correct image.
  - File paths are organized hierarchically in the ABO dataset (images256/<product_id>/<image_id>.jpg).

**Creating a JSON-Style Structure for Prompting**
  - A simplified data format is created to be used with prompting APIs or models. It includes only the key visual and textual fields needed to form questions.
    
    ```
    qa_data = sampled_df[['image_path', 'title', 'category']].to_dict(orient='records')
    ```
  
  -  Only essential fields (image_path, title, category) are retained.
  - JSON-style records are useful for feeding into APIs like Google Gemini 2.0 or local LLMs via Ollama.
  - This structure also supports automated question generation loops, where one prompt can be generated per item in qa_data.

**Saving for Prompt-Based QA Generation**
  - The processed data is saved or prepared for use in the next phase — generation of VQA pairs using AI models.
    
    ```
    import json
    
    with open('sampled_qa_data.json', 'w') as f:
      json.dump(qa_data, f, indent=4)
    ```
  
  - Output format (.json) is ideal for giving into prompt-based systems.
  - JSON export allows for easy tracking, error detection, and reuse.

Now coming to the **VQA Question generation** part

We used the Google Gemini 1.5 Flash model via the official Python SDK (google.generativeai) to generate multimodal Visual Question Answering (VQA) data. Gemini 1.5 Flash was selected for its support for image + text input, high-speed inference, and capability to perform fine-grained visual reasoning required for this task.
**Prompt Template Definition**
  - A prompt template is given to guide the model into generating appropriate single-word answer questions. The prompt gives instructions and examples.
  - The prompt is worded to request visual grounding and single-word answers, adhering to project guidelines.
  - Including metadata like title and category provides context and improves question relevance.
  - This prompt encourages the model to focus on visually inferable attributes like color, type, texture, etc.

**Google Gemini API Integration**
  - A JSON file of product listings is loaded.
  - The main_image_id for each product is used to construct the full image path.
  - PIL is used to open the image, and the Gemini SDK uploads it for use in multimodal input.
  - A unique prompt is dynamically generated per product, inserting the specific title and category as context.
  - The image and text prompt are sent to gemini-1.5-flash using:
    ```
    response = model.generate_content([image, prompt])
    ```
  - The response is parsed using a custom extract_json() function.
  - Each product's output is stored in a dictionary with keys: image_id, and qa_data (list of Q&A dictionaries).


The final generated VQA dataset contains:
- `image_id:` ID of the image used
- `qa_data:` List of 4–6 dictionaries, each containing:
    - `question:` A single question about the image
    - `answer:` A one-word answer grounded in the image

Here is an eaxmple of the same 

```
{
  "19502": {
    "image_id": "abc123",
    "qa_data": [
      {
        "question": "What material is the visible sole made of?",
        "answer": "Rubber"
      },
      {
        "question": "What color are the laces?",
        "answer": "White"
      }
    ]
  }
}

```
## Baseline model
**Model used**: BLIP-2 (Salesforce BLIP-2 Pretrained on Coco)
**Source**: Hugging Face – Salesforce/blip2-opt-2.7b
**Setup**: 
  - Model and processor are downloaded from Hugging Face.
  - Loaded to GPU (cuda) for faster inference.
**Pipeline**:  We have taken 80 : 20 split for training and testing the data. Each image from the dataset is loaded and paired with its corresponding question.The image and question are preprocessed via the BLIP-2 processor and fed to the model for answer generation. The model receives multimodal input (image + text question). The generated output is a free-form text answer(a short phrase or word).

**Evaluation metrics for baseline model**:
We are creating a CSV file of model predictions along with the ground truths (`predictions_results.csv`)  and filtering out the bad/invalid entries for further evalutaion. We are also normalizing text (turning them all into lower case) for better analysis and we have calculated different metrics and below are the results for the same.

```
Accuracy      : 0.3846
BERTScore F1  : 0.9764
ROUGE-L       : 0.3935
Token F1      : 0.3871
BLEU          : 0.0690
```
## Fine tuning the VQA model using LoRA (Low rank adaptation)
We experimented with multiple LoRA configurations varying in rank, alpha values, and module inclusion to study trade-offs between efficiency and accuracy. Here are the different configurations we have tried 
  - C1 = rank = 8, LoRA alpha = 8 without dense layer
  - C2 = rank = 8, LoRA alpha = 16 with dense layer
  - C3 = rank = 16, LoRA alpha = 16 without dense layer
  - C4 = rank = 32, LoRA alpha = 32 using DoRA with dense layer
  - C5 = rank = 32, LoRA alpha = 32 without dense layer

**Impact of LoRA rank and alpha values**: 
Higher values of rank and alpha lead to more capacity for adaptation but also higher compute/memory cost. Increasing the rank and alpha from 8→16 (C1 → C3) without using dense layers led to performance improvement i.e., accuracy went from 67% to 71%. Lower values of rank and alpha are more efficient but may underfit in low-resource settings. Mid-point like C3 offers a balance. Too high (r=32) might over-parameterize or overfit, especially with poor adapter design or training.

**Impact of including dense layer**: 
Only rank = 8 and alpha = 16 explicitly includes the dense module in target_modules. This configuration allows LoRA to adapt more of the feed-forward network, potentially improving generalization on visual-language tasks. Including dense improved performance slightly in early epochs, especially for fine-grained questions (e.g., identifying object attributes). Including the dense layer (as in C2 and C4) degraded performance significantly, possibly due to overfitting.

**DoRA (Decoposed Rank Adaptation)**: 
DoRA is intended to improve adaptation by separately modeling direction and scale. DoRA did not improve performance compared to non-DoRA settings at the same rank (C4 vs C5). Both performed equally poorly. This suggests that DoRA may not be beneficial for the BLIP model.

## Evaluation Metrics for Fine-tuned BLIP Configurations

| Config Name            | LoRA Rank (r) | Alpha (α) | Accuracy | BERTScore-F1  | ROUGE-L  | Token F1 | BLEU   |
|------------------------|---------------|-----------|----------|---------------|----------|----------|--------|
| 8_8_no_dense (C1)      | 8             | 8         | 0.6700   | 0.9859        | 0.6800   | 0.6700   | 0.1191 |
| 8_16_dense (C2)        | 8             | 16        | 0.3300   | 0.9771        | 0.3350   | 0.3350   | 0.0598 |
| 16_16_no_dense (C3)    | 16            | 16        | 0.7100   | 0.9896        | 0.7200   | 0.7100   | 0.1263 |
| 32_32_dora (C4)        | 32            | 32        | 0.3300   | 0.9771        | 0.3350   | 0.3350   | 0.0598 |
| 32_32_no_dense (C5)    | 32            | 32        | 0.3300   | 0.9771        | 0.3350   | 0.3350   | 0.0598 |

- 16_16_no_dense(C3) achieved the highest scores across all metrics.
- Configurations with r=32 and α=32 (both with and without DoRA or dense modules) performed poorly indicating overfitting. This Ssggests that higher rank values do not guarantee better performance and may lead to degradation.
- The 8_8_no_dense model showed decent performance (67% accuracy), making it a lightweight yet effective choice. However, it was consistently outperformed by the 16_16_no_dense model, showing that moderate rank (r=16) is a better tradeoff between model size and accuracy.
