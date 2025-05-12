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
    
    `df = pd.read_csv('/kaggle/input/abo-dataset-small/metadata.csv')`
    
  - The dataset has over 147,000 listings, so only a representative subset is used for initial experimentation.
  - Pandas is used for efficient filtering, sampling, and data transformation.
  - The dataset includes multilingual metadata, but this implementation primarily focuses on English-based metadata.

**Sampling a subset**
  - To reduce processing overhead and allow fast iteration, a small number of listings is sampled from the full dataset.
    
    `sampled_df = df.sample(n=500, random_state=42)`
    
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
    
    `qa_data = sampled_df[['image_path', 'title', 'category']].to_dict(orient='records')
  `
  
  -  Only essential fields (image_path, title, category) are retained.
  - JSON-style records are useful for feeding into APIs like Google Gemini 2.0 or local LLMs via Ollama.
  - This structure also supports automated question generation loops, where one prompt can be generated per item in qa_data.

**Saving for Prompt-Based QA Generation**
  - The processed data is saved or prepared for use in the next phase — generation of VQA pairs using AI models.
    
    `import json
  with open('sampled_qa_data.json', 'w') as f:
      json.dump(qa_data, f, indent=4)`
  
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
