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
- our final curated dataset contains of 1,59,176 rows and 3 columns which have the image path, question and answer. 

## Baseline model

We experimented with several popular vision-language models including VILT, CLIP, BLIP-2, and GIT, but each had limitations that made them less suitable for our VQA task:
  - VILT: Although lightweight, VILT produced low accuracy and often generated full sentences rather than short or single-word answers.
  - CLIP: Required a predefined dictionary of words and computed probabilities over this set, resulting in significant overhead and limiting flexibility in open-ended answer generation.
  - BLIP-2: Needed manually written prompts to perform well and was much slower during inference. It also tended to produce longer sentence-based answers.
  - GIT: Failed to perform on our dataset and produced zero accuracy.
Due to these constraints, we selected BLIP (Salesforce BLIP-VQA) as our base model. It supports direct image-question answering without excessive prompt engineering or reliance on constrained answer sets, and it performs reliably with short, relevant answers—making it a strong fit for visual question answering.


**Model used**: BLIP
**Source**: Hugging Face – Salesforce/blip-vqa-base
**Setup**: 
  - Model and processor are downloaded from Hugging Face.
  - Loaded to GPU (cuda) for faster inference.
**Pipeline**:  We have taken 80 : 20 split for training and testing the data. Each image from the dataset is loaded and paired with its corresponding question.The image and question are preprocessed via the BLIP processor and fed to the model for answer generation. The model receives multimodal input (image + text question). The generated output is a free-form text answer(a short phrase or word).


**Evaluation metrics for baseline model**:
We are creating a CSV file of model predictions along with the ground truths (`predictions_results.csv`)  and filtering out the bad/invalid entries for further evalutaion. We are also normalizing text (turning them all into lower case) for better analysis and we have calculated different metrics and below are the results for the same.

```
Accuracy      : 0.3846
BERTScore F1  : 0.9764
ROUGE-L       : 0.3935
Token F1      : 0.3871
BLEU          : 0.0690
```

We have also trained the baseline model with a part of the dataset (frac = 0.3) and 4 epochs and here the metrics for it.

```
Accuracy      : 0.1300
BERTScore F1  : 0.9736
ROUGE-L       : 0.1300
Token F1      : 0.1300
BLEU          : 0.0231
```

We have also done training using VILT and here are the metrics for it 

```
Accuracy      : 0.2967
BERTScore F1  : 0.9768
ROUGE-L       : 0.2995
Token F1      : 0.2975
BLEU          : 0.0529
```

As BLIP had better accuracy we have proceeded with it.

## Fine tuning the VQA model using LoRA (Low rank adaptation)
We arefine-tuning the BLIP (Bootstrapping Language-Image Pre-training) model for Visual Question Answering (VQA) using the LoRA (Low-Rank Adaptation) technique to efficiently adapt a pre-trained transformer model. Below is a detailed explanation of the code implementation.

1. **Model and Processor Initialization:**
   ```
   from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor
    )
    
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
   ```
   
   - The above code loads the pre-trained BLIP model and its corresponding processor, which is responsible for preparing both text and image inputs for the model.
  
2. **LoRA Configuration and PEFT Integration:**
   ```
   lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value", "projection", "fc1", "fc2", "key", "qkv"],
    lora_dropout=0.1,
    bias="none",
    )
    ```
   
   - Sets the LoRA configuration parameters for fine-tuning the attention and feedforward layers of the model.
     
   ```
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
   ```
   
   - Prepares the model for parameter-efficient fine-tuning and prints the number of trainable parameters.
     
     
3. **Dataset Preparation:**

   ```
   class CSVVQADataset(torch.utils.data.Dataset):
    def _init_(self, dataframe, processor):
        self.data = dataframe
        self.processor = processor

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        item = self.data.iloc[idx]
        image = Image.open(item['image_path']).convert("RGB")
        text = item['question']
        answer = item['answer']
        ...
   ```
   
   - Defines a custom PyTorch dataset that reads from a CSV file.
   - Processes each item by loading the image, question, and answer, then applies the BLIP processor to create inputs and labels.
     
5. **Train-Validation Split:**

   ```
   train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
   ```

   - Splits the data into 80% training and 20% validation.
     
7. **DataLoader Creation:**

   ```
    train_dataset = CSVVQADataset(train_df, processor)
    valid_dataset = CSVVQADataset(valid_df, processor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=3, shuffle=False, pin_memory=True)
   ```

   - Creates DataLoader objects for batch-wise processing during training and validation.
     
9. **Training Preparation:**

   ```
   optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
   scaler = torch.cuda.amp.GradScaler()
   ```

   - Uses the AdamW optimizer and mixed-precision training with automatic gradient scaling.

   ```
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```

   - Detects and sets the training device (GPU if available).
     
11. **Training and Validation Loop:**
   
   ```
   for epoch in range(4):
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
   ```
   
   - Trains the model over multiple epochs using mixed-precision to speed up training.
   
   ```
       model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch+1} Validation"):
            for k in batch:
                batch[k] = batch[k].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            eval_loss += loss.item()
   ```
   
   - Evaluates the model on validation data in inference mode.
   
   ```
    train_loss /= len(train_dataloader)
    eval_loss /= len(valid_dataloader)
    tracking_information.append((train_loss, eval_loss))
   ```
   
   - Averages the loss over all batches.
     
11. **Model Checkpointing and Early Stopping:**
   
   ```
   if eval_loss < min_eval_loss:
    min_eval_loss = eval_loss
    model.save_pretrained("Model/blip-lora-saved")
    model.save_pretrained("/kaggle/working/blip-lora-saved")
    early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook >= patience:
            break
   ```
   
   - Saves the best model if validation loss improves.
   - Implements early stopping to prevent overfitting.
     
11. **Saving Training History:**

    ```
    pickle.dump(tracking_information, open("tracking_info.pkl", "wb"))
    ```

    - Saves training and validation loss history for further analysis.

We experimented with multiple LoRA configurations varying in rank, alpha values, and module inclusion to study trade-offs between efficiency and accuracy. 
Here is the basic template of the LoRA finetuning.

```
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

lora_config = LoraConfig(
    # use_rslora = True,
    init_lora_weights = "pissa",
    r=16,
    lora_alpha=16,
    target_modules=["query", "value", "projection", "fc1", "fc2", "key", "qkv"],
    lora_dropout=0.1,
    bias="none",
)
```

Here are the different configurations we have tried. C1 to C7 have been trained on a part of the dataset with more number of epochs.
  - C1 = rank = 8, LoRA alpha = 8 without dense layer, Frac = 0.3, epochs = 4
  - C2 = rank = 8, LoRA alpha = 16 with dense layer, Frac = 0.3, epochs = 4
  - C3 = rank = 16, LoRA alpha = 16 without dense layer, Frac = 0.3, epochs = 4
  - C4 = rank = 32, LoRA alpha = 32 using DoRA with dense layer, Frac = 0.3, epochs = 4
  - C5 = rank = 32, LoRA alpha = 32 without dense layer, Frac = 0.3, epochs = 4
  - C6 = rank = 16, LoRA alpha = 16 using RSLoRA without dense layer , Frac = 0.3, epochs = 2
  - C7 = rank = 16, LoRA alpha = 16 using PISSA without dense layer , Frac = 0.4, epochs = 2
  - C8 = rank = 16, LoRA alpha = 16, without dense layer on full dataset , Frac = 1, epochs = 1

**Impact of LoRA rank and alpha values**: 
Higher values of rank and alpha lead to more capacity for adaptation but also higher compute/memory cost. Increasing the rank and alpha from 8→16 (C1 → C3) without using dense layers led to performance improvement i.e., accuracy went from 67% to 71%. Lower values of rank and alpha are more efficient but may underfit in low-resource settings. Mid-point like C3 offers a balance. Too high (r=32) might over-parameterize or overfit, especially with poor adapter design or training.

**Impact of including dense layer**: 
Only rank = 8 and alpha = 16 explicitly includes the dense module in target_modules. This configuration allows LoRA to adapt more of the feed-forward network, potentially improving generalization on visual-language tasks. Including dense improved performance slightly in early epochs, especially for fine-grained questions (e.g., identifying object attributes). Including the dense layer (as in C2 and C4) degraded performance significantly, possibly due to overfitting.

**DoRA (Decoposed Rank Adaptation)**: 
DoRA is intended to improve adaptation by separately modeling direction and scale. DoRA did not improve performance compared to non-DoRA settings at the same rank (C4 vs C5). Both performed equally poorly. This suggests that DoRA may not be beneficial for the BLIP model.

**RSLoRA (Rank-Splitting LoRA)**:
RSLoRA is intended to introduce sparsity or decomposition in LoRA layers to reduce redundancy and improve efficiency. This offers decent performance with improved efficiency, making it suitable for resource-constrained deployments. However, it did not surpass standard LoRA in effectiveness, suggesting limited gain over simpler setups for this specific task.

**PISSA (Plug-and-Play Improved Spatially-Sparse Adapter)**:
This is a more advanced LoRA variant aiming for better spatial selectivity and sparsity in adaptation. Although designed for general efficiency, PISSA underperformed severely in this case. May require task-specific tuning or architectural modifications to be effective.

**Fine-tuning on VILT**: 
We have also tried fien-tuning on VILT but it had high complexity, high resource utilization and took more time to train so we did not continue implementing it.

## Evaluation Metrics for Fine-tuned BLIP Configurations

| Config Name            | LoRA Rank (r) | Alpha (α) | Accuracy | BERTScore-F1  | ROUGE-L  | Token F1 | BLEU   |
|------------------------|---------------|-----------|----------|---------------|----------|----------|--------|
| 8_8_no_dense (C1)      | 8             | 8         | 0.6700   | 0.9859        | 0.6800   | 0.6700   | 0.1191 |
| 8_16_dense (C2)        | 8             | 16        | 0.3300   | 0.9771        | 0.3350   | 0.3350   | 0.0598 |
| 16_16_no_dense (C3)	   | 16            | 16	       | 0.7225	  | 0.9880        |	0.7323	 | 0.7227	  | 0.1285 |
| 32_32_dora (C4)        | 32            | 32        | 0.3300   | 0.9771        | 0.3350   | 0.3350   | 0.0598 |
| 32_32_no_dense (C5)    | 32            | 32        | 0.3300   | 0.9771        | 0.3350   | 0.3350   | 0.0598 |
| 16_16_rslora (C6)        | 16           | 16        | 0.6700   | 0.9817        | 0.6800   | 0.6700   | 0.1191 |
| 16_16_pissa (C7)        | 16           | 16        | 0.3300   | 0.9771        | 0.3350   | 0.3350   | 0.0598 |
| 16_16_no_dense with full dataset(C8)    | 16            | 16        | 0.6155  | 0.9842       | 0.6247   | 0.6155   | 0.1095 |

- 16_16_no_dense(C3) achieved the highest scores across all metrics.
- Configurations with r=32 and α=32 (both with and without DoRA or dense modules) performed poorly indicating overfitting. This Ssggests that higher rank values do not guarantee better performance and may lead to degradation.
- The 8_8_no_dense model showed decent performance (67% accuracy), making it a lightweight yet effective choice. However, it was consistently outperformed by the 16_16_no_dense model, showing that moderate rank (r=16) is a better tradeoff between model size and accuracy.

- For the 16_16_no_dense model we have made inference on an online dataset after training with the 2 different cases i.e., a) full dataset and 1 epoch and b) a part of dataset (frac = 0.3) with more epochs
    - **Link to the dataset:** https://www.kaggle.com/datasets/henrychibueze/vqa-dataset
    - Here are the metrics for both the cases
      
      | Case                             | Accuracy | BERTScore-F1  | ROUGE-L  | Token F1 | BLEU   |
      |----------------------------------|----------|---------------|----------|----------|--------|
      | Full dataset and 2 epoch         | 0.8591   | 0.9994 | 0.8591 | 0.8591 | 0.1528 |
      | Part of the dataset (frac = 0.3) and 4 epochs | 0.8650 | 0.9994 | 0.8650 | 0.8650 | 0.1538 |

      
- For the last selected model i.e., **16_16_no_dense with a part of dataset (frac = 0.3) and more number of epochs (4 epochs)**, the total number of parameters are 390,570,812 and trainable parameters are 5,898,240 (trainable%: 1.5102)

## Challenges encountered:

The dataset was very large to train upon and even though we have used accelerator, it was not quite useful. It took minimum of 15-20 hrs for training on the full dataset we have generated.

We also tried ViLT fine-tuning using LoRA but the architecture itself was too complex and GPU accelerators available weren't enough for running on a limited dataset too 

## Dependencies

- PyTorch for model training and inference.
- Transformers (by Hugging Face) for using and fine-tuning the BLIP and other vision-language models.
- PEFT (Parameter-Efficient Fine-Tuning) for implementing LoRA, DoRA, and related techniques.
- BERTScore, ROUGE, BLEU for evaluation metrics.
- Hugging Face Hub utilities for model and tokenizer access.
- CUDA 12 and NVIDIA libraries for GPU acceleration and compatibility.
- Common Python packages such as NumPy, Pandas, Matplotlib, and TQDM for data processing, visualization, and progress tracking.
- Bits and Bytes module for running on cuda and optimisation.
- All dependencies and exact versions are included in the `requirements.txt` file

## Links for all models
https://www.kaggle.com/models/sasisnigdhayadavalli/blip-8-8-no-dense/Transformers/default/1

https://www.kaggle.com/models/sasisnigdhayadavalli/blip-8-16-qkv/Transformers/default/1

https://www.kaggle.com/models/sasisnigdhayadavalli/blip-16-16-no-dense/Transformers/default/1 (selected model for inference on hidden dataset)

https://www.kaggle.com/models/sasisnigdhayadavalli/8-16-no-dense/Transformers/default/1

https://www.kaggle.com/models/sasisnigdhayadavalli/8-16-no-dense/Transformers/default/1

https://www.kaggle.com/models/sasisnigdhayadavalli/32-32-no-dense/Transformers/default/1

https://www.kaggle.com/models/sasisnigdhayadavalli/32-32-dora-all-target-modules/Transformers/default/1

https://www.kaggle.com/models/sasisnigdhayadavalli/16-16-rslora-no-dense-epoch2/Transformers/default/1

https://www.kaggle.com/models/sasisnigdhayadavalli/16-16-rslora-no-dense/Transformers/default/1

https://www.kaggle.com/models/sasisnigdhayadavalli/16-16-pissa-no-dense-epoch2/Transformers/default/1

https://www.kaggle.com/models/sasisnigdhayadavalli/16-16-full-no-dense/Transformers/default/1
