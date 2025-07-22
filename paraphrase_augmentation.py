import torch
import pandas as pd
import ast
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Load paraphrasing model (T5-based) ===
model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# === Set device and move model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def paraphrase(caption, tokenizer, model, device):
    text = "paraphrase: " + caption + " </s>"
    encoding = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=256
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=60,
        do_sample=True,
        top_k=90,               
        top_p=0.90,
        temperature=1.2,         
        no_repeat_ngram_size=2
    )
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased

# Change this to your input CSV file path
input_csv = "/Users/tanchunye/personal/OMSCS/CS7643-DL/final_project/flickr_annotations_30k.csv"
df = pd.read_csv(input_csv)

# === Process each row ===
augmented_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        original_captions = ast.literal_eval(row['raw'])  # List of 5 captions
    except:
        print(f"⚠️ Error parsing row {idx}, skipping.")
        continue

    paraphrased_captions = []
    for caption in original_captions:
        try:
            new_caption = paraphrase(caption, tokenizer, model, device)
        except Exception as e:
            print(f"⚠️ Error paraphrasing row {idx}: {e}")
            new_caption = ""
        paraphrased_captions.append(new_caption)

    combined_captions = original_captions + paraphrased_captions

    # Add to new dataset row
    new_row = row.copy()
    new_row['raw'] = str(combined_captions)
    augmented_rows.append(new_row)

# Change this to your output CSV file path
output_csv = "/Users/tanchunye/personal/OMSCS/CS7643-DL/final_project/flickr_annotations_30k_augmented.csv"
pd.DataFrame(augmented_rows).to_csv(output_csv, index=False)

print("✅ Augmented CSV saved:", output_csv)
