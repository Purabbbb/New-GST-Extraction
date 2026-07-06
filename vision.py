from pathlib import Path
import json
import base64
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

INPUT_DIR = Path("dataset_new")
OUTPUT_DIR = Path("ground_truth")

OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT = """
You are an expert invoice parser.

Extract the following fields.

Return ONLY valid JSON.

{
  "invoice_number":"",
  "invoice_date":"",
  "vendor_name":"",
  "buyer_name":"",
  "gstin":"",
  "pan":"",
  "cgst":"",
  "sgst":"",
  "igst":"",
  "subtotal":"",
  "total_amount":"",
  "po_number":"",
  "invoice_type":""
}

Rules:
- Do not hallucinate.
- If a field is missing, return "".
- Preserve formatting exactly.
- Return JSON only.
"""

for image_path in tqdm(sorted(INPUT_DIR.glob("*.png"))):

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}"
                    }
                ]
            }
        ]
    )

    text = response.output_text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {"raw_response": text}

    out_file = OUTPUT_DIR / f"{image_path.stem}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)