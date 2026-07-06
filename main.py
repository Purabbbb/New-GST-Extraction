from header_extractor import extract_header
from table_extractor import extract_tables

IMAGE_PATH = "invoice_test.png"

print("=" * 80)
print("HEADER EXTRACTION")
print("=" * 80)

header = extract_header(IMAGE_PATH)

print("\nHeader Crop:")
print(header["header_crop_path"])

print("\nOCR File:")
print(header["ocr_file"])

print("\nOCR Text Preview:")
print(header["ocr_text"][:2000])

print("\n")
print("=" * 80)
print("TABLE EXTRACTION")
print("=" * 80)

tables = extract_tables(IMAGE_PATH)

print(f"\nTotal Tables Found: {len(tables)}")

for table in tables:

    print("\n" + "=" * 60)

    print(
        f"TABLE {table['table_number']}"
    )

    print(
        f"Crop: {table['crop_path']}"
    )

    print("\nDataFrame:")

    print(
        table["dataframe"]
    )