import json
import os

def extract_fix_code(json_path, output_dir="fixed_code"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # print(data)

    # Handle both dict with list under a key or raw list
    items = data["Pytorch_to_JAX_Examples"] \
        # if isinstance(data, list) else data.get("data", [])

    for item in items:
        example_id = item.get("Example_id")
        fix_code = item.get("LLM_fix_output")
        print(example_id)

        if example_id is not None and fix_code is not None:
            file_path = os.path.join(output_dir, f"{example_id}.py")
            print(file_path)
            with open(file_path, "w", encoding="utf-8") as code_file:
                code_file.write(fix_code)

    print(f"Extracted {len(items)} fix_code snippets to '{output_dir}'.")

# Example usage
extract_fix_code("pytorch_to_jax_code_rules_20_cases.json")
