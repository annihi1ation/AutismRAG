import json
import pandas as pd
import sys
import os

def json_to_sheet(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert dictionary to a list of tuples for the DataFrame
        # Assuming the JSON is a flat dictionary
        df = pd.DataFrame(list(data.items()), columns=['Field', 'Value'])
        
        # Print as Markdown
        print(df.to_markdown(index=False))
        
        # Also save to a CSV file in the same directory for convenience
        csv_path = os.path.splitext(json_path)[0] + '.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nAlso saved to: {csv_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to the file mentioned in the prompt
        file_path = "/data2/leyizhao/CommTool/data/Alarifi-2024-Interventions addressing challeng.json"
    
    json_to_sheet(file_path)
