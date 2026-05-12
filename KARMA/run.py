import glob
import os
from karma import KARMAPipeline
from karma.config import create_default_config

# Create configuration
config = create_default_config(api_key=os.environ.get("OPENROUTER_API_KEY", ""))
config.model.name = "google/gemini-3-flash-preview"

# Initialize pipeline
pipeline = KARMAPipeline.from_config(config)

# Process all documents in testbase/
testbase_dir = "/data2/leyizhao/CommTool/testbase"
documents = sorted(glob.glob(os.path.join(testbase_dir, "*.pdf")))
print(f"Found {len(documents)} documents in {testbase_dir}")

results = pipeline.process_batch(documents, output_dir="output")

# Analyze results
total_triples = sum(len(result.integrated_triples) for result in results)
print(f"Extracted {total_triples} total knowledge triples from {len(documents)} documents")

# Access results (example for the first document)
if results:
    print(f"\nExample triples from first document:")
    for triple in results[0].integrated_triples[:5]:
        print(f"{triple.head} --[{triple.relation}]--> {triple.tail} (confidence: {triple.confidence:.2f})")

print("\nPer-document knowledge graphs saved to output/local_kg and summaries to output/summaries.")
