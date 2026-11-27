import json
import os
from sklearn.datasets import fetch_20newsgroups

def export_20newsgroups():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data", "Datasets", "20Newsgroups")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "20newsgroups_corpus.json")
    
    categories = [
        'comp.graphics',
        'rec.autos',
        'sci.space',
        'sci.med',
        'talk.politics.mideast'
    ]
    
    print(f"Fetching 20 Newsgroups subset for categories: {categories}...")
    dataset = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    docs = []
    for idx, (text, label_idx) in enumerate(zip(dataset.data, dataset.target)):
        cleaned_text = text.strip()
        if len(cleaned_text) > 100:  # Only include non-empty substantial documents
            docs.append({
                "id": idx + 1,
                "category": dataset.target_names[label_idx],
                "text": cleaned_text
            })
            
    print(f"Exported {len(docs)} documents to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": "20Newsgroups",
            "num_documents": len(docs),
            "categories": categories,
            "documents": docs
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    export_20newsgroups()
