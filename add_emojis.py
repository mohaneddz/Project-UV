import json
import re

def add_emojis(text):
    # Add emojis to conclusion lines
    text = re.sub(r'    REJECT H₀', '   ✅ REJECT H₀', text)
    text = re.sub(r'    FAIL TO REJECT H₀', '   ❌ FAIL TO REJECT H₀', text)
    text = re.sub(r'    Unexpected', '   ⚠️ Unexpected', text)
    # Also for dict and table
    text = re.sub(r"'Result': ' Reject H₀'", "'Result': '✅ Reject H₀'", text)
    text = re.sub(r"'Result': ' Fail to reject'", "'Result': '❌ Fail to reject'", text)
    text = re.sub(r' > Reject H₀<', ' >✅ Reject H₀<', text)
    text = re.sub(r' > Fail to reject<', ' >❌ Fail to reject<', text)
    # Add other emojis back
    text = re.sub(r'\n Sample Size:', '\n📊 Sample Size:', text)
    text = re.sub(r'\n Spearman Correlation Results:', '\n📈 Spearman Correlation Results:', text)
    text = re.sub(r'\n Pearson Correlation', '\n📈 Pearson Correlation', text)
    text = re.sub(r'\n Conclusion', '\n🔬 Conclusion', text)
    text = re.sub(r'\n Group Statistics:', '\n📊 Group Statistics:', text)
    text = re.sub(r'\n Model Summary:', '\n📊 Model Summary:', text)
    text = re.sub(r'\n Full Model Summary:', '\n📊 Full Model Summary:', text)
    text = re.sub(r'\n Model Performance Comparison:', '\n📊 Model Performance Comparison:', text)
    text = re.sub(r'\n Key Metrics:', '\n📈 Key Metrics:', text)
    text = re.sub(r'\n Model Comparison:', '\n📈 Model Comparison:', text)
    text = re.sub(r'\n Adjusted R²', '\n📈 Adjusted R²', text)
    text = re.sub(r'\n Optimal Lag Period by Country:', '\n📊 Optimal Lag Period by Country:', text)
    text = re.sub(r'\n REGRESSION ANALYSIS:', '\n📊 REGRESSION ANALYSIS:', text)
    text = re.sub(r'\n HYPOTHESIS TESTING RESULTS:', '\n🔬 HYPOTHESIS TESTING RESULTS:', text)
    return text

def clean_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    def clean_cell(cell):
        if 'source' in cell:
            cell['source'] = [add_emojis(line) for line in cell['source']]
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'data' in output:
                    for key, value in output['data'].items():
                        if isinstance(value, str):
                            output['data'][key] = add_emojis(value)
                        elif isinstance(value, list):
                            output['data'][key] = [add_emojis(item) if isinstance(item, str) else item for item in value]
                if 'text' in output:
                    output['text'] = [add_emojis(line) for line in output['text']]
        return cell
    
    notebook['cells'] = [clean_cell(cell) for cell in notebook['cells']]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    clean_notebook(r"d:\Programming\School\Data mining\Project UV\countries-uv.ipynb")