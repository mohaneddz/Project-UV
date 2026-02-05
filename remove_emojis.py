import json
import re

def remove_emojis(text):
    # Regex pattern to match emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002700-\U000027BF"  # dingbats
        "\U0001f926-\U0001f937"  # gestures
        "\U00010000-\U0010ffff"  # other unicode
        "\u2640-\u2642"  # gender symbols
        "\u2600-\u2B55"  # misc symbols
        "\u200d"  # zero width joiner
        "\u23cf"  # eject symbol
        "\u23e9"  # fast forward
        "\u231a"  # watch
        "\ufe0f"  # variation selector
        "\u3030"  # wavy dash
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def clean_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    def clean_cell(cell):
        if 'source' in cell:
            cell['source'] = [remove_emojis(line) for line in cell['source']]
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'data' in output:
                    for key, value in output['data'].items():
                        if isinstance(value, str):
                            output['data'][key] = remove_emojis(value)
                        elif isinstance(value, list):
                            output['data'][key] = [remove_emojis(item) if isinstance(item, str) else item for item in value]
                if 'text' in output:
                    output['text'] = [remove_emojis(line) for line in output['text']]
        return cell
    
    notebook['cells'] = [clean_cell(cell) for cell in notebook['cells']]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    clean_notebook(r"d:\Programming\School\Data mining\Project UV\countries-uv.ipynb")