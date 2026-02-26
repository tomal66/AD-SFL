import json
import sys

def optimize_notebook():
    file_path = 'demo_setup.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = "".join(cell['source'])
        
        # Patch the test_loader block in Cell 10
        if "test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)" in source:
            lines = []
            for line in cell['source']:
                if "test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)" in line:
                    lines.extend([
                        "test_loader = DataLoader(\n",
                        "    test_data, \n",
                        "    batch_size=batch_size, \n",
                        "    shuffle=False, \n",
                        "    num_workers=4, \n",
                        "    pin_memory=True\n",
                        ")\n"
                    ])
                else:
                    lines.append(line)
            cell['source'] = lines

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')

if __name__ == '__main__':
    optimize_notebook()
