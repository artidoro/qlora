import json
import os
import click
import random

def convert_to_fastchat_datas(data_path, output_path):
    key_prefix = data_path.split('/')[-1].split('.')[0]
    fastchat_datas = []
    datas = json.load(open(data_path, "r"))
    for index, data in enumerate(datas):
        id_ = key_prefix + '_' + str(index)

        user_message = data['instruction']
        if data.get("input", ""):
            user_message += "\n" + data["input"]
        assistant_message = data['output']

        conversations = [
            {"from": "human", "value": user_message},
            {"from": "gpt", "value": assistant_message},
        ]
        data = {
            "id": id_,
            "conversation": conversations,
        }

        fastchat_datas.append(data)

    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    print(f"Data size : {len(fastchat_datas)}")
    print("Head 3 datas")
    print(json.dumps(fastchat_datas[:3], indent=2, ensure_ascii=False))
    with open(output_path, 'w') as f:
        json.dump(fastchat_datas, f, ensure_ascii=False, indent=2)


def convert_line_json_data_to_fastchat_datas(data_path, output_path):
    with open(data_path, "r") as r_f:
        datas = []
        for line in r_f:
            datas.append(json.loads(line.rstrip()))

    print(f"Data size : {len(datas)}")
    print("Head 3 data")
    print(json.dumps(datas[:3], indent=2, ensure_ascii=False))
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    with open(output_path, "w") as w_f:
        json.dump(datas, w_f, indent=2, ensure_ascii=False)

def detect_json_file_type(file_path):
    is_per_line = True
    
    with open(file_path, 'r') as f:
        try:
            for line in f:
                json.loads(line)
                break
        except json.JSONDecodeError:
            is_per_line = False

    if is_per_line:
        return 'per_line'
    else:
        return 'whole_file'

def conver_data(data_path, output_path):
    if detect_json_file_type(data_path) == "per_line":
        return convert_line_json_data_to_fastchat_datas(data_path, output_path)
    else:
        return convert_to_fastchat_datas(data_path, output_path)



def merge_data(input_path, output_path, sample_n):
    data_paths = [os.path.join(input_path, file) for file in os.listdir(input_path)]
    datas = []
    for data_path in data_paths:
        sub_datas = json.load(open(data_path, "r"))
        print(data_path, len(sub_datas))
        datas.extend(sub_datas)

    print(f"data size : {len(datas)}")
    if sample_n:
        print(f"sample {sample_n} data")
        datas = random.sample(datas, sample_n)
    print("Head 5 data")
    print(json.dumps(datas[:5], indent=2, ensure_ascii=False))
    with open(output_path, 'w') as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)


@click.command()
@click.option('--func', type=click.Choice(['convert', 'merge']), required=True, help='Function name, options are convert or merge')
@click.option('--input_path', type=str, required=True, help='Input files, can be multiple files')
@click.option('--output_path', type=str, required=True, help='Output file, can only be one file')
@click.option('--sample_n', type=int, required=False, help='Output directory, can only be one directory')
def process_data(func, input_path, output_path, sample_n):
    """
    Process data in the specified input directories according to the specified function, 
    and output to the specified output directory.
    """
    if func == 'convert':
        # Implement the convert functionality here
        conver_data(input_path, output_path)
        print(f'Performing convert operation on {input_path} and output to {output_path}')
    elif func == 'merge':
        # Implement the merge functionality here
        merge_data(input_path, output_path, sample_n)
        print(f'Performing merge operation on {input_path} and output to {output_path}')
    else:
        click.echo("The function name must be either 'convert' or 'merge'")

if __name__ == '__main__':
    process_data()
