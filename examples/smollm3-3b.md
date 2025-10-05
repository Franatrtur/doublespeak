# Model Name: `HuggingFaceTB/SmolLM3-3B-Base`

A pretty capable model for both text and code.

## Demo 1 - Article

#### Settings

*Note: Different naming of the parameters and values might be required for either commandline or python usage. Check the documentation.*
```
model-name="HuggingFaceTB/SmolLM3-3B-Base"
end-bias=3 
top-p=0.65
ending=natural
```

#### Input

 - Opening:
```
In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.
```
 - Secret message: `send 0.5btc to adress 0xe4ffd093`

#### Output

```
In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. "It was one of the strangest things I have ever seen," says scientist Charles Stewart. "They seemed like they had just walked out of a fantasy novel. We knew that we had to do something to protect them, before they were completely destroyed."
What did the unicorns eat? That is still unknown, as the scientists were not allowed to leave the valley. But, it was suggested that they ate wild horses, camels, and llamas. It was also reported that the unicorns only left the valley at night to forage for food.
Why is it important to preserve this unique ecosystem? Well, we need to do our part to save these animals from extinction. "The world is changing rapidly," says Stewart. "We need to do everything we can to protect these animals, before it is too late."
The scientists are currently working on a plan to protect the unicorns, and their habitat.
```

## Demo 2 - Code

#### Settings

*Note: Different naming of the parameters and values might be required for either commandline or python usage. Check the documentation.*
```
model-name="HuggingFaceTB/SmolLM3-3B-Base"
end-bias=6
top-p=0.75
ending=natural
```

#### Input

 - Opening:
```
import os

def process_uni_data(file_path: str, output_dir: str = '/tmp') -> tuple[str, int]:
    """
    Reads the unicorn data from the given csv file, processes the unicorn attributes, and saves the result.

    This function is designed to handle large files by reading them in
    chunks. It will return the most
```
 - Secret message: `send 0.5btc to adress 0xe4ffd093`

#### Output

```
import os

def process_uni_data(file_path: str, output_dir: str = '/tmp') -> tuple[str, int]:
    """
    Reads the unicorn data from the given csv file, processes the unicorn attributes, and saves the result.

    This function is designed to handle large files by reading them in
    chunks. It will return the most frequent class and its corresponding label,
    which can be used to create a map between unicorn attributes and their
    labels.

    Args:
        file_path (str): Path to the csv file containing unicorn data.
        output_dir (str, optional): Path to the directory where the output file will be saved. Defaults to '/tmp'.

    Returns:
        tuple[str, int]: Most frequent class label and its frequency
    """

    output_file_path = os.path.join(output_dir, 'processed_data.txt')
    if os.path.exists(output_file_path):
        print('Processed file exists at %s' % output_file_path)
        with open(output_file_path, 'r') as f:
            labels = f.read()
            label = int(labels)
        return output_file_path, label

    max_class = -1
    with open(file_path, 'r') as f:
        print('Processing file at %s' % file_path)
        num_classes = 0
        line_count = 0
        label = -1
        label_map = dict()
        for line in f:
            label = line.split(',')[0]
            if label in label_map:
                continue
            label_map[label] = num_classes
            num_classes += 1

            unicorns = [x for x in line.split(',')[1:] if x]
            max_class = max(max_class, unicorns.count('uni'))
            if max_class > num_classes:
                max_class = num_classes

        max_class -= 1

    if max_class < 0:
        raise ValueError('Maximum unicorn class label not found!')

    print('Max class label: %d' % max_class)
    print('Number of classes: %d' % num_classes)

    with open(output_file_path, 'w') as f:
        f.write(str(max_class))

    return output_file_path, max_class
```