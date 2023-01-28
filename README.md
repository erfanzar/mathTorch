# mathTorch

 mathTorchis a Python library for deep learning from scratch.

## Installation

Use the package manager [pip](https://pypi.org/project/mathTorch/) to install mathTorch.

```bash
pip install mathTorch --user
```

## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


## Usage

```python
import mathTorch.nn as nn

# Creating just a simple Layer

linear = nn.Linear(1,5)
relu = nn.ReLU()

# x you input data

x = relu(linear(x))


```

## utils

```python
    from mathTorch.utils.utils import *

    # Reading Toml File
    toml_file = read_toml('<path>.toml')

    # Reading Txt File
    txt_file = read_txt('<path>.txt')

    # Reading Json File
    json_file = read_json('<path>.json')

    # Reading Yaml File
    yaml_file = read_yaml('<path>.yaml')

    # Downloading file from url
    download('https://urlhere')

    # Time functions
    time_sec:str = time_since(since,percent)
    time_min:str = as_minutes(seconds)


    ## and many many more :) go to source ;\ and check
```



## lightning

```python
    from mathTorch.utils.lightning import *

    # Reading Toml File
    

    #  check if attr exist return attr or false and pass
    out = attr_exist_check_(attr)

    # calculating iou
    iou = iou(box1,box2)

      # calculating iou
    iou_error = bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7)

     # calculating iou avg
    avg_iou_acc = avg_iou(boxes,clusters)

    # Kmeans
    anchors = kmeans(boxes, k, dist=np.median)

    ## and many many more :) go to source ;\ and check
```


## all utils functions 

```python
    from mathTorch.utils.lightning import *
    from mathTorch.utils.interface import *
    from mathTorch.utils.utils import *
    from mathTorch.utils.nlp import *
    from mathTorch.utils.fix_dll import *
```

## Beta

This Package is still in beta mode and I'm working on it

Please make sure to update tests as appropriate.


## 🚀 About Me
Hi there 👋
I like to train deep neural nets on large datasets 🧠.
Among other things in this world:)

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Used By

This project is used by the following companies:

- I Don't think like anyone would like to use it why should you use mathTorch when you can use Pytorch v2.0.0 :)



## Author

- [@erfanzar](https://www.github.com/erfanzar)
