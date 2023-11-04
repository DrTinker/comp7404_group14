# comp7404_group14
comp7404_group14

## Project structure
```
    - configs: 在这个模块中，定义所有可以配置的内容，方便在将来更改。例如：超参数、文件夹路径、标志等。可以是json、yaml等文件类型。

    - dataloader: 数据加载器。我们把数据加载、数据集定义和数据预处类和函数都放在这里。

    - evaluation: 评估部分，是评估模型的性能和准确性相关的代码。

    - notebook：如果使用Jupyter Notebook实验代码，可以放在这里。

    - data：存放数据的地方

    - models：存放训练好的模型的地方

    - modules：主要功能块存放的地方，比如Transformers、某个Network主体部分。

    - utils：在多个地方使用的一些函数方法。tools或者scripts等都可以放在这。

    - test：一些测试脚本

    - requirements.txt：项目依赖库
```txt
