# Tencent Hunyuan 3D Backend Wrapper

这个项目的目标很简单：

- 输入一个任务 JSON
- 只处理 `need_generation=true` 的对象
- 调用腾讯混元 3D 生成模型
- 返回后端需要的结果 JSON

后端真正需要关心的主入口只有一个：

```python
process_generation_task(payload, api_key=API_KEY)
```

对应实现文件是 [`task_model_service.py`](/Users/jan/Desktop/Picture2Model/task_model_service.py)。

## 目录说明

核心文件：

- `task_model_service.py`：后端调用入口
- `hunyuan_3d.py`：腾讯混元 3D 底层调用逻辑
- `requirements.txt`：Python 依赖

测试和示例：

- `test_run.py`：最小真实调用示例
- `sample_task_local.json`：示例输入 JSON
- `test_task_model_service.py`：单元测试

运行产物：

- `generated_models/`：生成结果目录

## 环境要求

- Python 3.10+

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

如果本地还没有 `certifi`，请额外安装：

```bash
python3 -m pip install certifi
```

## 主函数

主函数定义在：

[`task_model_service.py`](/Users/jan/Desktop/Picture2Model/task_model_service.py#L286)

```python
process_generation_task(
    payload,
    api_key=None,
    config=None,
    object_generator=None,
    model_reference_builder=None,
    progress_callback=None,
)
```

后端正常接入时，最常用的是这两个参数：

- `payload`：输入任务 JSON，支持 `dict` 或 JSON 字符串
- `api_key`：腾讯混元 API Key

最小调用方式：

```python
from task_model_service import process_generation_task

result = process_generation_task(payload, api_key="你的API_KEY")
```

## 输入 JSON 格式

输入格式如下：

```json
{
  "task_id": "task_001",
  "objects": [
    {
      "object_id": "obj_001",
      "label": "laptop",
      "crop_url": "https://example.com/tasks/task_001/crops/obj_001.png",
      "need_generation": true,
      "model_url": ""
    },
    {
      "object_id": "obj_002",
      "label": "cup",
      "crop_url": "https://example.com/tasks/task_001/crops/obj_002.png",
      "need_generation": false,
      "model_url": "https://example.com/assets/models/cup_001.glb"
    }
  ]
}
```

字段说明：

- `task_id`：任务 ID
- `objects`：对象列表
- `object_id`：对象 ID
- `label`：对象名称
- `crop_url`：对象裁剪图地址，也可以是本地路径
- `need_generation`：是否需要调用混元生成
- `model_url`：已有模型地址。当前封装里如果 `need_generation=false`，该对象会被直接跳过，不会出现在 `generated_models` 返回结果里

## 输出 JSON 格式

输出格式如下：

```json
{
  "task_id": "task_001",
  "generated_models": [
    {
      "object_id": "obj_001",
      "label": "laptop",
      "model_url": "https://example.com/tasks/task_001/models/obj_001.glb"
    }
  ]
}
```

说明：

- 只返回本次真正生成过的对象
- `need_generation=false` 的对象不会写进 `generated_models`

## 后端调用示例

最小示例：

```python
from task_model_service import process_generation_task

payload = {
    "task_id": "task_001",
    "objects": [
        {
            "object_id": "obj_001",
            "label": "laptop",
            "crop_url": "input_images/pencil.png",
            "need_generation": True,
            "model_url": "",
        },
        {
            "object_id": "obj_002",
            "label": "cup",
            "crop_url": "",
            "need_generation": False,
            "model_url": "https://example.com/assets/models/cup_001.glb",
        },
    ],
}

result = process_generation_task(payload, api_key="你的API_KEY")
print(result)
```

如果后端希望返回 URL 形式的 `model_url`，可以这样调用：

```python
from task_model_service import (
    build_task_model_url_builder,
    process_generation_task,
)

result = process_generation_task(
    payload,
    api_key="你的API_KEY",
    model_reference_builder=build_task_model_url_builder("https://example.com"),
)
```

这样返回的 `model_url` 会是：

```text
https://example.com/tasks/<task_id>/models/<object_id>.glb
```

## 可选配置

如果你要调整输出目录、模型版本或轮询时间，可以传 `TaskGenerationConfig`：

```python
from pathlib import Path

from task_model_service import TaskGenerationConfig, process_generation_task

config = TaskGenerationConfig(
    output_dir=Path("generated_models/tasks"),
    model="hy-3d-3.1",
    generate_type="Normal",
    face_count=100000,
    enable_pbr=True,
    poll_interval=8.0,
    timeout=900,
)

result = process_generation_task(
    payload,
    api_key="你的API_KEY",
    config=config,
)
```

配置项说明：

- `output_dir`：输出根目录
- `model`：腾讯混元模型版本，支持 `hy-3d-3.0` 和 `hy-3d-3.1`
- `generate_type`：生成模式，支持 `Normal`、`LowPoly`、`Geometry`、`Sketch`
- `face_count`：可选面数
- `enable_pbr`：是否启用 PBR
- `poll_interval`：轮询状态间隔，单位秒
- `timeout`：单个对象最长等待时间，单位秒

## 进度输出

如果后端想拿到运行中的状态，可以传 `progress_callback`：

```python
from task_model_service import process_generation_task

def progress_printer(event: dict) -> None:
    print(event)

result = process_generation_task(
    payload,
    api_key="你的API_KEY",
    progress_callback=progress_printer,
)
```

典型阶段包括：

- `loading_input`
- `submitting`
- `submitted`
- `polling`
- `downloading`
- `completed`

## 本地测试

### 1. 直接跑最小测试脚本

[`test_run.py`](/Users/jan/Desktop/Picture2Model/test_run.py) 是最小真实调用示例。

你只需要修改两个地方：

- `API_KEY`
- `INPUT_JSON`

然后运行：

```bash
python3 test_run.py
```

### 2. 命令行方式运行

也可以直接运行封装脚本：

```bash
python3 task_model_service.py \
  --input-json sample_task_local.json \
  --output-json task_result_real.json
```

如果想本地离线测试输入输出链路，不调用腾讯接口：

```bash
python3 task_model_service.py \
  --input-json sample_task_local.json \
  --output-json task_result_local.json \
  --mock
```

## 输出文件位置

默认输出目录：

```text
generated_models/tasks/
```

以 `task_id=task_001` 为例，典型结果结构如下：

```text
generated_models/tasks/task_001/
├── models/
│   └── obj_001.glb
├── response.json
└── work/
    └── obj_001/
        ├── latest_model.json
        ├── result.json
        └── ...
```

说明：

- `models/`：最终统一给后端使用的模型文件
- `response.json`：本次任务的最终返回 JSON
- `work/`：每个对象的中间产物和下载结果

## 异常说明

主函数可能抛出以下异常：

- `TaskPayloadError`：输入 JSON 不合法
- `TaskGenerationError`：某个对象生成失败
- `Hunyuan3DError`：腾讯混元底层调用异常，通常会被包装进 `TaskGenerationError`

建议后端调用时这样处理：

```python
from task_model_service import (
    TaskGenerationError,
    TaskPayloadError,
    process_generation_task,
)

try:
    result = process_generation_task(payload, api_key="你的API_KEY")
except TaskPayloadError as exc:
    print(f"输入参数错误: {exc}")
except TaskGenerationError as exc:
    print(f"生成失败: {exc}")
```

## 输入限制和注意事项

- `need_generation=true` 时，`crop_url` 必须存在
- `crop_url` 支持远程 URL，也支持本地文件路径
- 如果输入图片过大，底层会自动压缩
- 如果对象主体过小，腾讯模型侧可能直接失败
- `hy-3d-3.1` 不支持 `LowPoly`
- 返回结果里的 `model_url` 默认是本地绝对路径；如果需要 HTTP 地址，请传 `model_reference_builder`

## 推荐给后端的最小交付文件

如果只保留最小可用集，建议保留：

- `task_model_service.py`
- `hunyuan_3d.py`
- `requirements.txt`

如果还希望保留本地测试能力，再加上：

- `test_run.py`
- `sample_task_local.json`
