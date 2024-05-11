# Emotion Sign Language Translation

## 框架图

![无法显示](./assets/arc.png)

## 附加

```bash
# 解决 huggince face 无法下载模型
# 目录：/root/miniconda3/lib/python3.8/site-packages/huggingface_hub/constants.py 
# 将 ENDPOINT = os.getenv("HF_ENDPOINT") or ("https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co") 类似的修改成
ENDPOINT = " https://hf-mirror.com"

# 命令行删除 git 所有进程：
ps aux | grep git | awk '{print $2}' | xargs kill -9
```

## 运行项目

```python
def gen_videos_vectors(x, y):
    # 代码在utils里边
    pass


# 首先要运行 keypoints预处理
gen_videos_vectors('./data/How2Sign/pending_keypoints', './data/How2Sign/keypoints')
```

## 待更新
