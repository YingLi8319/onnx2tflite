import os
import cv2
import logging
import numpy as np

LOG = logging.getLogger("Quantization DataLoder :")

class RandomLoader(object):
    def __init__(self, target_size):
        self.target_size = target_size
        LOG.warning(f"Generate quantization data from random, it's will lead to accuracy problem!")
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index > 5:
            raise StopIteration()
        self.index += 1
        return [np.random.randn(*self.target_size).astype(np.float32)]
    
class ImageLoader(object):
    """
    generate data for quantization from image datas.
    img_quan_data = (img - mean)/std, it's important for accuracy of model.

    新增：
      - debug/log_first/recursive/valid_exts 參數
      - 進入 __iter__/__next__ 時印出前幾筆的 shape/dtype/min/max（正規化前/後）
      - 安全處理 batch 維度（None → 1）
      - 讀檔失敗會略過並告警，不會整個中斷
      - __len__ 以利外部快速檢查
    """
    DEFAULT_FORMATS = ['.jpg', '.png', '.jpeg']

    def __init__(self, img_root, target_size,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 debug=True, log_first=3,
                 recursive=False, valid_exts=None):
        import os, numpy as np

        assert os.path.exists(img_root), f"{img_root} is not exists, please check!"

        self.debug = bool(debug)
        self.log_first = int(log_first)
        self.valid_exts = [e.lower() for e in (valid_exts or self.DEFAULT_FORMATS)]
        self.recursive = bool(recursive)

        # 掃描影像清單
        if self.recursive:
            fns = []
            for r, _, files in os.walk(img_root):
                for fn in files:
                    if os.path.splitext(fn)[-1].lower() in self.valid_exts:
                        fns.append(os.path.join(r, fn))
        else:
            fns = [os.path.join(img_root, fn) for fn in os.listdir(img_root)
                   if os.path.splitext(fn)[-1].lower() in self.valid_exts]

        fns = sorted(fns)
        self.fns = fns
        self.nums = len(self.fns)
        assert self.nums > 0, f"No images detected in {img_root}."

        # 原邏輯保留：>100 提醒
        try:
            LOG  # 若庫內有 LOG，沿用
            if self.nums > 100:
                LOG.warning(f"{self.nums} images detected, the number of recommended images is less than 100.")
            else:
                LOG.info(f"{self.nums} images detected.")
        except NameError:
            if self.debug:
                if self.nums > 100:
                    print(f"[ImageLoader] WARN: {self.nums} images detected (recommend <= 100).")
                else:
                    print(f"[ImageLoader] INFO: {self.nums} images detected.")

        # 解析 target_size: [batch, H, W, C]
        self.batch = target_size[0]
        self.size  = target_size[1:-1]  # (H, W)
        # 安全處理 batch
        try:
            self.batch = int(self.batch) if self.batch is not None else 1
        except Exception:
            self.batch = 1
        if self.batch < 1:
            self.batch = 1
        self.size = tuple(int(x) for x in self.size)

        # mean/std 轉成可 broadcast 形狀（1,1,3）
        if mean is not None:
            mean = np.array(mean, dtype=np.float32).reshape((1, 1, -1))
        if std is not None:
            std = np.array(std, dtype=np.float32).reshape((1, 1, -1))
        self.mean, self.std = mean, std

        if self.debug:
            print(f"[ImageLoader] root={img_root}")
            print(f"[ImageLoader] total_files={self.nums}  show_first={min(self.nums, self.log_first)}")
            for i, p in enumerate(self.fns[:self.log_first]):
                print(f"[ImageLoader] sample[{i}]: {p}")
            print(f"[ImageLoader] input_target_size (HxW)={self.size}  batch={self.batch}")
            print(f"[ImageLoader] valid_exts={self.valid_exts}")
            print(f"[ImageLoader] mean={None if self.mean is None else self.mean.flatten().tolist()}")
            print(f"[ImageLoader] std ={None if self.std  is None else self.std.flatten().tolist()}")

    def __len__(self):
        return self.nums

    def __iter__(self):
        self.index = 0
        self._yielded = 0
        if self.debug:
            print(f"[ImageLoader] __iter__ start; nums={self.nums}, batch={self.batch}, size={self.size}")
        return self

    def __next__(self):
        import cv2, numpy as np, os

        while self.index < self.nums:
            fn = self.fns[self.index]
            self.index += 1

            img = cv2.imread(fn)
            if img is None:
                if self.debug:
                    print(f"[ImageLoader][WARN] cv2.imread failed: {fn}")
                continue  # 跳過壞檔

            # BGR -> RGB、resize
            try:
                img = cv2.resize(img, self.size)[:, :, ::-1]
            except Exception as e:
                if self.debug:
                    print(f"[ImageLoader][WARN] resize/RGB failed: {fn} ({e})")
                continue

            arr = img.astype(np.float32)

            # 轉換前資訊
            if self.debug and self._yielded < self.log_first:
                try:
                    print(f"[ImageLoader][pre ] shape={arr.shape} dtype={arr.dtype} "
                          f"range=({float(arr.min())},{float(arr.max())})  file={os.path.basename(fn)}")
                except Exception:
                    pass

            # 正規化
            if self.mean is not None:
                arr = (arr - self.mean)
            if self.std is not None:
                arr = arr / self.std

            # 轉換後資訊
            if self.debug and self._yielded < self.log_first:
                try:
                    print(f"[ImageLoader][post] shape={arr.shape} dtype={arr.dtype} "
                          f"range=({float(arr.min())},{float(arr.max())})")
                except Exception:
                    pass

            # 加 batch 維度、必要時複製
            arr = np.expand_dims(arr, axis=0)  # [1,H,W,C]
            if isinstance(self.batch, int) and self.batch > 1:
                arr = np.repeat(arr, self.batch, axis=0).astype(np.float32)

            self._yielded += 1
            return [arr]  # TFLite 代表性資料格式：List[np.ndarray]

        if self.debug:
            print(f"[ImageLoader] StopIteration; yielded={self._yielded}/{self.nums}")
        raise StopIteration()

    
