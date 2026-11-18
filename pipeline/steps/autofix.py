"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml

from ..common import ERROR, ROOT, parse_yolo_line

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _clip_box(text):
    class_id, cx, cy, width, height = parse_yolo_line(text)
    left = max(0.0, cx - width / 2)
    top = max(0.0, cy - height / 2)
    right = min(1.0, cx + width / 2)
    bottom = min(1.0, cy + height / 2)
    if right <= left or bottom <= top:
        return None
    values = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    return " ".join([str(class_id), *(str(value) for value in values)])


def fix(ds, issues, cfg):
    if ds.root.resolve().is_relative_to((ROOT / "data" / "raw").resolve()):
        raise ValueError("Autofix chỉ được chạy trên bản sao làm việc.")

    fixes = []

    def record(image, line, action, before, after):
        fixes.append(dict(image=image, line=line, action=action, before=before, after=after))

    line_errors = defaultdict(lambda: defaultdict(set))
    remove_images = set()
    missing_labels = set()
    rename_classes = False
    for issue in issues:
        if issue.severity != ERROR:
            continue
        code = issue.code
        if code in REMOVE_CODES or code.startswith("DUP_IMAGE_"):
            remove_images.add(issue.image)
        elif code == "BAD_CLASS_NAME":
            rename_classes = True
        elif code == "MISSING_LABEL_FILE":
            missing_labels.add(issue.image)
        elif issue.line is not None and (code in DROP_CODES or code == "BOX_OUT_OF_IMAGE"):
            line_errors[issue.image][issue.line].add(code)

    if rename_classes:
        data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) if ds.yaml_path.exists() else None
        data = data or {}
        previous_names = ds.yaml_names()
        names = dict(enumerate(cfg["classes"]))
        if previous_names != names:
            data["names"] = cfg["classes"]
            ds.yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            record("data.yaml", None, "rename_classes", str(previous_names), str(names))

    for image_path in ds.image_files():
        stem = image_path.stem
        if stem in remove_images and image_path.is_file():
            image_path.unlink()
            ds.label_path(stem).unlink(missing_ok=True)
            record(stem, None, "remove_image", image_path.name, "")

    for stem in sorted(missing_labels - remove_images):
        if ds.image_path(stem) is not None and not ds.label_path(stem).exists():
            record(stem, None, "needs_label", "(không có file)", "chuyển người gán")

    for stem in sorted(line_errors.keys() - remove_images):
        path = ds.label_path(stem)
        if not path.is_file():
            continue
        original = path.read_bytes().decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
        updated = []
        for line, raw in enumerate(original, 1):
            codes = line_errors[stem].get(line)
            if not codes or not raw.strip():
                updated.append(raw)
                continue

            before = raw.rstrip("\r\n")
            if codes & DROP_CODES:
                after = None
            else:
                try:
                    after = _clip_box(before)
                except (ValueError, OverflowError):
                    after = None

            log_before = before.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
            if after is None:
                record(stem, line, "drop_line", log_before, "")
            elif after != before:
                updated.append(after + raw[len(before):])
                record(stem, line, "clip_box", log_before, after)
            else:
                updated.append(raw)

        if updated != original:
            path.write_bytes("".join(updated).encode("utf-8", errors="surrogateescape"))

    return fixes
