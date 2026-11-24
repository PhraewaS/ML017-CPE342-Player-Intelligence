# ML017 · Player Intelligence System · v10

โค้ดสำหรับโปรเจกต์ CPE342 Machine Learning ของทีม ML017 แยกตามงานทั้งห้า ไฟล์ v10 ที่ทีมส่งได้คะแนน **0.85475** ตามหน้าผลการส่ง Kaggle

## สร้าง CSV ฉบับเดิม

ใช้ Python 3.13 และเปิด Terminal ในโฟลเดอร์ที่มี README นี้ ตัวอย่างคำสั่งใช้ PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe build_v10.py --check
```

บน macOS หรือ Linux ใช้ `python3.13 -m venv .venv` แล้วใช้ `.venv/bin/python` แทน `.\.venv\Scripts\python.exe` คำสั่งจะสร้าง `outputs/ML017_submission_v10.csv` ขนาด 25,889 แถวและ 6 คอลัมน์ พร้อมเทียบ SHA-256 กับไฟล์ v10 เดิม ขั้นตอนนี้ใช้ผลทำนายที่บันทึกไว้ในโฟลเดอร์ของแต่ละงาน จึงรันได้ทันทีหลังติดตั้งไลบรารี

## โค้ดฉบับส่งและงานทดลอง

คำสั่งสร้าง CSV ฉบับส่งเริ่มจาก `build_v10.py` และใช้ผลทำนายใน `Task1/` ถึง `Task5/` ร่วมกับ `shared/` แต่ละ Task มีโค้ดเตรียมข้อมูล ฝึกโมเดล หรือคำนวณผลที่ใช้กับ v10 ตามวิธีรันด้านล่าง ส่วนการทดลองเปรียบเทียบโมเดลและวิเคราะห์ผลเพิ่มเติมอยู่ใน [experiments/](experiments/README.md)

## โค้ดของแต่ละงาน

| โฟลเดอร์ | งาน | ไฟล์หลัก |
|---|---|---|
| [Task1](Task1) | ตรวจจับผู้เล่นโกง (F2) | `train_accounts.py` ฝึกโมเดลสถิติการเล่นและปรับตามลักษณะบัญชี สูตรนี้มาจาก v8 |
| [Task2](Task2) | แบ่งกลุ่มผู้เล่น (Macro F1) | `train_segments_v5.py` ประมาณการกระจายของฟีเจอร์ในแต่ละกลุ่มและผสมกับโมเดลเดิม |
| [Task3](Task3) | ทำนายยอดใช้จ่าย 30 วัน (NMAE) | `spend_model.py` ฝึกโมเดลยอดใช้จ่าย และ `missing_segment.py` ปรับคำทำนายเมื่อข้อมูลกลุ่มผู้เล่นขาดหาย |
| [Task4](Task4) | จำแนกชื่อเกมจากภาพ (Macro F1) | `predict.py` ผสมคะแนนของ ConvNeXt, CLIP, DINO และโมเดล kernel |
| [Task5](Task5) | ตรวจความผิดปกติของบัญชี (F3) | `predict.py` ใช้กฎจาก `signature_rules.py`, `task5_rules.py` และ `stability.py` กับข้อมูลสี่ช่วงเวลา |

Task3 เป็นงานที่ปรับใน v10 ส่วน Task1, Task2, Task4 และ Task5 ใช้ผลชุดเดียวกับ v9

## รันและแก้โมเดลทีละ Task

คำสั่งด้านล่างใช้ `python` ใน environment ที่ติดตั้ง `requirements.txt` แล้ว ถ้ายังไม่ได้เปิดใช้งาน `.venv` ให้แทน `python` ด้วยพาธตามตัวอย่างด้านบน ใช้ไฟล์ ZIP ต้นฉบับของการแข่งขันหรือโฟลเดอร์ที่แตกแล้วเป็น `--data` และเก็บข้อมูลไว้ใน `data/` หรือโฟลเดอร์นอก repository ตัวอย่างนี้สมมติว่าวาง ZIP ไว้ใน `data/`

```powershell
$data = "data/player-intelligence-system-cpe-342-ml.zip"

python -m Task1.train_accounts --data $data --out trained/task1.npy
python -m Task2.train_segments_v5 --data $data --out trained/task2.npy

python -m Task3.spend_model --data $data --out trained/amount.npy
python -m Task3.missing_segment --data $data --base trained/amount.npy --out trained/task3.npy

python -m Task4.predict --out trained/task4.npy
python -m Task5.predict --data $data --out trained/task5.npy
```

Task1 และ Task3 ใช้ผลทำนายจากโมเดลฐานใน `cache/` ประกอบการฝึก ตัวเลือก `--audit-out` ของทั้งสองขั้นใน Task3 ใช้บันทึกผลตรวจเพื่อวิเคราะห์ต่อ Task5 รันกฎกับ `05_anomaly_detection/test.csv` ซึ่งเป็นข้อมูลที่โจทย์ให้สำหรับงานนี้

### ฝึกส่วน kernel ของ Task4 ใหม่

วาง `task4_images.zip` ที่มี `train.csv`, `test.csv` และภาพไว้ใน `data/` วางน้ำหนัก [DINOv2 Small](https://huggingface.co/Xenova/dinov2-small/blob/main/onnx/model.onnx) และ [CLIP ViT-B/32 Vision](https://huggingface.co/Xenova/clip-vit-base-patch32/blob/main/onnx/vision_model.onnx) ใน `models/` โดยตั้งชื่อไฟล์ `dinov2-small.onnx` และ `clip-vision.onnx` ตามลำดับ

```powershell
python -m pip install Pillow==12.3.0 onnxruntime
python -m Task4.extract_task4_embeddings --images data/task4_images.zip --models models --out embeddings --provider cpu
python -m Task4.train_task4 --images data/task4_images.zip --embeddings embeddings --audit --out trained/kernel_probabilities.npy
python -m Task4.predict --probabilities trained/kernel_probabilities.npy --out trained/task4.npy
```

ขั้นดึง embeddings อาจใช้เวลานานบน CPU คู่มือ [Task4/COMMANDS.md](Task4/COMMANDS.md) บันทึกวิธีฝึกโมเดลภาพสี่ตัวของทีม ซึ่งใช้ข้อมูลภาพ น้ำหนักโมเดล และ environment ของแต่ละชุด การประกอบ v10 ใช้คะแนนที่บันทึกไว้ของทั้งสี่ตัวร่วมกับผล kernel ใน `Task4/predictions/`

### ประกอบ CSV หลังแก้โค้ด

ระบุเฉพาะ Task ที่รันใหม่ได้ ตัวอย่างนี้ใช้ผลใหม่ของ Task3 และผล v10 สำหรับอีกสี่งาน ตัวเลือก `--check` ใช้เมื่อต้องการยืนยันว่าผลตรงกับ v10 เดิม

```powershell
python build_v10.py --task3 trained/task3.npy --out outputs/task3_experiment.csv
```

ตัวเลือก `--task1` ถึง `--task5` รับไฟล์ `.npy` แบบหนึ่งมิติ 25,889 ค่าในลำดับเดียวกับ `sample_submission.csv` สำหรับ Task4 ให้ส่ง **ป้ายคลาส** จาก `Task4.predict` เข้า `build_v10.py` ส่วนไฟล์ probabilities ใช้กับ `Task4.predict --probabilities`

## นำขึ้น GitHub

แตก ZIP แล้วนำ **ไฟล์และโฟลเดอร์ข้างใน** ขึ้น repository ของทีม โดยให้ `README.md`, `build_v10.py`, `shared/` และ `Task1/` ถึง `Task5/` อยู่ที่ราก repository เก็บ `.gitignore` ไว้ด้วย เพราะตั้งค่าให้ข้ามข้อมูลดิบ ผลการทดลอง และน้ำหนักโมเดลที่ดาวน์โหลดไว้ ชุดโค้ดนี้มีผลทำนายที่ใช้สร้าง v10 จึงควรใช้ repository แบบ Private และให้ผู้ตรวจเข้าถึงตามที่รายวิชาระบุใน LEB2
