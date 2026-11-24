# การทดลองเสริมของทีม ML017

ผลการทดลองด้านล่างใช้วิเคราะห์ภาพและเปรียบเทียบโมเดลบนข้อมูลฝึก ส่วนวิธีสร้าง CSV ฉบับ v10 อยู่ใน [README หลัก](../README.md) และขั้นตอนเตรียมข้อมูลกับฝึกโมเดลภาพอยู่ใน [Task4/COMMANDS.md](../Task4/COMMANDS.md)

เปิด Terminal ที่ราก repository และแตกชุดภาพทางการไว้ใน `data_task4/` ตามคู่มือ Task4 สคริปต์ใช้ข้อมูลภาพชุดฝึกและไฟล์จากการทดลองเดิมใน `work/review/` สำหรับงานที่ต้องใช้ PyTorch และ timm ให้ใช้ environment ของ Task4

| สคริปต์ | สิ่งที่ตรวจและไฟล์ที่ต้องใช้ |
|---|---|
| `rotation_probe.py` | เปรียบเทียบภาพเดิมและภาพแก้เอียง ใช้ดัชนีโฟลด์ ConvNeXt ใน `work/review/t4_runs/` และผล OOF ใน `work/review/t4_oof_logprob_ens.npy` |
| `backbone_benchmark.py` | เปรียบเทียบ embeddings ของตัวเข้ารหัสและมุมมองภาพ ใช้แคช `t4_views_*_train.npy` จาก `--work` และผล OOF กับดัชนีโฟลด์เดิมจาก `work/review/` |
| `patch_classifier.py` | ทดสอบตัวจำแนกที่ใช้ patch features บนชุดตรวจ ใช้แคช patch, checkpoint ของ DINOv3 และ `val_indices.npy` ที่เก็บข้าง checkpoint |
| `scene_match_report.py` | เปรียบเทียบผล OOF ตามความคล้ายของฉากและสร้างภาพตัวอย่างคู่ฉาก ใช้ `t4_scene_retrieval.npz` จากตัวสร้างแคช และผล OOF เดิมใน `work/review/` |

```shell
python -m experiments.Task4.rotation_probe .
python -m experiments.Task4.backbone_benchmark . --work work/review
python -m experiments.Task4.patch_classifier . --cache work/review/t4_dinov3_strict_patches.npy --checkpoint work/review/t4_scene_dinov3_strict/best.pt
python -m experiments.Task4.scene_match_report . work/review
```
