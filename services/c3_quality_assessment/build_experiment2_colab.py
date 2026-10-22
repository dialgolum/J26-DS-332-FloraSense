"""Package Experiment 2 support separately from immutable V2 data; never train."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]


def notebook(dataset_hash, support_hash):
    cells=[]
    def md(source):
        cells.append({"cell_type":"markdown","metadata":{},"source":source.splitlines(keepends=True)})
    def code(source):
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":source.splitlines(keepends=True)})
    md('''# FloraSense C3 Experiment 2 — fixed-split A/B, hosted Colab T4

Upload both `c3_features_v2_20261022.zip` and `c3_experiment2_support.zip` to
**My Drive / FloraSense_C3 / experiment2**. Use Runtime → Change runtime type →
T4 GPU. Run cells in order. No local runtime, new splits or Experiment 1 weights.
Class 1 is Not Acceptable (root) / One Side (leaf: branch origins, not foliage).
''')
    code('''# 1. Mount Drive
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
DRIVE_DIR = Path("/content/drive/MyDrive/FloraSense_C3/experiment2")
''')
    code('''# 2. Validate archives and extract into a fresh runtime directory
import hashlib, json, shutil, stat, zipfile
from datetime import datetime, timezone
def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()
DATA_SHA = "'''+dataset_hash+'''"
SUPPORT_SHA = "'''+support_hash+'''"
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
WORK = Path("/content") / ("c3_experiment2_" + stamp)
WORK.mkdir(exist_ok=False)
DATA_ROOT, CODE_ROOT = WORK / "dataset", WORK / "support"
for name, expected, destination in [
    ("c3_features_v2_20261022.zip",DATA_SHA,DATA_ROOT),
    ("c3_experiment2_support.zip",SUPPORT_SHA,CODE_ROOT)]:
    local = WORK / name
    shutil.copyfile(DRIVE_DIR / name,local)
    assert sha256(local)==expected, "Wrong or incomplete archive: " + name
    with zipfile.ZipFile(local) as archive:
        names=archive.namelist()
        assert len(names)==len(set(names))
        for member in archive.infolist():
            path=(destination/member.filename).resolve()
            assert path.is_relative_to(destination.resolve())
            assert ":" not in member.filename and "\\\\" not in member.filename
            assert not stat.S_ISLNK(member.external_attr >> 16)
        archive.extractall(destination)
print("Archives verified and extracted")
''')
    code('''# 3. Dependencies and hosted GPU (keep Colab's matched CUDA torch/torchvision)
import sys, subprocess, platform
assert platform.system() != "Windows", "Hosted Colab only"
subprocess.check_call([sys.executable,"-m","pip","install","-q","scikit-learn==1.7.2","pandas>=2.2,<3","Pillow>=10","matplotlib>=3.8"])
import torch
assert torch.cuda.is_available(), "Select a T4 GPU runtime"
print("GPU",torch.cuda.get_device_name(0),"torch",torch.__version__)
sys.path.insert(0,str(CODE_ROOT))
from services.c3_quality_assessment.experiment2_training import validate_dataset, run_architecture, compare_results
''')
    code('''# 4. Validate all V2 data and six saved partitions; no model construction
release, manifests, splits = validate_dataset(DATA_ROOT)
print(release["dataset_version"],release["workbook_sha256"])
for task in ("root","leaf"):
    print(task,manifests[task].label.value_counts().to_dict())
    for fold,frame in enumerate(splits[task],1):
        print("fold",fold,frame.groupby(["split","label"]).size().to_dict())
RUN_ROOT = DRIVE_DIR / "runs" / stamp
print("Output:",RUN_ROOT)
''')
    md('''## Training configuration

Both approaches load the identical exported train/validation/test rows. Neither
outer nor inner folds are generated. Plant_ID namespaces Sample_ID; varieties
may appear across partitions as in Experiment 1. This is not held-out-variety
evaluation. Root and leaf remain independent. No Plant_ID feature is supplied.

Same original + two fixed augmentation tensors: EXIF orientation, fit to 208,
center on gray 224 canvas, ImageNet normalization, flip, ±4° rotation and mild
color jitter. No crops; junctions must remain visible. Training-only class
weights; seed 42 plus fold; AdamW, batch 16, head LR .001, layer4 LR .00001,
weight decay .01, up to 40 epochs, patience 6. B starts layer4 adaptation at
epoch 1 (no warm-up); both keep BatchNorm running statistics frozen. Validation
weighted loss selects checkpoints. Test partitions are evaluated only afterward.

A and B are initialized independently from ImageNet, never from a full-data
checkpoint. Review results jointly after both complete; do not tune B based on
A outer-test errors. Fold checkpoints are saved to Drive; no deployment refit or
test-selected best fold is produced. Rerunning an existing arm is refused.
''')
    code('''# 5. Run architecture A: frozen backbone + head, separate root/leaf models
run_architecture(DATA_ROOT,RUN_ROOT,"A")
''')
    code('''# 6. Run architecture B: layer4 + head, exact same saved partitions/images
run_architecture(DATA_ROOT,RUN_ROOT,"B")
''')
    code('''# 7. Compare only after both arms complete
import pandas as pd
compare_results(RUN_ROOT)
display(pd.read_csv(RUN_ROOT/"model_comparison.csv"))
print((RUN_ROOT/"comparison_report.md").read_text())
from IPython.display import display, Image as DisplayImage
for architecture in ("A","B"):
    for task in ("root","leaf"):
        display(DisplayImage(filename=str(RUN_ROOT/architecture/task/"confusion_matrix.png")))
''')
    code('''# 8. Export checkpoints, OOF predictions and reports
from google.colab import files
archive=shutil.make_archive("/content/c3_experiment2_models","zip",root_dir=RUN_ROOT)
backup=RUN_ROOT.parent/(stamp+"_models.zip")
shutil.copyfile(archive,backup)
print("Drive backup:",backup)
files.download(archive)
''')
    md('''## Limits and inference handoff

Checkpoints: `{A,B}/{root,leaf}/fold_{1,2,3}.pth`, each with full ResNet18 state,
class mapping, preprocessing and dataset/split provenance. These are fold models;
no full-data final_model.pth is selected by outer scores. Original Experiment 1
artifacts are not used or modified.

The state-dict structure matches C3 inference, but the current general-quality
loader uses a hardcoded path and a different resize transform. Do not replace
its checkpoint. Future feature inference must use the task-specific path/classes
and exact metadata preprocessing (EXIF + contain/padding + normalization).

Minority support is small; corrected P07 has no unacceptable roots, P05 no One
Side leaves and P08 no data. Missing-class per-plant metrics are unavailable.
Previously reviewed outer data is a comparative benchmark, not a fresh external
test. Branch qualification/visibility issues remain; no labels are altered or
test examples excluded. Colab disconnects preserve Drive checkpoints; resume is
not implemented, so start a new run directory after interruption.
''')
    for i,cell in enumerate(cells):
        cell['id']=f'c3-exp2-{i:02d}'
        if cell['cell_type']=='code':compile(''.join(cell['source']),'colab_cell','exec')
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'},'accelerator':'GPU','colab':{'gpuType':'T4','provenance':[]}},'cells':cells}


def main():
    out=ROOT/'data/quality_assessment/processed/experiment2/colab_support'
    out.mkdir(parents=True,exist_ok=True)
    archive=out/'c3_experiment2_support.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('services/__init__.py','')
        z.writestr('services/c3_quality_assessment/__init__.py','')
        z.write(ROOT/'services/c3_quality_assessment/experiment2_training.py','services/c3_quality_assessment/experiment2_training.py')
    with archive.open('rb') as f:support_hash=hashlib.file_digest(f,'sha256').hexdigest()
    data=ROOT/'data/quality_assessment/processed/experiment2/v2_20261022/c3_features_v2_20261022.zip'
    with data.open('rb') as f:data_hash=hashlib.file_digest(f,'sha256').hexdigest()
    archive.with_suffix('.zip.sha256').write_text(support_hash+'  '+archive.name+'\n')
    target=ROOT/'services/c3_quality_assessment/notebooks/c3_experiment2_colab.ipynb'
    target.write_text(json.dumps(notebook(data_hash,support_hash),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(target);print(archive)


if __name__=='__main__':main()
