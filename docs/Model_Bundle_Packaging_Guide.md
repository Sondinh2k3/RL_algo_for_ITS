# Huong dan tao Sim Model Bundle `.sim.zip`

Muc tieu cua guide nay: tu mot model da train trong project MGMQ/RL, tao ra
mot file `*.sim.zip` de upload len MinIO. Sau do backend/ai-ops se tu xu ly
real topology, `simToReal`, `deployment_map.json`, compatibility check va
runtime bundle.

Guide nay khong tao runtime bundle truc tiep.

## 1. Phuong an nen dung

Voi pipeline backend hien tai, training team chi nen ban giao **sim bundle**:

```text
<network>-<version>.sim.zip
```

Trong ZIP nen co cac file logic bat buoc:

```text
sim_bundle_manifest.json
sim_network.json
policy.onnx
policy_meta.json
```

Backend/ai-ops se lam cac viec con lai:

- Lay real topology snapshot tu Core/backend.
- Lay mapping `simToReal` da duoc operator/integration xac nhan.
- Tao `deployment_map.json`.
- Validate compatibility.
- Ghi `compatibility_report.json`.
- Compose runtime bundle.
- Activate bundle sau review.

Khong nen upload RLlib checkpoint truc tiep len MinIO. Checkpoint nhu
`checkpoint_000031/algorithm_state.pkl` chi la artifact training. Truoc khi
tao `.sim.zip`, can export checkpoint thanh:

```text
policy.onnx
policy_meta.json
```

## 2. Khac voi runtime bundle

Khong dung flow nay cho sim bundle:

```bash
build-bundle v2 --deployment-map ... --output-zip ...
```

Lenh `build-bundle v2` trong `bundle-tooling/` dang tao bundle gan voi real IDs:

```text
model_manifest.json
network.json
deployment_map.json
feature_formula.json
intersections/cross_<real_cross_id>.json
```

Do la runtime/operator bundle path. Pipeline backend moi cua ban can
`*.sim.zip`, nen training side khong nen tao `deployment_map.json` va khong nen
map sang `real_cross_id` o buoc nay.

## 3. Chon training run

Vi du voi run Le Duc Tho hien tai:

```bash
export RUN_DIR="results_mgmq/mgmq_ppo_leductho_20260607_113716"
export TRIAL_DIR="$RUN_DIR/MGMQPPO_sumo_mgmq_v0_9171d_00000_0_2026-06-07_11-37-19"
export CHECKPOINT="$TRIAL_DIR/checkpoint_000031"
export SIM_CONFIG="network/leductho/intersection_config.json"
export TRAINING_CONFIG="$RUN_DIR/mgmq_training_config.json"
export NORMALIZER_STATE="$RUN_DIR/normalizer_state.json"
export NETWORK_ID="leductho"
export TENANT_ID="hcm_pilot"
export VERSION="v2026.06.07-leductho-ppo-001"
```

Chon checkpoint dua tren evaluation/metric tot nhat. Khong bat buoc dung
checkpoint cuoi.

## 4. Export policy artifacts

Tao thu muc export:

```bash
export EXPORT_DIR="$RUN_DIR/exported_policy"
mkdir -p "$EXPORT_DIR"
```

Lenh tren chi tao folder. Hai dong duoi day la **duong dan file can co sau khi
export**, khong phai command de go trong terminal:

```text
$EXPORT_DIR/policy.onnx
$EXPORT_DIR/policy_meta.json
```

Neu go truc tiep `$EXPORT_DIR/policy.onnx`, shell se hieu la ban dang muon
chay file `policy.onnx` nhu mot chuong trinh, nen se bao:

```text
bash: .../policy.onnx: No such file or directory
```

Dieu do co nghia la model **chua duoc export** sang ONNX/meta, khong phai loi
o buoc tao folder.

Sau khi chay exporter thanh cong, kiem tra bang:

```bash
ls -lh "$EXPORT_DIR/policy.onnx" "$EXPORT_DIR/policy_meta.json"
```

Neu `ls` van bao `No such file or directory`, dung tiep o day va export
checkpoint truoc. Sim bundle khong the dong goi neu thieu mot trong hai file
nay.

### 4.1. Export tu RLlib checkpoint

Project co script:

```text
scripts/export_mgmq_policy.py
```

Script nay se:

- Load RLlib checkpoint.
- Register custom MGMQ model va masked action distributions.
- Restore rieng policy bang `Policy.from_checkpoint()`, khong start SUMO env.
- Goi RLlib `policy.export_model(..., onnx=<opset>)`.
- Copy output ONNX thanh `$EXPORT_DIR/policy.onnx`.
- Sinh `$EXPORT_DIR/policy_meta.json` tu `mgmq_training_config.json`.

Truoc khi export, cai dependency ONNX neu venv chua co:

```bash
python -m pip install onnx
```

Chay export:

```bash
python scripts/export_mgmq_policy.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXPORT_DIR" \
  --training-config "$RUN_DIR/mgmq_training_config.json" \
  --network-id "$NETWORK_ID" \
  --version "$VERSION" \
  --opset 17
```

Kiem tra output:

```bash
ls -lh "$EXPORT_DIR/policy.onnx" "$EXPORT_DIR/policy_meta.json"
python -m json.tool "$EXPORT_DIR/policy_meta.json" | head -80
```

Neu script bao thieu package `onnx`, cai bang lenh `python -m pip install onnx`
trong dung virtualenv `(.venv)` roi chay lai.

Neu thay log dang sau, day la fallback hop le tren PyTorch moi:

```text
[export] RLlib ONNX export failed; falling back to clean LocalMGMQ exporter.
```

Nguyen nhan la RLlib generic exporter con dung `dynamic_axes`, trong khi
PyTorch ONNX exporter moi co the doi `dynamic_shapes`. Script se tu dong export
lai bang wrapper rieng cho local MGMQ va van tao:

```text
$EXPORT_DIR/policy.onnx
$EXPORT_DIR/policy_meta.json
```

Neu script fail khi load checkpoint, thuong la do checkpoint khong khop code
hien tai hoac thieu custom model/action distribution. Khi do can dung dung
commit/code da train checkpoint hoac sua script export theo model version do.
Script khong can env `sumo_mgmq_v0`; neu van gap loi env, kiem tra lai dang
chay dung version moi cua `scripts/export_mgmq_policy.py`.

`policy_meta.json` nen co du thong tin de runtime build input tensor va decode
output. Format chuan giong bundle mau cologne3:

```json
{
  "use_local_gnn": true,
  "obs_dim": 56,
  "base_obs_dim": 56,
  "window_size": 1,
  "max_neighbors": 4,
  "num_standard_phases": 8,
  "num_actions_per_phase": 7,
  "keep_action_index": 3,
  "input_names": [
    "self_features",
    "neighbor_features",
    "neighbor_mask",
    "neighbor_directions",
    "action_mask"
  ],
  "output_name": "logits",
  "checkpoint": "results_mgmq/.../checkpoint_000031"
}
```

Voi run Le Duc Tho trong `result.json`, cac gia tri dang thay:

- `action_mode`: `discrete_adjustment`
- `num_actions_per_phase`: `7`
- `keep_action_index`: `3`
- `base_obs_dim`: `56`
- `max_neighbors`: `4`
- `window_size`: `1`
- `num_agents`: `3`

Khong dua `normalizer_state.json` vao `.sim.zip` chuan nay, vi bundle mau
cologne3 chi co 4 file root.

## 5. Tao staging folder cho `.sim.zip`

Khuyen nghi dung layout:

```text
dist/sim-bundles/staging/<network>-<version>/
  sim_bundle_manifest.json
  sim_network.json
  policy.onnx
  policy_meta.json
```

Trong do:

- `sim_network.json`: sim topology de ai-ops compose voi real snapshot. File nay
  la canonical copy cua `network/leductho/intersection_config.json`.
- `policy.onnx`: model ONNX da export.
- `policy_meta.json`: metadata cua policy theo format backend dang doc.
- `sim_bundle_manifest.json`: manifest cua sim bundle.

## 6. Lenh tao sim bundle
 Chay tu root
project.

### 6.1. Config mot lan

```bash
export RUN_DIR="results_mgmq/mgmq_ppo_leductho_20260607_113716"
export TRIAL_DIR="$RUN_DIR/MGMQPPO_sumo_mgmq_v0_9171d_00000_0_2026-06-07_11-37-19"
export CHECKPOINT="$TRIAL_DIR/checkpoint_000031"
export EXPORT_DIR="$RUN_DIR/exported_policy"
export SIM_CONFIG="network/leductho/intersection_config.json"
export NETWORK_ID="leductho"
export TENANT_ID="hcm_pilot"
export VERSION="v2026.06.07-leductho-ppo-001"

export OUT_ZIP="dist/sim-bundles/${NETWORK_ID}-${VERSION}.sim.zip"
```

Neu dong goi model/network khac, thuong chi can doi:

- `RUN_DIR`
- `TRIAL_DIR`
- `CHECKPOINT`
- `SIM_CONFIG`
- `NETWORK_ID`
- `TENANT_ID`
- `VERSION`

### 6.2. Kiem tra policy da export

```bash
ls -lh "$EXPORT_DIR/policy.onnx" "$EXPORT_DIR/policy_meta.json"
```

Neu lenh tren bao `No such file or directory`, quay lai buoc 4.1 de export
policy truoc.

### 6.3. Build `.sim.zip`

```bash
python scripts/build_sim_bundle.py \
  --policy-onnx "$EXPORT_DIR/policy.onnx" \
  --policy-meta "$EXPORT_DIR/policy_meta.json" \
  --sim-config "$SIM_CONFIG" \
  --tenant-id "$TENANT_ID" \
  --network-id "$NETWORK_ID" \
  --version "$VERSION" \
  --training-run-id "$(basename "$RUN_DIR")" \
  --output-zip "$OUT_ZIP" \
  --clean
```

Lenh nay tu dong:

- Tao staging folder.
- Copy `policy.onnx`, `policy_meta.json`, sim config.
- Tao `sim_bundle_manifest.json`.
- Dong file `$OUT_ZIP`.

## 7. Kiem tra `.sim.zip`

Liet ke file trong ZIP:

```bash
unzip -l "$OUT_ZIP"
```

Kiem tra nhanh manifest:

```bash
python3 - <<'PY'
import json
import os
import zipfile

with zipfile.ZipFile(os.environ["OUT_ZIP"]) as zf:
    names = set(zf.namelist())
    manifest = json.loads(zf.read("sim_bundle_manifest.json"))

required = {"sim_bundle_manifest.json", "policy.onnx", "policy_meta.json", "sim_network.json"}
missing = sorted(required - names)

print(json.dumps({
    "missing": missing,
    "sim_bundle_id": manifest.get("sim_bundle_id"),
    "tenant_id": manifest.get("tenant_id"),
    "network_id": manifest.get("network_id"),
    "version": manifest.get("version"),
    "schema_version": manifest.get("schema_version"),
}, indent=2))

if missing:
    raise SystemExit(1)
PY
```

Neu output co `"missing": []` la OK. Zip chuan se chi co 4 file root:

```text
sim_bundle_manifest.json
sim_network.json
policy.onnx
policy_meta.json
```

## 8. Upload len MinIO

Dung MinIO client `mc`:

```bash
mc alias set local-minio http://localhost:9000 "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
mc cp "$OUT_ZIP" \
  "local-minio/traffic-rl-sim-bundles/${TENANT_ID}/${NETWORK_ID}/${VERSION}.sim.zip"
```

Hoac dung AWS CLI voi endpoint MinIO:

```bash
aws --endpoint-url http://localhost:9000 s3 cp "$OUT_ZIP" \
  "s3://traffic-rl-sim-bundles/${TENANT_ID}/${NETWORK_ID}/${VERSION}.sim.zip"
```

URI ban giao cho backend/ai-ops:

```text
s3://traffic-rl-sim-bundles/hcm_pilot/leductho/v2026.06.07-leductho-ppo-001.sim.zip
```

Neu service co endpoint pull sim bundle, goi endpoint backend theo contract cua
service do. Khong goi endpoint activate runtime bundle truc tiep truoc khi
ai-ops compose va review compatibility.

## 9. Checklist truoc khi upload

- `policy.onnx` ton tai va export tu dung checkpoint.
- `policy_meta.json` khop model/action mode runtime support.
- `sim_bundle_manifest.json` co field `sim_bundle_id`, `tenant_id`, `network_id`, `version`.
- `tenant_id`, `network_id`, `version` khop voi backend config.
- `sim_network.json` la sim topology cua dung network da train.
- ZIP co cac file nam o root, khong bi boc them prefix staging folder.
- Version MinIO la immutable, khong overwrite model cu.

## 10. Loi thuong gap

- Backend khong tim thay manifest: file trong ZIP bi nam duoi prefix
  `dist/sim-bundles/staging/...`; zip chuan phai co 4 file root.
- Backend khong compose duoc vi `networkId` khong khop real snapshot: sua
  `NETWORK_ID` va manifest.
- Runtime inference sai shape: xem lai `policy_meta.json`, dac biet
  `input_names`, `output_names`, `obs_dim`, `window_size`, `max_neighbors`,
  `action_mode`.
- Backend bao sim cross chua map: khong sua sim bundle truoc; can cung cap
  `simToReal` mapping/real snapshot cho backend.
