HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python src/train.py \
  name=puffin_K64_v2 \
  callbacks.model_checkpoint.dirpath=ismb26/models/puffin_K64_v2 \
  encoder=puffin \
  encoder.gnn_type=GAT \
  encoder.hidden_dim=512 \
  encoder.num_clusters=64 \
  encoder.num_res_gnn_layers=2 \
  encoder.num_seg_gnn_layers=2 \
  encoder.use_seg_res_cross_attn=false \
  encoder.proj_layer=true \
  encoder.esm_embed_dim=512 \
  encoder.input_feat_dim=512 \
  encoder.fuse_lm_method=sum \
  objective_type=dual \
  function_weight=1.0 \
  unit_weight=0.5 \
  entropy_weight=0.0 \
  mutual_weight=0.0

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python src/train.py \
  name=puffin_K32_v2 \
  callbacks.model_checkpoint.dirpath=ismb26/models/puffin_K32_v2 \
  encoder=puffin \
  encoder.gnn_type=GAT \
  encoder.hidden_dim=512 \
  encoder.num_clusters=32 \
  encoder.num_res_gnn_layers=2 \
  encoder.num_seg_gnn_layers=2 \
  encoder.use_seg_res_cross_attn=false \
  encoder.proj_layer=true \
  encoder.esm_embed_dim=512 \
  encoder.input_feat_dim=512 \
  encoder.fuse_lm_method=sum \
  objective_type=dual \
  function_weight=1.0 \
  unit_weight=0.5 \
  entropy_weight=0.0 \
  mutual_weight=0.0


HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python src/train.py \
  name=puffin_K16_v2 \
  callbacks.model_checkpoint.dirpath=ismb26/models/puffin_K16_v2 \
  encoder=puffin \
  encoder.gnn_type=GAT \
  encoder.hidden_dim=512 \
  encoder.num_clusters=16 \
  encoder.num_res_gnn_layers=2 \
  encoder.num_seg_gnn_layers=2 \
  encoder.use_seg_res_cross_attn=false \
  encoder.proj_layer=true \
  encoder.esm_embed_dim=512 \
  encoder.input_feat_dim=512 \
  encoder.fuse_lm_method=sum \
  objective_type=dual \
  function_weight=1.0 \
  unit_weight=0.5 \
  entropy_weight=0.0 \
  mutual_weight=0.0

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python src/train.py \
  name=puffin_K128_v2 \
  callbacks.model_checkpoint.dirpath=ismb26/models/puffin_K128_v2 \
  encoder=puffin \
  encoder.gnn_type=GAT \
  encoder.hidden_dim=512 \
  encoder.num_clusters=128 \
  encoder.num_res_gnn_layers=2 \
  encoder.num_seg_gnn_layers=2 \
  encoder.use_seg_res_cross_attn=false \
  encoder.proj_layer=true \
  encoder.esm_embed_dim=512 \
  encoder.input_feat_dim=512 \
  encoder.fuse_lm_method=sum \
  objective_type=dual \
  function_weight=1.0 \
  unit_weight=0.5 \
  entropy_weight=0.0 \
  mutual_weight=0.0


HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python src/train.py \
  name=mincut_K64 \
  callbacks.model_checkpoint.dirpath=ismb26/models/mincut_K64 \
  encoder=puffin \
  encoder.gnn_type=GAT \
  encoder.hidden_dim=512 \
  encoder.num_clusters=64 \
  encoder.num_res_gnn_layers=2 \
  encoder.num_seg_gnn_layers=2 \
  encoder.use_seg_res_cross_attn=false \
  encoder.proj_layer=true \
  encoder.esm_embed_dim=512 \
  encoder.input_feat_dim=512 \
  encoder.fuse_lm_method=sum \
  objective_type=dual \
  function_weight=0.0 \
  unit_weight=0.5 \
  entropy_weight=0.0 \
  mutual_weight=0.0 



HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python src/train.py \
  name=mincut_K32 \
  callbacks.model_checkpoint.dirpath=ismb26/models/mincut_K32 \
  encoder=puffin \
  encoder.gnn_type=GAT \
  encoder.hidden_dim=512 \
  encoder.num_clusters=32 \
  encoder.num_res_gnn_layers=2 \
  encoder.num_seg_gnn_layers=2 \
  encoder.use_seg_res_cross_attn=false \
  encoder.proj_layer=true \
  encoder.esm_embed_dim=512 \
  encoder.input_feat_dim=512 \
  encoder.fuse_lm_method=sum \
  objective_type=dual \
  function_weight=0.0 \
  unit_weight=0.5 \
  entropy_weight=0.0 \
  mutual_weight=0.0 \
  +schedule.warmup_epochs=0 \
  +schedule.ramp_epochs=0

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python src/train.py \
  name=mincut_K16 \
  callbacks.model_checkpoint.dirpath=ismb26/models/mincut_K16 \
  encoder=puffin \
  encoder.gnn_type=GAT \
  encoder.hidden_dim=512 \
  encoder.num_clusters=16 \
  encoder.num_res_gnn_layers=2 \
  encoder.num_seg_gnn_layers=2 \
  encoder.use_seg_res_cross_attn=false \
  encoder.proj_layer=true \
  encoder.esm_embed_dim=512 \
  encoder.input_feat_dim=512 \
  encoder.fuse_lm_method=sum \
  objective_type=dual \
  function_weight=0.0 \
  unit_weight=1.0 \
  entropy_weight=0.0 \
  mutual_weight=0.0 \
  +schedule.warmup_epochs=0 \
  +schedule.ramp_epochs=0

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python src/train.py \
  name=mincut_K128 \
  callbacks.model_checkpoint.dirpath=ismb26/models/mincut_K128 \
  encoder=puffin \
  encoder.gnn_type=GAT \
  encoder.hidden_dim=512 \
  encoder.num_clusters=128 \
  encoder.num_res_gnn_layers=2 \
  encoder.num_seg_gnn_layers=2 \
  encoder.use_seg_res_cross_attn=false \
  encoder.proj_layer=true \
  encoder.esm_embed_dim=512 \
  encoder.input_feat_dim=512 \
  encoder.fuse_lm_method=sum \
  objective_type=dual \
  function_weight=0.0 \
  unit_weight=1.0 \
  entropy_weight=0.0 \
  mutual_weight=0.0 \
  +schedule.warmup_epochs=0 \
  +schedule.ramp_epochs=0

v1: initial weight  unit_weight=0.1 , warmup epochs 3 epoch , ramp epochs 2 epoch # hardcoded in dual_model.py
v2: initial weight  unit_weight=0.0 , warmup epochs 5 epoch , ramp epochs 5 epoch # hardcoded in dual_model.py
mincut: warm up epochs 0 , ramp epochs 0 # hardcoded in dual_model.py