TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/dataset/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/dataset"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH