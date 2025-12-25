TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/data/data/wlb/ReCogDrive_github/dataset/maps"
export NAVSIM_EXP_ROOT="/mnt/data/data/wlb/ReCogDrive_github/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/data/data/wlb/ReCogDrive_github"
export OPENSCENE_DATA_ROOT="/mnt/data/data/wlb/ReCogDrive_github/dataset"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH