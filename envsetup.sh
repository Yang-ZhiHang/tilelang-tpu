# Please ensure you have completed the `needed specified by user` part in the following lines.

# set up sophgo environment
export TPU_MLIR_PATH=/workspace/tpu-mlir # needed specified by user.
source $TPU_MLIR_PATH/envsetup.sh

# set up cross-compile toolchain
export TOOLCHAINS_PATH=/workspace/toolchains # needed specified by user.
if [ ! -d "$PPL_THIRD_PARTY_PATH/toolchains_dir" ]; then
    ln -s $TOOLCHAINS_PATH $PPL_THIRD_PARTY_PATH/toolchains_dir
    echo "[INFO] Created symbolic link $PPL_THIRD_PARTY_PATH/toolchains_dir -> $TOOLCHAINS_PATH"
else
    echo "[INFO] toolchains path $PPL_THIRD_PARTY_PATH/toolchains_dir already exists."
fi

# set up TPU kernel path (needed by main.so init)
source ./tpu_demo/env.sh

# set up python path
export PYTHONPATH=.
