#!/bin/bash
# Final config sweep for fa4_hybrid_v2. Usage: bash sweep_v2.sh [sizes]
source /fsx/sampan/fa4_env.sh
cd /fsx/sampan/fa4_work
SIZES=${1:-16k,32k,64k,128k}
run() {  # run NAME STAGES STAGES_V HM KERNELS
  echo "##### $1 (K$2/V$3 HM=$4) #####"
  FLASH_ATTN_SM90_NUM_STAGES=$2 FLASH_ATTN_SM90_NUM_STAGES_V=$3 FLASH_ATTN_SM90_HEAD_MAJOR=$4 \
    timeout 2000 python bench_fine_v2.py $SIZES $5
}
run "baseline+v2 s4"      4 4 0 stock,v2
run "v2 s4 HM"            4 4 1 v2
run "v2 s4V2"             4 2 0 v2
run "v2 s4V2 HM"          4 2 1 v2
run "v2 s2 HM"            2 2 1 v2
