#!/bin/bash
###############################################################################
# 공용 DDP 학습 템플릿 — DGIST iREMB 슈퍼컴퓨터
#
# 사용법 (로그인 노드에서):
#   1. 이 파일을 프로젝트 폴더에 복사:
#        cp slurm/template_ddp.sh /scratch/mip25/wonbinlee/MyProject/batch_ddp.sh
#   2. 아래 "여기만 수정" 블록 + #SBATCH 값을 채움
#   3. 제출:
#        cd /scratch/mip25/wonbinlee/MyProject
#        sbatch batch_ddp.sh
#
# 작업 확인:
#   /opt/scripts/alljob
#   tail -f logs/<jobdir>/out.log
#   scancel <JOB_ID>
###############################################################################

# ── 안전장치 ──────────────────────────────────────────────────────
# set -e  : 명령이 하나라도 실패하면 스크립트 즉시 중단
# set -u  : 정의 안 된 변수를 쓰면 에러 (오타 방지)
# set -o pipefail : 파이프(|) 중 앞쪽 명령이 실패해도 감지


# ── Slurm 리소스 설정 ────────────────────────────────────────────
#SBATCH --job-name=CHANGE_ME
#    → 작업 이름. squeue/alljob에서 이 이름으로 보임
#SBATCH --partition=l40sq
#    → 파티션 선택. L40s GPU → l40sq, H200 GPU → h200q
#SBATCH --nodes=1
#    → 사용할 노드 수. DDP 단일노드면 1
#SBATCH --ntasks-per-node=1
#    → 노드당 프로세스 수. torchrun이 알아서 GPU별 프로세스를 띄우므로 1
#SBATCH --cpus-per-task=8
#    → 태스크당 CPU 코어 수. DataLoader num_workers에 맞춰 설정
#SBATCH --gres=gpu:1
#    → 노드당 GPU 수. DDP로 2장 사용
#SBATCH --mem=64G
#    → 메모리 요청량. 가이드: (전체메모리 / 전체GPU) × 사용GPU
#    → L40s 노드는 1TB/4GPU = 250G/GPU, 2GPU면 최대 500G까지 가능
#SBATCH --time=240:00:00
#    → 최대 실행 시간. 초과하면 자동 kill됨
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --exclude=iREMB-C-07,iREMB-C-09
#    → Slurm 기본 로그는 /dev/null로 버리고, 아래 exec으로 직접 관리
#    → 주의: 모듈 로드 등 exec 이전 에러는 이 설정 때문에 안 보임
#    →       디버깅 시 아래처럼 바꾸면 Slurm 로그도 남음:
#    →       #SBATCH --output=slurm-%x.%J.out
#    →       #SBATCH --error=slurm-%x.%J.err

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# === 여기만 수정 ===
# ═══════════════════════════════════════════════════════════════════

PROJECT_DIR=""
#    → TODO: 프로젝트 소스코드가 있는 디렉토리
#    → 예: /scratch/mip25/wonbinlee/4DGaussians
#    → 아래 cd로 이동 후, 상대경로(./data 등)의 기준이 됨

SIF_IMAGE=""
#    → TODO: Singularity 컨테이너 이미지 경로
#    → 예: /home/mip25/scratch/wonbinlee/pytorch.sif

EXPERIMENT_NAME=""
#    → TODO: 실험 이름. 로그 폴더명에 사용됨
#    → 예: dnerf/bouncingballs

NUM_GPUS=1
#    → DDP에서 사용할 GPU 수. #SBATCH --gres=gpu:N 과 반드시 일치시킬 것

CONDA_ENV=""
#    → conda 환경 이름. 없으면 빈 문자열 "" 그대로 두기
#    → 예: se3

# 컨테이너 안에서 실행할 학습 명령 (여러 줄은 \ 로 이어쓰기)
# 상대경로(./data 등)는 PROJECT_DIR 기준으로 해석됨
TRAIN_CMD=""
#    → TODO: 실제 학습 명령으로 교체
#    → 예: python train.py -s data/dnerf/bouncingballs --configs arguments/dnerf/bouncingballs.py

# ═══════════════════════════════════════════════════════════════════
# === 아래는 수정하지 않아도 됨 ===
# ═══════════════════════════════════════════════════════════════════

# ── 입력값 검증 ──────────────────────────────────────────────────
if [[ -z "$PROJECT_DIR" ]]; then
    echo "ERROR: PROJECT_DIR를 설정하세요" >&2; exit 1
fi
if [[ -z "$SIF_IMAGE" ]]; then
    echo "ERROR: SIF_IMAGE를 설정하세요" >&2; exit 1
fi
if [[ -z "$TRAIN_CMD" ]]; then
    echo "ERROR: TRAIN_CMD를 설정하세요" >&2; exit 1
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "ERROR: PROJECT_DIR가 존재하지 않음: $PROJECT_DIR" >&2; exit 1
fi
if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "ERROR: SIF_IMAGE가 존재하지 않음: $SIF_IMAGE" >&2; exit 1
fi

# ── 로그 디렉토리 생성 ───────────────────────────────────────────
# SLURM_JOB_ID는 Slurm이 자동으로 넣어주는 환경변수 (제출 시 고유번호)
JOBDIR=${PROJECT_DIR}/logs/${EXPERIMENT_NAME}_${SLURM_JOB_ID}
mkdir -p "$JOBDIR"

# exec: 이 시점부터 모든 stdout → out.log, stderr → error.log로 리다이렉트
# 터미널에는 안 찍히지만, tail -f 로 실시간 확인 가능
exec 1>"$JOBDIR/out.log"
exec 2>"$JOBDIR/error.log"

# ── 모듈 로드 ────────────────────────────────────────────────────
module purge
#    → 기존에 로드된 모듈 전부 제거 (충돌 방지)
module load Singularity/4.3.4
#    → Singularity 컨테이너 런타임 로드

# ── Job 정보 출력 ────────────────────────────────────────────────
echo "=== Job Info ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "GPUs:       $NUM_GPUS"
echo "Project:    $PROJECT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "Log dir:    $JOBDIR"
echo "================"

# ── 작업 디렉토리 이동 ───────────────────────────────────────────
cd "$PROJECT_DIR"
#    → 여기서 cd한 후, 컨테이너 안에서도 이 경로가 작업 디렉토리가 됨
#    → train.py 안의 상대경로(예: ./data/dnerf/...)는 이 디렉토리 기준

# ── conda 활성화 명령 조립 ───────────────────────────────────────
if [[ -n "$CONDA_ENV" ]]; then
    CONDA_INIT="source /opt/conda/etc/profile.d/conda.sh && conda activate $CONDA_ENV &&"
else
    CONDA_INIT=""
fi

# ── 학습 실행 ────────────────────────────────────────────────────
# singularity exec --nv : GPU 드라이버를 컨테이너에서 볼 수 있게 함
# --bind A:B           : 호스트의 A 경로를 컨테이너 안에서 B로 마운트
#                        /scratch/mip25를 바인드해야 데이터/코드 접근 가능
# bash -c "..."        : 컨테이너 안에서 실행할 명령을 묶음

srun --mpi=pmix singularity exec --nv \
    --bind /scratch/mip25:/scratch/mip25 \
    --bind /home/mip25:/home/mip25 \
    "$SIF_IMAGE" \
    bash -c "
        ${CONDA_INIT}
        nvidia-smi
        ${TRAIN_CMD}
    " 2>&1 | tee "$JOBDIR/train.log"
#    → tee: 출력을 train.log에도 저장하면서 동시에 out.log에도 기록

echo "Exit code: $?"
echo "End: $(date)"
