#!/usr/bin/env bash
# Build the Tau3 latest-VERL vLLM V1 image on P5 and push it to AWS ECR.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.tau3.vllm20.v1}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.20.0-cu129}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
ECR_REPOSITORY="${ECR_REPOSITORY:-tau3-verl-vllm20-v1}"
PLATFORM="${PLATFORM:-linux/amd64}"
TAU2_COMMIT="${TAU2_COMMIT:-220b47844fb74d4351037e81055cf1e2948e4734}"
BUILD_CONTEXT="${BUILD_CONTEXT:-$PROJECT_ROOT}"
ALLOW_DIRTY_IMAGE_BUILD="${ALLOW_DIRTY_IMAGE_BUILD:-0}"
RUN_IMAGE_PREFLIGHT="${RUN_IMAGE_PREFLIGHT:-1}"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Error: required command not found: $1" >&2
        exit 1
    }
}

require_cmd aws
require_cmd docker

GIT_AVAILABLE=0
if command -v git >/dev/null 2>&1 && [ -d "$PROJECT_ROOT/.git" ]; then
    GIT_AVAILABLE=1
fi

SOURCE_REVISION="${SOURCE_REVISION:-${GIT_COMMIT:-}}"
FULL_SOURCE_REVISION="${FULL_SOURCE_REVISION:-}"
if [ "$GIT_AVAILABLE" = "1" ]; then
    SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)}"
    FULL_SOURCE_REVISION="${FULL_SOURCE_REVISION:-$(git -C "$PROJECT_ROOT" rev-parse HEAD)}"
else
    SOURCE_REVISION="${SOURCE_REVISION:-s3sync-$(date -u +%Y%m%dT%H%M%SZ)}"
    FULL_SOURCE_REVISION="${FULL_SOURCE_REVISION:-$SOURCE_REVISION}"
    echo "Warning: Git metadata not available under $PROJECT_ROOT; using SOURCE_REVISION=$SOURCE_REVISION" >&2
fi

IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d)-${SOURCE_REVISION}}"
LOCAL_IMAGE="${LOCAL_IMAGE:-tau3-verl-vllm20-v1:${IMAGE_TAG}}"

if [ "$GIT_AVAILABLE" = "1" ] && [ "$ALLOW_DIRTY_IMAGE_BUILD" != "1" ] && [ -n "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "Error: tracked working tree changes are present. Commit/push before building a reproducible image." >&2
    echo "Set ALLOW_DIRTY_IMAGE_BUILD=1 only for throwaway debugging images." >&2
    git -C "$PROJECT_ROOT" status --short --untracked-files=no >&2
    exit 1
fi

ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")}"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "----------------------------------------------------------------"
echo "Building Tau3 vLLM V1 image"
echo "Project root: $PROJECT_ROOT"
echo "Dockerfile: $DOCKERFILE"
echo "Base image: $BASE_IMAGE"
echo "Platform: $PLATFORM"
echo "ECR repo: $ECR_REPOSITORY"
echo "AWS region: $AWS_REGION"
echo "Image URI: $IMAGE_URI"
echo "Source revision: $FULL_SOURCE_REVISION"
echo "----------------------------------------------------------------"

aws ecr describe-repositories \
    --repository-names "$ECR_REPOSITORY" \
    --region "$AWS_REGION" >/dev/null 2>&1 || \
aws ecr create-repository \
    --repository-name "$ECR_REPOSITORY" \
    --image-scanning-configuration scanOnPush=true \
    --region "$AWS_REGION" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REGISTRY"

docker build \
    --pull \
    --platform "$PLATFORM" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "TAU2_COMMIT=$TAU2_COMMIT" \
    -f "$PROJECT_ROOT/$DOCKERFILE" \
    -t "$LOCAL_IMAGE" \
    "$BUILD_CONTEXT"

if [ "$RUN_IMAGE_PREFLIGHT" = "1" ]; then
    docker run --rm --gpus all --net=host --ipc=host "$LOCAL_IMAGE" -lc \
        'cd /workspace/verl_tau3_sdpo && python scripts/p5_preflight_vllm_v1.py --skip-model-config'
fi

docker tag "$LOCAL_IMAGE" "$IMAGE_URI"
docker push "$IMAGE_URI"

IMAGE_DIGEST="$(aws ecr describe-images \
    --repository-name "$ECR_REPOSITORY" \
    --image-ids "imageTag=$IMAGE_TAG" \
    --query 'imageDetails[0].imageDigest' \
    --output text \
    --region "$AWS_REGION")"

cat <<EOF
{
  "image_uri": "$IMAGE_URI",
  "image_digest": "$IMAGE_DIGEST",
  "image_uri_by_digest": "${ECR_REGISTRY}/${ECR_REPOSITORY}@${IMAGE_DIGEST}",
  "local_image": "$LOCAL_IMAGE",
  "base_image": "$BASE_IMAGE",
  "aws_region": "$AWS_REGION",
  "ecr_repository": "$ECR_REPOSITORY",
  "source_revision": "$FULL_SOURCE_REVISION",
  "git_metadata_available": "$GIT_AVAILABLE"
}
EOF
