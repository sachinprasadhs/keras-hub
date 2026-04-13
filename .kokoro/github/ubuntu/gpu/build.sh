set -e

export KAGGLE_KEY="$(cat ${KOKORO_KEYSTORE_DIR}/73361_keras_kaggle_secret_key)"
export KAGGLE_USERNAME="$(cat ${KOKORO_KEYSTORE_DIR}/73361_keras_kaggle_username)"

if [[ -z "${KAGGLE_KEY}" ]]; then
   echo "KAGGLE_KEY is NOT set"
   exit 1
fi

if [[ -z "${KAGGLE_USERNAME}" ]]; then
   echo "KAGGLE_USERNAME is NOT set"
   exit 1
fi

set -x
cd "${KOKORO_ROOT}/"

export DEBIAN_FRONTEND=noninteractive
cd "${KOKORO_ROOT}/"

PYTHON_BINARY="/usr/bin/python3.10"

"${PYTHON_BINARY}" -m venv venv
source venv/bin/activate
# Check the python version
python --version
python3 --version

# setting the LD_LIBRARY_PATH manually is causing segmentation fault
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:"
# Check cuda
nvidia-smi
nvcc --version

cd "src/github/keras-hub"
pip install -U pip setuptools psutil

if [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip install -r requirements-tensorflow-cuda.txt --progress-bar off \
      --timeout 1000

elif [ "$KERAS_BACKEND" == "jax" ]
then
   echo "JAX backend detected."
   pip install -r requirements-jax-cuda.txt --progress-bar off --timeout 1000

elif [ "$KERAS_BACKEND" == "torch" ]
then
   echo "PyTorch backend detected."
   pip install -r requirements-torch-cuda.txt --progress-bar off --timeout 1000
fi


# Temporarily relax Config to allow 3.10 for GPU tests.
sed -i 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' pyproject.toml

pip install --no-deps -e "." --progress-bar off
pip install huggingface_hub

# Detect changed files
echo "Detecting changed files..."
# Fetch master branch to compare against
git fetch origin master --depth=1 || echo "Failed to fetch origin master, falling back to all tests."

CHANGED_FILES=$(git diff --name-only origin/master...HEAD || true)
echo "Changed files: $CHANGED_FILES"

CHANGED_MODELS=$(echo "$CHANGED_FILES" | grep 'keras_hub/src/models/' | cut -d/ -f1-4 | sort | uniq || true)
echo "Changed models: $CHANGED_MODELS"

TEST_TARGET="keras_hub"

# Check for non-model changes that require full tests
# We ignore API files, tools, and converters as they are usually model-specific or registration
NON_MODEL_CHANGES=$(echo "$CHANGED_FILES" | \
  grep -v 'keras_hub/src/models/' | \
  grep -v 'keras_hub/api/' | \
  grep -v 'tools/' | \
  grep -v 'convert_.*\.py$' | \
  grep -v '\.md$' || true)

if [[ -n "$CHANGED_MODELS" && -z "$NON_MODEL_CHANGES" ]]; then
  echo "Only model files (and safe whitelisted files) changed. Targeting specific models."
  TEST_TARGET=""
  for model_dir in $CHANGED_MODELS; do
    if [ -d "$model_dir" ]; then
      TEST_TARGET="$TEST_TARGET $model_dir"
    fi
  done
  # If no valid directories found, fallback to all
  if [[ -z "$TEST_TARGET" ]]; then
    TEST_TARGET="keras_hub"
  fi
else
  echo "Non-model files changed or fallback triggered. Running all tests."
  TEST_TARGET="keras_hub"
fi

# Check for doc-only changes to skip tests entirely
NON_DOC_CHANGES=$(echo "$CHANGED_FILES" | grep -v '\.md$' || true)
if [[ -z "$NON_DOC_CHANGES" && -n "$CHANGED_FILES" ]]; then
  echo "Only doc fixes detected. Skipping tests."
  exit 0
fi

# Run Tests
PYTEST_ARGS="--check_gpu --run_large"
if [ "${RUN_XLARGE:-0}" == "1" ]
then
   PYTEST_ARGS="$PYTEST_ARGS --run_extra_large"
fi

echo "Running: pytest $TEST_TARGET $PYTEST_ARGS --cov=keras_hub"
pytest $TEST_TARGET $PYTEST_ARGS --cov=keras_hub
