FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY material_review/requirements-inference.txt /tmp/requirements-inference.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-inference.txt

# Local model inference code and deterministic composition/validation code.
COPY model.py preprocessing.py utils.py get_predictions.py ensemble_predictions.py ./
COPY get_biomarker_kaggle.py build_experimental_submission.py \
     build_joint_exploration_submission.py \
     build_t2_veinskel10_or_submission.py compact_submission_zip.py ./
COPY material_review/pipeline.py material_review/weights_manifest.json \
     material_review/artifacts_manifest.json ./material_review/
COPY material_review/optic_disc_segformer.py ./material_review/
COPY material_review/entrypoint.sh /usr/local/bin/kirby-entrypoint
RUN chmod 0755 /usr/local/bin/kirby-entrypoint

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/kirby-entrypoint"]
CMD ["help"]
