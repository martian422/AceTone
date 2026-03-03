CUDA_VISIBLE_DEVICES=0,1 gunicorn "deqa_server:create_app()" \
    --bind 127.0.0.1:18087 \
    --workers 2 \
    --worker-class sync \
    --timeout 120