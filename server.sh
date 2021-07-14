MODEL_NAME="$(python -c 'import inference;print(inference.MODEL)')"

docker run -t --rm -p 8500:8500  \
   -v "$(pwd)/serving_model/${MODEL_NAME}:/models/${MODEL_NAME}" \
   -e MODEL_NAME=${MODEL_NAME}  tensorflow/serving:1.14.0
