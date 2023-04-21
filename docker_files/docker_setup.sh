#!/bin/bash
# CUDA version from the command-line argument
command=${1:-build}
cuda_version=${2:-11.3}
torch_version=${3:-1.12.1}
final_tag=nimakondori/cpsc_533y_proj:torch_${torch_version}_cuda_${cuda_version}

echo "Final Tag == $final_tag"

if [[ "$command" == "build" ]]; then
	docker build --build-arg cuda_version=$cuda_version --build-arg torch_version=$torch_version --tag $final_tag .

	docker push $final_tag
elif [[ "$command" == "run" ]]; then  
	
	 docker run -it -d \
	          --gpus device=ALL \
     	      --name=nima_cpsc533y_project  \
      	      --volume=$HOME/workspace/repos/cpsc533y_project:/workspace/cpsc533y_project \
      	      --volume=$HOME/workspace/datasets/as_tom/:/mnt/data/ \
			  --shm-size 8G\
      	      $final_tag
else
	echo "invalid command. Use build or run"
fi 
