
# :fire: ProMaC

Code release of paper:

**Leveraging Hallucinations to Reduce Manual Prompt
Dependency in Promptable Segmentation**



## Quick Start
<!-- The prompt-dialogue of varies abilities are saved in [dataset](https://github.com/crystraldo/StableLLAVA/tree/main/dataset). -->

<!-- The synthesized prompt-dialogue datasets of various abilities are saved in [dataset](https://github.com/crystraldo/StableLLAVA/tree/main/dataset). Please follow the steps below to generate datasets with LLaVA format. -->

<!-- 1. Use [SD-XL](https://github.com/crystraldo/StableLLAVA/blob/main/stable_diffusion.py) to generate images as training images. It will take ~13s to generate one image on V100.-->
<!-- python stable_diffusion.py --prompt_path dataset/animal.json --save_path train_set/animal/-->
<!-- 2. Use [data_to_llava](https://github.com/crystraldo/StableLLAVA/blob/main/data_to_llava.py) to convert dataset format for LLaVA model training. -->
<!-- ```
python data_to_llava.py --image_path train_set/ --prompt_path dataset/ --save_path train_ano/
``` -->

### Download Dataset
1. Download the datasets from the follow links:
   
**Camouflaged Object Detection Dataset**
- **[COD10K](https://github.com/DengPingFan/SINet/)**
- **[CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)**
- **[CHAMELEON](https://www.polsl.pl/rau6/datasets/)**
2. Put it in ./data/.
### Running GenSAM on CHAMELON Dataset with LLaVA1/LLaVA1.5
1. When playing with LLaVA, this code was implemented with Python 3.8 and PyTorch 2.1.0. You can install all the requirements via:
```bash
pip install -r requirements_llava.txt
```
2. Our GenSAM is a training-free test-time adaptation approach, so you can play with it by running:
```bash
python main.py --config config/CHAMELON.yaml
```
or
```bash
sh script_llava.sh
``` 
# ProMaC_code
