# HRGR: Enhancing Image Manipulation Detection via Hierarchical Region-aware Graph Reasoning (ICME2025)
**Authors: Xudong Wang, Jiaran Zhou, Huiyu Zhou , Junyu Dong, [Yuezun Li](https://yuezunli.github.io/) (corresponding author)**

## Usage

### Install

- Create a conda virtual environment and activate it:

```bash
conda create -n hrgr python=3.9 -y
conda activate hrgr
```

+ Install `torch==1.11` with `CUDA==11.3`:

```bash
conda install numpy==1.24.3 pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -y
```

- Install `timm==0.6.11` and `mmcv-full==1.5.3`:

```bash
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install timm==0.6.11 mmsegmentation==0.27.0
```

- Install other requirements:

```bash
pip install yapf==0.33.0 imgaug==0.4.0 scikit-image==0.22.0 scikit-learn==1.2.2 tensorboard==2.13.0
```

- Compile CUDA operators
```bash
cd ./ops_dcnv3 && python setup.py install && cd ../
cd ./ops_ssn && python setup.py install && cd ../
```
### Data Preparation


Download `data.zip` and `work_dirs.zip` to obtain the preprocessed datasets and trained weights form [Google Drive](https://drive.google.com/drive/folders/10UpyruBp7Lw-t4eIiH1CeCdITvW6Dgex?usp=sharing).
Extract them and place them in the root of this project.


### Evaluation

To evaluate our `HRGR`, run:

```bash
python test.py <config-file> <checkpoint> --eval mIoU --more_eval true
```

More examples can be found in `dist_test.sh`. Uncomment the corresponding content and run:

```bash
sh dist_test.sh
```

### Training

To train our `HRGR`, run:

```bash
python train.py <config-file>
```

More examples can be found in `dist_train.sh`. Uncomment the corresponding content and run:

```bash
sh dist_train.sh
```

## Citations
If HRGR helps your research or work, please kindly cite our paper. The following is a BibTeX reference.
```bibtex
@inproceedings{wang2025hrgr,
  title={HRGR: Enhancing Image Manipulation Detection via Hierarchical Region-aware Graph Reasoning},
  author={Wang, Xudong and Zhou, Jiaran and Zhou, Huiyu and Dong, Junyu and Li, Yuezun},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)},
  year={2025}
}
```