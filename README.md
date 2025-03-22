# Intelligent Column Chromatography Prediction Model

## Citation
If you use this work in your research, please cite:
```bibtex
@misc{wu2024intelligentchemicalpurificationtechnique,
      title={Intelligent Chemical Purification Technique Based on Machine Learning}, 
      author={Wenchao Wu and Hao Xu and Dongxiao Zhang and Fanyang Mo},
      year={2024},
      eprint={2404.09114},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.09114}, 
}
```

## Environment Configuration
**Python Version**: 3.9

### Core Dependencies
| Package | Version | Installation Command |
|---------|---------|----------------------|
| RDKit   | 2023.9.2 | `conda install -c conda-forge rdkit` |
| PyTorch | 2.1.0   | `pip install torch==2.1.0` |
| Mordred | 1.2.0   | `pip install mordred==1.2.0` |
| pandas  | 2.1.4   | `pip install pandas==2.1.4` |

### Recommended Installation
```bash
# Create conda environment
conda create -n chromatography python=3.9
conda activate chromatography

# Install core packages
conda install -c conda-forge rdkit==2023.9.2
pip install torch==2.1.0 pandas==2.1.4 mordred==1.2.0
```
