```markdown
# Automatic Segmentation with Deformation Model

This repository provides an automatic segmentation pipeline for aortic geometry, combining **LoGBNet** for voxel-wise segmentation and a **GNN-LDDMM** deformation model for improved surface alignment with CT image gradients.

## 🔹 Voxel Segmentation

For voxel segmentation, we use LoGBNet. Please refer to the original repository:  
➡️ https://github.com/adlsn/LoGBNet.git  
📄 Publication:  
An et al., *"Hierarchical LoG Bayesian Neural Network for Enhanced Aorta Segmentation,"* ISBI 2025.

## 🔹 GNN-LDDMM Deformation

For the GNN-LDDMM deformation model:

1. **Download the dataset** from Zenodo:  
   🔗 https://doi.org/10.5281/zenodo.16663170

2. **Also download the ZIP files** located in the `Data` folder of the following GitHub repository:  
   🔗 https://github.com/shanjierenyidp/pytorch-NURBS-implementation-for-Vascular-Surface  
   → Extract the ZIP files into the same `Data` folder, preserving original names and directory structure.

---

## 🔧 Usage Instructions

Run the following scripts in order:

- **Preprocess the predicted CT data:**
  ```bash
  python process_CT.py
  ```

- **Smooth the initial segmentation:**
  ```bash
  python GNN-LDDMM_presmooth.py
  ```

- **Deform the surface to improve alignment with image gradients:**
  ```bash
  python GNN-LDDMM_deform.py
  ```

- **Perform uncertainty quantification (10 realizations):**
  ```bash
  python GNN-LDDMM_deform_UQ.py
  ```

---

## 📄 License

Copyright © 2025 Du et al.
```
