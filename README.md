# Quantifying Hippocampus Volume for Alzheimer's Progression

### Background
Alzheimer's disease (AD) is a progressive neurodegenerative disorder that results in impaired neuronal (brain cell) function and eventually, cell death. AD is the most common cause of dementia. Clinically, it is characterized by memory loss, inability to learn new material, loss of language function, and other manifestations.

For patients exhibiting early symptoms, quantifying disease progression over time can help direct therapy and disease management.

A radiological study via MRI exam is currently one of the most advanced methods to quantify the disease. In particular, the measurement of hippocampal volume has proven useful to diagnose and track progression in several brain disorders, most notably in AD. Studies have shown a reduced volume of the hippocampus in patients with AD.

The hippocampus is a critical structure of the human brain (and the brain of other vertebrates) that plays important roles in the consolidation of information from short-term memory to long-term memory. In other words, the hippocampus is thought to be responsible for memory and learning (that's why we are all here, after all!)

![hippocampus-small](https://github.com/hungryPanko/3D_CV_hippocampal_volume/assets/101350589/fd6bc96c-858c-4b91-a8ac-20eaa183f11d)
Humans have two hippocampi, one in each hemisphere of the brain. They are located in the medial temporal lobe of the brain.

According to Nobis et al., 2019, the volume of hippocampus varies in a population, depending on various parameters, within certain boundaries, and it is possible to identify a "normal" range taking into account age, sex and brain hemisphere.

You can see this in the image below where the right hippocampal volume of women across ages 52 - 71 is shown.
![Screenshot 2023-08-24 at 10 45 54 AM](https://github.com/hungryPanko/3D_CV_hippocampal_volume/assets/101350589/fba79b62-e6bf-4020-9752-333d446965fc)

There is one problem with measuring the volume of the hippocampus using MRI scans, though - namely, the process tends to be quite tedious since every slice of the 3D volume needs to be analyzed, and the shape of the structure needs to be traced. The fact that the hippocampus has a non-uniform shape only makes it more challenging. Do you think you could spot the hippocampi in this axial slice below?

![Screenshot 2023-08-24 at 10 46 26 AM](https://github.com/hungryPanko/3D_CV_hippocampal_volume/assets/101350589/97216bb9-6558-475c-97d7-34a22364ec21)
In this project, we will focus on the technical aspects of building a segmentation model and integrating it into the clinician's workflow, leaving the dataset curation and model validation questions largely outside the scope of this project.

### Project Overview
This project has end-to-end AI system which features a machine learning algorithm that integrates into a clinical-grade viewer and automatically measures hippocampal volumes of new patients, as their studies are committed to the clinical imaging archive.

Fortunately we don't need to deal with full heads of patients. Our (fictional) radiology department runs a HippoCrop tool which cuts out a rectangular portion of a brain scan from every image series, making your job a bit easier, and our committed radiologists have collected and annotated a dataset of relevant volumes, and even converted them to NIFTI format!

The dataset that contains the segmentations of the right hippocampus and you will use the U-Net architecture to build the segmentation model.

### The Dataset
We are using the "Hippocampus" dataset from the Medical Decathlon competition. This dataset is stored as a collection of NIFTI files, with one file per volume, and one file per corresponding segmentation mask. The original images here are T2 MRI scans of the full brain. As noted, in this dataset we are using cropped volumes where only the region around the hippocampus has been cut out. This makes the size of our dataset quite a bit smaller, our machine learning problem a bit simpler and allows us to have reasonable training times. You should not think of it as "toy" problem, though. Algorithms that crop rectangular regions of interest are quite common in medical imaging. Segmentation is still hard.

### Local Environment
PyTorch (preferably with CUDA)
nibabel
matplotlib
numpy
pydicom
Pillow (should be installed with pytorch)
tensorboard
