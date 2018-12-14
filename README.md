# TensorFlow-Keras Unet

## Inria Aerial Images
The [Inria Aerial Image Labelling Dataset](https://project.inria.fr/aerialimagelabeling/) contains pixel labels from satellite images for two semantic classes: building and not building

### Get the data

```bash
wget https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001 &
wget https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002 &
wget https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003 &
wget https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004 &
wget https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005 &

```
Wait for all these jobs to complete

```bash
sudo apt install p7zip-full
7z x aerialimagelabeling.7z.001
unzip NEW2-AerialImageDataset.zip
rm -i aerialimagelabeling.7z.* 
rm -i NEW2-AerialImageDataset.zip
```
