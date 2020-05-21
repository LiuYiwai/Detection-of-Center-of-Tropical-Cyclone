# Detection-of-Center-of-Tropical-Cyclone
A simple implementation of Objective Detection of Center of Tropical Cyclone in Remotely Sensed Infrared Images(Neeru Jaiswal and C. M. Kishtawal)

## How to use

process:
```python
python process_data.py --input_dir [input_dir] --output_dir [output_dir] --batch [batch] \
                    --img_size [img_size] --var_kernel [var_kernel] --gradient_kernel [gradient_kernel] \
                    --smooth_kernel [smooth_kernel]
```

### default
  --input_dir       = './data'
  
  --output_dir      = './out'
  
  --batch           = 32
  
  --img_size        = 256
  
  --var_kernel      = 3
  
  --gradient_kernel = 5
  
  --smooth_kernel   = 3
