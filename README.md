# s1_to_ndvi


Before to start, you need to set SNAP command line. Below is shown link for SNAP command line tutorial: 
```
http://step.esa.int/docs/tutorials/SNAP_CommandLine_Tutorial.pdf
```

For Linux, download the 64-bit unix installer from the STEP website (http://step.esa.int). 
```
$ chmod +x esa-snap_sentinel_unix_4_0.sh      
$ ./esa-snap_sentinel_unix_4_0.sh 
```
And then you need to specify gpt folder in the code. 

For processing S2, sen2cor must be downloaded from http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-9/      
Also, you can follow this tutorial to preprocess S2 https://www.youtube.com/watch?v=1F37ScOrKqM      

Based on the paper "Crop NDVI Monitoring Based on Sentinel 1" by Filgueiras et al. (doi: 10.3390/rs11121441):
```
file:///tmp/mozilla_zhamilya0/remotesensing-11-01441.pdf
```
