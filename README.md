# Neural Speech Synchronization–Enhancement with Implicit Sampling Rate Offset Mitigation
Abstract-Distributed microphone array (DMA) or so-called wireless acoustic sensor network becomes increasingly popular in various speech interaction applications due to the superiority in sound acquisition capacity over conventional regularized microphone arrays. The challenge of using DMAs that heavily impacts the performance of downstream tasks lies in the asynchrony caused by sampling rate offsets (SROs) across devices, since distributed microphones are usually driven by independent clock oscillators. Existing SRO compensation methods typically follow a two-stage scheme, where the SRO is first explicitly estimated by statistical signal processing and then compensated by resampling in the time or frequency domain. The time complexity of these algorithms is quite high. Due to the great success of deep neural networks (DNNs) in speech processing and in order to make the front-end synchronizer compatible with back-end speech enhancer, in this work we first propose a lightweight dual-channel end-to-end SRO compensation network (SyncNet), which performs implicit SRO estimation and waveform reconstruction in a unified framework. This model can achieve performance comparable to conventional algorithms while incurring significantly lower latency, highlighting the strong potential of DNNs for SRO mitigation. We further extend this baseline model to a joint synchronization--enhancement network (SAENet) for DMAs. Experimental results show that SAENet consistently delivers superior speech enhancement performance in multiple conditions over two-stage approaches. 
# SyncNet
![https://github.com/MirMagchw/SAENet/blob/main/SyncNet/SyncNet.jpg](https://github.com/MirMagchw/SAENet/blob/main/SyncNet/SyncNet.jpg)
# SAENet
![https://github.com/MirMagchw/SAENet/edit/main/SAENet/SAENet.jpg](https://github.com/MirMagchw/SAENet/blob/main/SAENet/SAENet.jpg)
Packages:
```bash
torch                    2.4.1+cu118
torchaudio               2.4.1+cu118
torchvision              0.19.1+cu118
```
