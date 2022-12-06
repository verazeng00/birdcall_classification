# Bird Classification
## Kaggle
[[Cornel Birdcall Identification]](https://www.kaggle.com/competitions/birdsong-recognition/overview)

## Dataset
https://www.kaggle.com/competitions/birdsong-recognition/data
- Sample testdata: https://www.kaggle.com/datasets/shonenkov/birdcall-check


## Train Data Preprocessing
(`/kaggle/input/birdsong-recognition/train_audio/`)
- Resample
- Convert time into 5 minutes: cut or pad
- Convert it into 3 channels
- Data augmentation: time shift
- Create a spectrogram
                # 3 = channel
                # 224: number of mels
                # 313 = sample_rate / hop_len * time = sample_rate / (n_fft/2) * time

- Add augmentation into spectrogram

## Test Data Preprocessing 
(`/kaggle/input/birdsong-recognition/test_audio/`)
- Based on row_id, get part of the test_csv with the same row_id
- On [[test.csv]](`/kaggle/input/birdcall-check/test.csv`), it has a colmn named `site`, whose value = `site_1`, `site_2` or `site_3`
- If `site` == `site_1` or `site_2`, for a same `audio_id`, it has multiple rows. It has a column named `seconds`, which is the end time of this `audio_id` and start time = end time -5s
- If `site` == `site_3`, every `audio_id`, it only has one row, which represents the whole audio 

<b>Site3</b>
  - Resample
  - Cut time as 5 s

      If <5s: Pad time

      Else: Cut the time based on start_time:end_time
  - Add it into 3 channels
  - Convert it into spectro gram >> the shape would be batch, 3, 224, 313
  - Put it into dataloader
  - When test: unsqueeze(0) >> get rid of batch size 
  - Build an inner_batch for a batch, because it includes mutiple 5s short mp3 

<b>Site2, Site1</b>
- Resample
- Convert it into 3 channels
- Spectrogram
- It would be 224, 313
- Put it into dataloader, batch_size = 1
- When it is on test, if shape[0] > 16, make it into iter

## Platform
Kaggle 

## Script



## Refer
1. https://www.kaggle.com/code/hidehisaarai1213/inference-pytorch-birdcall-resnet-baseline#Data-Loading
