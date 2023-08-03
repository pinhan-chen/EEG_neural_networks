# EEG Data Analysis with Neural Networks

## Download required dataset

```console
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zVQ3SceLutLkzWk6uPIKwFgoNH0Fa63l' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zVQ3SceLutLkzWk6uPIKwFgoNH0Fa63l" -O data.zip && rm -rf /tmp/cookies.txt
```

### Unzip data.zip

```console
unzip data.zip
```

Then you should get a data folder including the required data files.

## Run the model

Open main.ipynb and make sure that you can access your data.

Then you can get the final result of the analysis.

