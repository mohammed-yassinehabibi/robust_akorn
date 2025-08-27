## Donwload the sudoku datasets
```
cd data
bash download_satnet.sh
bash download_rrn.sh
cd ..
```

## Training 
### AKOrN
```
python train_sudoku.py --exp_name=sudoku_akorn  --eval_freq=10 --epochs=100 --model=akorn --lr=0.001 --T=16 --use_omega=True --global_omg=True --init_omg=0.5 --learn_omg=True 
```

### ItrSA and Transformer
```
# ItrSA
python train_sudoku.py --exp_name=sudoku_itrsa  --eval_freq=10 --epochs=100 --model=itrsa --lr=0.0005 --T=16
# Transformer
python train_sudoku.py --exp_name=sudoku_itrsa  --eval_freq=10 --epochs=100 --model=itrsa --lr=0.0005 --T=1 --L=8
```

## Evaluation

###  Inference with the test-time extension of the Kuramoto updates($T_{\rm eval}=128$) .
```
export data=ood # id or ood
python eval_sudoku.py --data=${data} --model=akorn --model_path=runs/sudoku_akorn/ema_99.pth --T=128 
```

###  Test-time extension of the K-updates + Energy-based voting ($T_{\rm eval}=128, (num\ random\ samples)=100$ ) .
```
python eval_sudoku.py --data=${data} --model=akorn --model_path=runs/sudoku_akorn/ema_99.pth --T=128 --K=100 --evote_type=sum 
``` 


### Performance table
| Model                   | ID | OOD |
|-------------------------|--------------------|-----------------|
| Transformer                | 98.6±0.3       | 5.2±0.2       |
| ItrSA ($T_{\rm eval}=32$)           | 95.7±8.5         | 34.4±5.4  |
| AKOrN ($T_{\rm eval}=128$)             | 100.0±0.0  | 51.7±3.3  |
| AKOrN ($T_{\rm eval}=128, K=100$)             | 100.0±0.0  | 81.6±1.5 |
| AKOrN ($T_{\rm eval}=128, K=4096$)             | 100.0±0.0  | 89.5±2.5 |

### Visualization of oscillator dynamics over timesteps
![sudoku](https://github.com/user-attachments/assets/97f9e6ed-0667-40c9-93a8-c45b5886b43b)

