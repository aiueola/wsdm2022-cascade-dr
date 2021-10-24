## Doubly Robust Off-Policy Evaluation for Ranking Policies under the Cascade Behavior Model

---

### About
This repository contains the code for the synthetic experiment conducted in the paper Doubly Robust Off-Policy Evaluation for Ranking Policies under the Cascade Behavior Model by Haruka Kiyohara, Yuta Saito, Tatsuya Matsuhiro, Yusuke Narita, Nobuyuki Shimizu, and Yasuo Yamamoto, which has been accepted to WSDM2022.

### Dependencies
This repository supports Python 3.7 or newer.

- numpy==1.20.0
- pandas==1.2.1
- scikit-learn==0.24.1
- hydra==1.0.6

Cloning `obp` from the source is also required to run the code:
```bash
git clone https://github.com/st-tech/zr-obp
cd zr-obp
python setup.py install
```

### Running the code
To run the real-world experiment, navigate to the `src/` directory and run the following commands.
Make sure to connect path to the cloned `obp` directory to run the experiment.

(i) experiment for varying data size, with the fixed slate size.
```bash
python main.py setting=n_rounds
```

(ii), (iii) experiment for varying slate size and policy similarities, with the fixed data size.
```bash
python main.py
```
Once the code is finished executing, you can find the results (`squared_error.csv`, `relative_ee.csv`, `configuration.csv`) in `./logs/` directory. Lower value is better for squared error and relative estimation error (relative-ee).
