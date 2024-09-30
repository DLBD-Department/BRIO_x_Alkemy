Here is some additional data for the submission of the paper _Bias Amplification Chains in ML-based systems with an application to credit scoring_ to [BEWARE 2024](https://sites.google.com/view/beware2024).

It contains:
- a folder for the training and the test set;
- a folder of the output of our analyses;
- the function used to compute hazards.

FreqvRef tests are performed with Kullback-Leibler divergence with Laplacian smoothing, set against the automated threshold with high sensitivity.

FreqvFreq tests are performed both for target 'performance' and 'predicted performance' with Jensen-Shannon, aggregating function the mean, automated threshold with high sensitivity.

**Remark**. The way the tool is devised, when observing less than $30$ individual it returns a message saying that there are not enough observations to say anything.