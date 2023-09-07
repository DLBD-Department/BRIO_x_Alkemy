# BRIOxAlkemy
BRIOxAlkemy is the outcome of the partnership between [Alkemy](https://www.alkemy.com/en) and [BRIO](https://sites.unimi.it/brio/). The Alkemy's team responsible for this project is the Deep Learning and Big Data Department (DL&BD), an internal innovation team of the Alkemy group. 

## Motivation and use case
The tool is thought to be used for inspecting machine learning systems that could be affected by biases in their predictions. 

A typical scenario where the tool can help is the following: consider a database containing details of individuals, with their age, gender, and level of education. Consider an algorithm which tries to predict whether each of them is likely to default on credit. The user wishes to check if age is a sensitive factor in such prediction. The user feeds the tool our dataset, the output of the run of the predictive algorithm, and mark the feature of age as sensitive. Currently the tool allows the user to compare either how the behaviour of the algorithm with respect to age differs from an ``optimal'' behaviour (in this case, the user might consider optimal the case where each age group equally succeeds), or how different age groups perform with respect to one another.

These two analyses take the names of FreqVsRef and FreqVsFreq, described in a section below. 

## Usage via frontend
The tool can be used through a web browser interacting with the provided frontend. It comes with a Makefile that allows to easily build and run the Docker image that encapsulate the code. 

Provided that you have Docker up and running, to build the application (needed only the first time the tool is used) run:
- `make build`
To run the application:
- `make frontend`
Using your preferred web browser, navigate to `localhost:5000` in order to access the tool frontend.

To stop the application, run:
- `make stop`

## Usage via python library
The main functionalities of the tools are also available as python library, named `brio`. You can install it via pip, doing `pip install brio`. The bias detection analyses can be performed directly using the `FreqVsRefBiasDetector` and `FreqVsFreqBiasDetector` classes' interfaces. 

## Focus on the bias detection analyses
### FreqVsRef Analysis
This analysis performs a comparison between the behaviour of the AI system and a target desirable behaviour, expressed as probability distribution. 

This analysis is implemented in the `FreqVsRefBiasDetector` class, in the `bias` sub-module. 

The method `compare_root_variable_groups` computes the distance, in terms of normalized KL divergence, of the observed distribution of `target_variable` conditioned to `root_variable` (the sensitive feature) with respect to a reference distribution (the "Ref" of the analysis name) that is passed as parameter. The distance is compared with a threshold that represents a "safety" limit: distances bigger than the thresolds are considered signs of potential biases. The output is a tuple of three elements:
- a list with the distances of each category frequency of the sensitive variable from the expected reference
- a list of booleans, with the results of the comparison of the previous point's list with the provided threshold (distances < thresold)
- the thresold (provided as input or computed by the tool).

The method `compare_root_variable_conditioned_groups` performs the same calculation described above but for each sub-group implied by the Cartesian product of the categories of `conditioning_variables`, a list of available features present in dataframe and selected by the user. The computation is performed only if the sub-groups are composed of at least `min_obs_per_group` observations, with a default of 30. The output is, for each sub-group, a tuple of four elements:
- number of observations of the group
- a list with the distances of each category frequency of the sensitive variable from the expected reference
- a list of booleans, with the results of the comparison of the previous point's list with the provided threshold (distances < thresold)
- the thresold (provided as input or computed by the tool).

### FreqVsFreq Analysis
This analysis performs a comparison between the behaviour of the AI system with respect to a sensitive class and the behaviour of the AI system with respect to another sensitive class related to the same sensitive feature. In case of multi-class sensitive features, the results of the comparisons are aggregated using an aggregation function selected by the user. 

This analysis is implemented in the `FreqVsFreqBiasDetector` class, in the `bias` sub-module.

The method `compute_distance_between_frequencies` computes the JS divergence or the TV distance as selected for the `observed_distribution`, an array with the distribution of the `target_variable` conditioned to `root_variable` (the sensitive feature). The final value is provided using the selected aggregation function, relevant in case of multi-classes root variables.

The method `compare_root_variable_groups` computes the mutual distance, in terms of JS or TVD, of the categories of the observed distribution of `target_variable` conditioned to `root_variable`.

The method `compare_root_variable_conditioned_groups` performs the same calculation described above but for each sub-group implied by the Cartesian product of the categories of `conditioning_variables`, with the same constraints as for the `FreqsVsRefBiasDetector` class.

For both methods the output is a tuple similar to the ones described for the FreqsVsRef method, but with an additional element given by the standard deviation of the distances, provided only when the root variable is a multi-class feature. 

## What's next
Currently (September 2023) we plan to implement functionalities for the Opacity section, which is now empty. Furthermore, we want to introduce a risk measurement analysis, which will provide an overall risk assessment of a model using a series of bias and opacity checks.

## Call to action!
We hope to raise interest in the data science community and ask for support! Anyone interested in expanding and improving our tool is more than welcome! You can do that opening a pull request for a functionality you wish to include. Also bugs warnings are very important and welcome. 

Another way to cooperate with us is getting in touch with our team: just send an e-mail to DLBDDepartment@alkemy.com with your ideas, proposals or suggestions. 