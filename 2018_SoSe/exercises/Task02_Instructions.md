Dear students,

the task2 will focus on basic fact-checking. It will consist in training 3 different models to perform over 3 different datasets (technically 2 different datasets + 1 merged)

#### The datasets

dataset1: fake_or_real_news.csv.zip

- **Column 1: the title.**
- **Column 2: the statement.**
- **Column 3: the label. [fake, real]**

dataset2 = liar_liar_paints_on_fire.zip

- Column 1: the ID of the statement ([ID].json).
- **Column 2: the label. [half-true, false, mostly-true, barely-true, true, pants-fire]**
- **Column 3: the statement.**
- Column 4: the subject(s).
- Column 5: the speaker.
- Column 6: the speaker's job title.
- Column 7: the state info.
- Column 8: the party affiliation.
- Column 9-13: the total credit history count, including the current statement.
	- 9: barely true counts.
	- 10: false counts.
	- 11: half true counts.
	- 12: mostly true counts.
	- 13: pants on fire counts.
- Column 14: the context (venue / location of the speech or statement).

(Most important columns - in general - are in bold)

dataset3 = [ds1' and ds2'](https://github.com/SmartDataAnalytics/MA-INF-4222-NLP-Lab/blob/master/2018_SoSe/exercises/task2_datasets1_2.zip)

I just created a [script](https://github.com/SmartDataAnalytics/MA-INF-4222-NLP-Lab/blob/master/2018_SoSe/exercises/script_dataset3.py) that returns to you the train and test splits for dataset3 respecting the scikit-learn interface. 
(please let me know if you find a bug, it might happen since I just coded this to make your life easier :))

#### The configurations 

- configuration 1) model a => dataset1 (train / test split)
- configuration 2) model b => dataset2 (train / validation / test split) *you can use all dataset columns here*
- configuration 3) model a => dataset2 | model2 => dataset1 (test) 
- configuration 4) model c => dataset3 (train / test split)

The following performance measures should be reported => [precision, recall, f-mesaure, accuracy]

Therefore, your outcomes will be as follows:

- configuration 1
	- model a - train - [performance measures][0:4]
	- model a - test - [performance measures][0:4]
- configuration 2
	- model b - train - [performance measures]
	- model b - validation - [performance measures]
	- model b - test - [performance measures]
- configuration 3 (no train!)
	- model a - test - dataset2 - [performance measures]
	- model b - test - dataset1 - [performance measures]

- configuration 4
	- model c - train - [performance measures]
	- model c - test - [performance measures]

note2: when reporting your performance measures for configuration 1 and configuration 4, consider => K-fold cross-validation, k=5, train=0.75, random_state = 4222 (that is very important in order to achieve reproducibility!!!)

#### The baseline

a. Based on the code below (credits to Katharine), implement your classifier (e.g. Random Forest, NN, SVM or ...) using scikit-learn   
https://github.com/kjam/random_hackery/blob/master/Attempting%20to%20detect%20fake%20news.ipynb
note: A datacamp tutorial of this code is also available at: https://www.datacamp.com/community/tutorials/scikit-learn-fake-news

#### Your model

b. Adapt (add/modify) the baseline by adding/modifying new/the features

#### Exporting your results (benchmarking)

Export your metadata following the [MEX vocabulary](https://github.com/METArchive/mex-vocabulary). An example of file can be found [here](https://github.com/METArchive/examples).

Within the next couple of days I will send an exact template for this file considering this task, in case you’re not that familiar with this kind of files.

note: 

- If you want to export directly from Python, there is this library that is useful [RDFLib](https://github.com/RDFLib/rdflib)
- Also, an example of a tool that uses this RDFLib to generate the graph (ttl, rdf, json-ld, etc..) metadata (in another not related format though, just as an example) can be found [here](https://github.com/NLP2RDF/pyNIF-lib)

All clear? :)

*if you have questions, please either open an issue on GitHub or send an email to this list, since it might affect other students too*

Thanks!
Diego.

— 
NLP Labs - MA-INF-4222
University of Bonn
[please give us a star :-)](https://github.com/SmartDataAnalytics/MA-INF-4222-NLP-Lab/stargazers)

   
