# Final Task - NLP Lab 
**Tools Used** 
- MIT Information Extraction Lib that is based on dlib - a high-performance machine-learning library[1]. MITIE makes use of several state-of-the-art techniques including the use of distributional word embeddings[2] and Structural Support Vector Machines[3]. MITIE offers several pre-trained models providing varying levels of support for English trained using a variety of linguistic resources (e.g., CoNLL 2003, ACE, Wikipedia, Freebase, and Gigaword).

- A Multi-task Approach for Named Entity Recognition on Social Media Data [4]. The system uses a Multi-task Neural Network as a feature extractor.           

Model from MITIE & MTA for NER were trained based on WNUT17 datasets provided [here](http://noisy-text.github.io/2017/emerging-rare-entities.html). I have created [this](https://github.com/khattaksaad/lab_final_task/blob/master/Final_Report.ipynb) file to report my results based on evaluation dataset provided by W-NUT17 team on their website. Although results are not pretty fine, but can be found in the report. For recognition purpose, I only used PERSON, ORGANIZATION & LOCATION to train both the models and then used only these three for reporting of my results as well. Overall accuracy and precision+recall has been reported for each category and are presented in form of a graph at end of the report. 












## References

[1] Davis E. King. Dlib-ml: A Machine Learning Toolkit. Journal of Machine Learning Research 10, pp. 1755-1758, 2009.

[2] Paramveer Dhillon, Dean Foster and Lyle Ungar, Eigenwords: Spectral Word Embeddings, Journal of Machine Learning Research (JMLR), 16, 2015.

[3] T. Joachims, T. Finley, Chun-Nam Yu, Cutting-Plane Training of Structural SVMs, Machine Learning, 77(1):27-59, 2009.

[4] Aguilar, S. Maharjan, A.P. Lopez-Monroy and T. Solorio A Multi-task Approach for Named Entity Recognition in Social Media Data

[5] [2017 The 3rd Workshop on Noisy User-generated Text (W-NUT)](http://noisy-text.github.io/2017/) 
