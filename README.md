# Kaggle-Bikes
[Trello Board](https://trello.com/b/EZbPiQSd/kaggle)

# Rough Notes
Model assesment, we use MAE since this is a regresssion problem. - Maybe have a think about other metrics?
## Phase 1 - Comparing single vs all stations
- Use same train/val split on each stations data when combining all the stations into one model so that they are comparable.





# Assignment
### Phase 1:

In this phase, you will be given the data of 75 stations (Station 201 to Station 275) for the period of one month. The task is to predict availability for each station for the next 3 months.

There are two approaches for this task

- Train a separate model for each station. 
- Train a single model on all stations together.

Implement your models based on both approaches and check which approach is better. Investigate and discuss the results.

(The training data is given by Train.zip, the test data is given by test.csv).

(Build your models and submit the predictions according to the format given by example_leaderboard_submission.csv).
### Phase 2:

Now you will be given a set of linear models trained on other stations (Station 1 to Station 200) with the training data from a whole year. Although these models are not trained on the stations to be predicted, they can still be used since there should be some similarity among different stations. To successfully use these models can help reuse the knowledge learned from a whole year's data.


The task then is to figure out how to predict the stations in Phase 1 by only using these trained models. Investigate the resulting performances and compare to your own classifiers in Phase 1.

(The pre-trained linear models are given by Models.zip).
### Phase 3:

Try to achieve an even better performance by designing a approach to combine your own models with the given linear models.
