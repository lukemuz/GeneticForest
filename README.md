# GeneticForest
This module fits an ensemble machine learning model optimized by a genetic algorithm.  This is an experimental attempt to improve upon other ensemble models like RandomForest and Gradient Boosted Machines.  
## Model Description
GeneticForest fits an ensemble as a linear combination of tree models.  This is essentially the same model structure as both RandomForest (with weights all equal to 1) and Gradient Boosted Machines (with the trees fit sequentially on the gradient of the loss function).  

GeneticForest is initialized by selecting subsets of the columns of the input data and building small (max_depth < 8) trees on those subsets.  A linear model with Elastic Net regularization stacks these trees and provides weights to the linear combination.

After initialization, the subsequent "generation" of trees is selected by "mating" pairs of trees.  Selection occurs randomly, weighted by the coefficients of the linear model.  Mating produces a tree by randomly selecting columns used in each tree.  This generation of trees is again stacked with a linear model with Elastic Net regularization.  This is repeated for a specified number of generations.

## Results
So far, this model has only been tested on the Numerai dataset.  This produces fine results, but falls short of the XgBoost benchmark model produced by the Numerai team.  More testing and tweaking of the algorithm is needed, along with tests on more types of datasets to determine whether this is a fruitful path of research.  

## Implementation
This model is implemented entirely using Scikit-Learn.  Specifically, it leverages StackingRegressor from the ensemble module, and DecisionTreeRegressor from the tree module.  Currently, the module has not been generalized to other base learners or stacking types. 

## Possible Improvements and Future Research
Currently, the optimization focuses on which combinations of data elements from X produce a tree with a large contribution (coefficient) in the stacked model.  This explores these data combinations well, but unlike in gradient boosting, trees are built independent of one another, and this creates a lot of redundancy in the model.  Perhaps this could be improved with a hybrid approach of fixing a subset of trees from one generation to the next and fitting additional trees on pseudo-residuals (in the spirit of gradient boosting). 

Another improvement is computational speed.  Sklearn's DecisionTreeRegressor is really not designed for maximizing efficient use in a large ensemble model.  Using the tree engine from Catboost, for example, could greatly improve speed of the model, and would allow iteration on ideas.

Another possible improvement is to set up the optimization on combinations of splits instead of on combinations of predictors.  


