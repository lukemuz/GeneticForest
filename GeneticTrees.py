from sklearn import tree, ensemble,linear_model
import numpy as np
class GeneticTreeRegressor(tree.DecisionTreeRegressor):
    def __init__(self, *,col_list,
                 criterion="mse",
                 splitter="random",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 ccp_alpha=0.0):

        self.col_list=col_list

        super().__init__(criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)
        
    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted="deprecated"):
        X_sub=X[self.col_list]
        super().fit(
            X_sub, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)
        return self

    def predict(self, X, check_input=True):
        X_sub=X[self.col_list]
        return(super().predict(X_sub,check_input=True))

from random import sample, choices,uniform

def choices_wo_replacement(pop,k,weights):
    out_list=[]
    i=0
    while i<k:
        selection=choices(pop,weights=weights,k=1)
        selection=selection[0]
        if selection in out_list:
            selection=choices(pop,weights,k=1)
            selection=selection[0]
        else:
            out_list.append(selection)
            i=i+1
    return(out_list)


class GeneticForest():
    def __init__(self,base_estimator=GeneticTreeRegressor,
    stacking_estimator=ensemble.StackingRegressor):
        self.base_estimator=base_estimator
        self.stacking_estimator=stacking_estimator
        self.fitted_model=None

    def initial_generation(self,X,y,n_cols=5,max_depth=3,n_estimators=10):
        estimator_list=[]
        ##generate trees
        for i in range(n_estimators):
            ##sample columns
            all_cols=list(X.columns)
            s=sample(population=all_cols,k=n_cols)
            est_name=str(i)

            est=self.base_estimator(max_depth=max_depth,col_list=s)
            estimator_list.append((est_name,est))
        
        #create ensemble model
        stacker=self.stacking_estimator(estimator_list,n_jobs=-1,cv=2,final_estimator=linear_model.ElasticNetCV())
        stacker.fit(X,y)
        return(stacker)
    
    def next_generation(self,stacked_model,X,y,n_cols=5,max_depth=3,mutation_rate=.1,n_estimators=10):
        w_trees=stacked_model.final_estimator_.coef_
        w_trees=w_trees+.001 ##add small weight to every tree

        all_cols=list(X.columns)

        def mate(est1,est2,mutation_rate,all_cols):
            m=uniform(0,1)
            if m>mutation_rate:
                total_cols=stacked_model.estimators_[est1].col_list+ stacked_model.estimators_[est2].col_list
            else:
                random_cols=sample(population=all_cols,k=n_cols)
                total_cols=stacked_model.estimators_[est1].col_list+random_cols
            col_list=choices_wo_replacement(pop=total_cols,k=n_cols,weights=np.ones(len(total_cols)))
            new_estimator=self.base_estimator(max_depth=max_depth,col_list=col_list)
            return(new_estimator)
        
        new_estimators=[]
        for i in range(n_estimators):
            ##select two estimators based on linear coefficient, mate them, and add to list
            int_estimators=[*range(len(w_trees))]
            estimator_couple=choices_wo_replacement(int_estimators,k=2,weights=w_trees)
            est1=int(estimator_couple[0])
            est2=int(estimator_couple[1])
            new_est=mate(est1,est2,mutation_rate,all_cols)
            new_est_name=str(i)
            new_estimators.append((new_est_name,new_est))
        
        new_stacked_model=self.stacking_estimator(new_estimators,n_jobs=-1,cv=2,final_estimator=linear_model.ElasticNetCV())
        new_stacked_model.fit(X,y)
        return(new_stacked_model)

    def fit(self,X,y,n_cols=5,max_depth=3,n_generations=3,n_estimators=10,mutation_rate=.1):
        print("Fitting initial generation")
        stacked_model=self.initial_generation(X,y,n_cols=n_cols,max_depth=max_depth,n_estimators=n_estimators)
        remaining_generations=n_generations-1
        while remaining_generations > 0:
            print("Remaining Generations: "+str(remaining_generations))
            stacked_model=self.next_generation(stacked_model,X,y,n_cols=n_cols,max_depth=max_depth,n_estimators=n_estimators,mutation_rate=mutation_rate)
            remaining_generations=remaining_generations-1
        self.fitted_model=stacked_model
        return(self)











