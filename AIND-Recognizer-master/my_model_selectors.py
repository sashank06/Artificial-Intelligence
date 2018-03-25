import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    #Calculation with the help of references from udacity forums : NUmber of Parameters for BIC Calculation

    '''If we develop the HMM using the GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths) from hmmlearn we are calculating the following parameters that are the ones we use in BIC:

        Initial state occupation probabilities = numStates

        Transition probabilities = numStates*(numStates - 1)

        Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars

        numMeans and numCovars are the number of means and covars calculated. One mean and covar for each state and features. Then the total number of parameters are:

        Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities
        -------------------------
        https://discussions.udacity.com/t/parameter-in-bic-selector/394318/2
        Now take a look at this attributes definition for the hmmlearn library: https://github.com/hmmlearn/hmmlearn/blob/master/hmmlearn/hmm.py#L1061
        Transition probs are the transmat array, which is n_components X n_components, but since we know they add to 1.0, the last row can be calculated from the others, so the final tally for learned parameters here is n_components*(n_components-1)
        Starting probabilities are the startprob array and are learned. This is size n_components, but since they add up to 1.0, we only consider the number of free parameters to be n_components - 1
        the number of means is n_components*n_features
        variances are the size of the covars array, which for a “diag” like we are using is also n_components*n_features
        => add them up and you get n_components*n_components + 2*n_components*n_features - 1
        ftp://metron.sta.uniroma1.it/RePEc/articoli/2002-LX-3_4-11.pdf
        the SelectorBIC best component should be selected from the component that can return lowest score.'''


    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # TODO implement model selection based on BIC scores

        best_score = float("inf")
        best_model = None
        for num_components in range(self.min_n_components,self.max_n_components + 1):
            try:
                new_model = self.base_model(num_components) # create a new Gaussian HMM model
                logL  = new_model.score(self.X,self.lengths)
                logN = np.log(self.X.shape[0])
                features = self.X.shape[1]
                #p = num_components + num_components * (num_components - 1) + num_components * features * 2
                p = num_components * num_components  + (num_components * features * 2) - 1
                BIC = -2 * logL + p * logN
                if BIC < best_score:
                    best_score = BIC
                    best_model = new_model
            except:
                pass
        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # TODO implement model selection based on DIC scores
        best_score = float('-inf')
        best_model = None

        for n_components in range(self.min_n_components, self.max_n_components+1):
            log_likelihood = float('-inf')
            other_likelihood = []

            try:
                new_model = self.base_model(n_components)
                log_likelihood = new_model.score(self.X, self.lengths)
            except:
                pass

            for word in self.hwords:
                if word != self.this_word:
                    X, lengths = self.hwords[word]
                try:
                    other_likelihood.append(new_model.score(X, lengths))
                except:
                    pass

            DIC = log_likelihood - np.average(other_likelihood)
            
            if  DIC > best_score:
                best_score = DIC
                best_model = new_model

        return best_model







class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        #https://discussions.udacity.com/t/selectorcv-fails-with-indexerror-list-index-out-of-range/397820/4
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # TODO implement model selection using CV
        best_score = float("-inf")
        best_model = None
        n_splits = 3
        if len(self.sequences) < 2:
            return None
        elif len(self.sequences) == 2 :
            n_splits = 2
        
        for num_components in range(self.min_n_components,self.max_n_components + 1):
            split_method = KFold(n_splits=n_splits)
            scores = []
            log_likelihood = []

            for cv_train,cv_test in split_method.split(self.sequences):
                try:
                    X_train,lengths_train = combine_sequences(cv_train,self.sequences)
                    X_test,  lengths_test  = combine_sequences(cv_test, self.sequences)
                    new_model = self.base_model(num_components)
                    self.X = X_train
                    self.lengths = lengths_train
                    trained_model = new_model.fit(self.X,self.lengths)
                    #log_likelihood = new_model.score(X_test,lengths_test)
                    scores.append(trained_model.score(X_test,lengths_test))
                except:
                    pass

            if len(scores) > 0:
                new_score = np.mean(scores)
                if new_score > best_score:
                    best_score = new_score
                    best_model = new_model
        return best_model

